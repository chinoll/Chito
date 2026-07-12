"""Asynchronous GRPO rollout coordination."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import AsyncIterable
from dataclasses import replace
from enum import Enum, auto

from .errors import (
    EngineClosedError,
    InvalidRolloutGroupError,
    RolloutAlreadyStartedError,
    RolloutFailedError,
    RolloutNotStartedError,
    SourceExhaustedError,
)
from .models import (
    RolloutConfig,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
    TrainingBatch,
)
from .protocols import (
    GroupPostHook,
    InferenceBackend,
    RewardFunction,
    RolloutContext,
    RolloutWorkflow,
)


class _EngineState(Enum):
    NEW = auto()
    RUNNING = auto()
    EXHAUSTED = auto()
    FAILED = auto()
    CLOSING = auto()
    CLOSED = auto()


class RolloutEngine:
    """Runs complete fixed-size groups while training consumes ready batches.

    ``rollout`` registers one asynchronous prompt source and starts an owned
    background producer. It returns immediately; producer failures are surfaced
    by ``next_batch``.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        workflow: RolloutWorkflow,
        reward_function: RewardFunction,
        config: RolloutConfig,
        post_hook: GroupPostHook | None = None,
    ) -> None:
        self._backend = backend
        self._workflow = workflow
        self._reward_function = reward_function
        self._config = config
        self._post_hook = post_hook

        self._condition = asyncio.Condition()
        self._update_lock = asyncio.Lock()
        self._group_slots = asyncio.Semaphore(config.max_concurrent_groups)
        self._accepted: deque[RolloutGroup] = deque()

        self._state = _EngineState.NEW
        self._admission_open = True
        self._active_groups = 0
        self._policy_version = config.initial_policy_version
        self._failure: BaseException | None = None
        self._producer_task: asyncio.Task[None] | None = None

    async def rollout(self, prompts: AsyncIterable[RolloutPrompt]) -> None:
        """Start consuming one async prompt source in the background."""
        if not hasattr(prompts, "__aiter__"):
            raise TypeError("prompts must be an AsyncIterable[RolloutPrompt]")

        async with self._condition:
            self._raise_if_closed()
            if self._state is not _EngineState.NEW:
                raise RolloutAlreadyStartedError(
                    "rollout accepts exactly one prompt source"
                )
            self._state = _EngineState.RUNNING
            self._producer_task = asyncio.create_task(
                self._consume_source(prompts), name="chito-rollout-producer"
            )

    async def next_batch(self) -> TrainingBatch:
        """Wait for exactly ``train_batch_size`` accepted complete groups."""
        async with self._condition:
            while True:
                if self._state in (_EngineState.CLOSING, _EngineState.CLOSED):
                    raise EngineClosedError("rollout engine is closed")
                if self._failure is not None:
                    raise RolloutFailedError(self._failure) from self._failure
                if len(self._accepted) >= self._config.train_batch_size:
                    groups = tuple(
                        self._accepted.popleft()
                        for _ in range(self._config.train_batch_size)
                    )
                    self._condition.notify_all()
                    return TrainingBatch(groups=groups)
                if self._state is _EngineState.NEW:
                    raise RolloutNotStartedError("call rollout() before next_batch()")
                if self._state is _EngineState.EXHAUSTED:
                    raise SourceExhaustedError(tuple(self._accepted))
                await self._condition.wait()

    async def update_weights(self, update: object) -> int:
        """Drain active groups, atomically update the backend, and return version."""
        async with self._update_lock:
            async with self._condition:
                self._raise_if_closed()
                self._raise_if_failed()
                self._admission_open = False
                self._condition.notify_all()

                while self._active_groups > 0:
                    await self._condition.wait()
                    self._raise_if_closed()
                    self._raise_if_failed()

                new_policy_version = self._policy_version + 1

            try:
                await self._backend.update_weights(
                    update, new_policy_version=new_policy_version
                )
            except BaseException:
                await self._restore_admission_after_update()
                raise

            async with self._condition:
                self._policy_version = new_policy_version
                if self._state in (_EngineState.NEW, _EngineState.RUNNING):
                    self._admission_open = True
                self._condition.notify_all()
                return new_policy_version

    async def aclose(self) -> None:
        """Cancel internal work, wake all waiters, and close the backend once."""
        async with self._condition:
            if self._state is _EngineState.CLOSED:
                return
            if self._state is _EngineState.CLOSING:
                await self._condition.wait_for(
                    lambda: self._state is _EngineState.CLOSED
                )
                return

            self._state = _EngineState.CLOSING
            self._admission_open = False
            producer = self._producer_task
            self._condition.notify_all()

        if producer is not None and not producer.done():
            producer.cancel()
        if producer is not None:
            try:
                await producer
            except asyncio.CancelledError:
                pass

        try:
            async with self._update_lock:
                await self._backend.aclose()
        finally:
            async with self._condition:
                self._state = _EngineState.CLOSED
                self._condition.notify_all()

    async def _consume_source(
        self, prompts: AsyncIterable[RolloutPrompt]
    ) -> None:
        try:
            async with asyncio.TaskGroup() as group_tasks:
                async for prompt in prompts:
                    if not isinstance(prompt, RolloutPrompt):
                        raise TypeError("prompt source must yield RolloutPrompt values")
                    await self._group_slots.acquire()
                    group_tasks.create_task(self._run_group_in_slot(prompt))
        except asyncio.CancelledError:
            if not self._is_closing():
                await self._record_failure(
                    RuntimeError("rollout producer was cancelled unexpectedly")
                )
        except Exception as exc:
            await self._record_failure(self._first_exception(exc))
        else:
            async with self._condition:
                if self._state is _EngineState.RUNNING:
                    self._state = _EngineState.EXHAUSTED
                self._condition.notify_all()

    async def _run_group_in_slot(self, prompt: RolloutPrompt) -> None:
        try:
            await self._run_group(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._record_failure(self._first_exception(exc))
            raise
        finally:
            self._group_slots.release()

    async def _run_group(self, prompt: RolloutPrompt) -> None:
        policy_version = await self._admit_group()
        accepted_group: RolloutGroup | None = None
        try:
            samples = await self._run_samples(prompt, policy_version)
            group = RolloutGroup(
                prompt=prompt,
                samples=samples,
                policy_version=policy_version,
            )
            self._validate_group(group, prompt, policy_version)
            accepted_group = (
                await self._post_hook(group) if self._post_hook is not None else group
            )
            if accepted_group is not None:
                self._validate_group(accepted_group, prompt, policy_version)
        finally:
            await self._release_group()

        if accepted_group is not None:
            await self._publish_group(accepted_group)

    async def _run_samples(
        self, prompt: RolloutPrompt, policy_version: int
    ) -> tuple[RolloutSample, ...]:
        tasks: list[asyncio.Task[RolloutSample]] = []
        async with asyncio.TaskGroup() as sample_tasks:
            for sample_index in range(self._config.group_size):
                tasks.append(
                    sample_tasks.create_task(
                        self._run_rewarded_sample(
                            prompt, sample_index, policy_version
                        )
                    )
                )
        return tuple(task.result() for task in tasks)

    async def _run_rewarded_sample(
        self, prompt: RolloutPrompt, sample_index: int, policy_version: int
    ) -> RolloutSample:
        context = RolloutContext(
            backend=self._backend,
            sample_index=sample_index,
            policy_version=policy_version,
        )
        sample = await self._workflow.run(context, prompt)
        self._validate_sample(
            sample,
            prompt=prompt,
            sample_index=sample_index,
            policy_version=policy_version,
            require_reward=False,
        )
        if sample.reward is not None:
            raise InvalidRolloutGroupError(
                "workflow must return an unrewarded RolloutSample"
            )
        reward = float(await self._reward_function(prompt, sample))
        if not math.isfinite(reward):
            raise InvalidRolloutGroupError("reward function must return a finite value")
        return replace(sample, reward=reward)

    async def _admit_group(self) -> int:
        async with self._condition:
            while not self._admission_open:
                self._raise_if_closed()
                self._raise_if_failed()
                await self._condition.wait()
            self._raise_if_closed()
            self._raise_if_failed()
            self._active_groups += 1
            return self._policy_version

    async def _release_group(self) -> None:
        async with self._condition:
            self._active_groups -= 1
            if self._active_groups < 0:
                raise RuntimeError("active group count became negative")
            self._condition.notify_all()

    async def _publish_group(self, group: RolloutGroup) -> None:
        capacity = self._config.accepted_queue_capacity
        async with self._condition:
            while len(self._accepted) >= capacity:
                if self._failure is not None or self._is_closing():
                    return
                await self._condition.wait()
            if self._failure is not None or self._is_closing():
                return
            self._accepted.append(group)
            self._condition.notify_all()

    async def _record_failure(self, exc: BaseException) -> None:
        async with self._condition:
            if self._failure is None and not self._is_closing():
                self._failure = exc
                self._state = _EngineState.FAILED
            self._condition.notify_all()

    async def _restore_admission_after_update(self) -> None:
        async with self._condition:
            if self._state in (_EngineState.NEW, _EngineState.RUNNING):
                self._admission_open = True
            self._condition.notify_all()

    def _validate_group(
        self,
        group: RolloutGroup,
        prompt: RolloutPrompt,
        policy_version: int,
    ) -> None:
        if not isinstance(group, RolloutGroup):
            raise InvalidRolloutGroupError(
                "post_hook must return RolloutGroup or None"
            )
        if group.prompt != prompt:
            raise InvalidRolloutGroupError("post_hook changed prompt identity")
        if group.policy_version != policy_version:
            raise InvalidRolloutGroupError("post_hook changed group policy_version")
        if len(group.samples) != self._config.group_size:
            raise InvalidRolloutGroupError(
                "post_hook changed the fixed group sample count"
            )

        indices = sorted(sample.sample_index for sample in group.samples)
        if indices != list(range(self._config.group_size)):
            raise InvalidRolloutGroupError(
                "group sample_index values must exactly cover the fixed group"
            )
        for sample in group.samples:
            self._validate_sample(
                sample,
                prompt=prompt,
                sample_index=sample.sample_index,
                policy_version=policy_version,
                require_reward=True,
            )

    @staticmethod
    def _validate_sample(
        sample: RolloutSample,
        *,
        prompt: RolloutPrompt,
        sample_index: int,
        policy_version: int,
        require_reward: bool,
    ) -> None:
        if not isinstance(sample, RolloutSample):
            raise InvalidRolloutGroupError(
                "workflow must return a RolloutSample"
            )
        if sample.prompt_id != prompt.prompt_id:
            raise InvalidRolloutGroupError("sample prompt_id does not match prompt")
        if sample.sample_index != sample_index:
            raise InvalidRolloutGroupError("workflow changed sample_index")
        if sample.policy_version != policy_version:
            raise InvalidRolloutGroupError("sample policy_version is inconsistent")

        prompt_length = len(prompt.token_ids)
        if sample.token_ids[:prompt_length] != prompt.token_ids:
            raise InvalidRolloutGroupError("sample does not preserve exact prompt tokens")
        if any(sample.loss_mask[:prompt_length]):
            raise InvalidRolloutGroupError("prompt tokens must be masked from loss")
        if not any(sample.loss_mask[prompt_length:]):
            raise InvalidRolloutGroupError(
                "sample must contain at least one trainable token after the prompt"
            )
        if require_reward and sample.reward is None:
            raise InvalidRolloutGroupError("accepted samples must contain rewards")

    def _raise_if_closed(self) -> None:
        if self._state in (_EngineState.CLOSING, _EngineState.CLOSED):
            raise EngineClosedError("rollout engine is closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RolloutFailedError(self._failure) from self._failure

    def _is_closing(self) -> bool:
        return self._state in (_EngineState.CLOSING, _EngineState.CLOSED)

    @staticmethod
    def _first_exception(exc: BaseException) -> BaseException:
        if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            return RolloutEngine._first_exception(exc.exceptions[0])
        return exc
