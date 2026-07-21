"""The single Ray actor that owns dataset iteration, Workflow, and vLLM."""

from __future__ import annotations

import asyncio
import math
import random
import socket
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING

from .models import (
    RolloutConfig,
    RolloutGroup,
    RolloutPrompt,
    RolloutSample,
    TrainingBatch,
    TrainingSample,
)
from .protocols import RolloutContext, RolloutWorkflow

if TYPE_CHECKING:
    from .vllm_backend import VllmBackend


@dataclass(frozen=True, slots=True)
class _RayServiceAddress:
    address: str
    namespace: str
    actor_name: str


@dataclass(frozen=True, slots=True)
class _BatchRequest:
    rank: int
    world_size: int
    step_id: int


@dataclass(frozen=True, slots=True)
class _ServiceInfo:
    state: str
    policy_version: int
    world_size: int


@dataclass(frozen=True, slots=True)
class _NcclRendezvous:
    master_address: str
    master_port: int
    world_size: int
    trainer_rank: int = 0
    receiver_rank_offset: int = 1


@dataclass(frozen=True, slots=True)
class _WeightManifest:
    batch_id: int
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtype_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.batch_id < 0:
            raise ValueError("batch_id must be non-negative")
        if not self.names:
            raise ValueError("weight manifest must not be empty")
        if len(self.names) != len(self.shapes) or len(self.names) != len(
            self.dtype_names
        ):
            raise ValueError("weight manifest fields must have equal length")


@dataclass(frozen=True, slots=True)
class _WeightUpdateTicket:
    update_id: int
    batch_id: int
    new_policy_version: int


@dataclass(slots=True)
class _TrainingStep:
    batch_id: int
    policy_version: int
    samples: tuple[TrainingSample, ...]
    delivered_ranks: set[int]


@dataclass(slots=True)
class _PendingWeightUpdate:
    ticket: _WeightUpdateTicket
    receive_task: asyncio.Task[object]


class _ServiceState(Enum):
    STARTING = auto()
    RUNNING = auto()
    DRAINING = auto()
    UPDATING = auto()
    FAILED = auto()
    CLOSED = auto()


class _RolloutService:
    """One process owns all V1 rollout state and the inference runtime."""

    def __init__(
        self,
        dataset: Sequence[object],
        workflow: RolloutWorkflow,
        config: RolloutConfig,
        train_world_size: int,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset must not be empty")
        if not isinstance(workflow, RolloutWorkflow):
            raise TypeError("workflow does not implement RolloutWorkflow")
        global_count = config.global_sample_count(train_world_size)
        if global_count % config.group_size:
            raise ValueError(
                "per_device_train_batch_size * train_world_size must be "
                "divisible by group_size"
            )

        self._dataset = dataset
        self._workflow = workflow
        self._config = config
        self._train_world_size = train_world_size
        self._global_sample_count = global_count

        self._condition = asyncio.Condition()
        self._ready_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._state = _ServiceState.STARTING
        self._failure: BaseException | None = None
        self._backend: VllmBackend | None = None
        self._current_step: _TrainingStep | None = None
        self._current_step_task: asyncio.Task[_TrainingStep] | None = None
        self._next_step_task: asyncio.Task[_TrainingStep] | None = None
        self._channel_task: asyncio.Task[None] | None = None
        self._pending_update: _PendingWeightUpdate | None = None

        self._active_policy_version = config.initial_policy_version
        self._next_batch_id = 0
        self._next_update_id = 0
        self._last_updated_batch_id: int | None = None
        self._epoch = 0
        self._offset = 0
        self._order = self._make_order()

    async def ready(self) -> _ServiceInfo:
        """Initialize Workflow and vLLM once, then report service identity."""
        async with self._ready_lock:
            self._raise_if_unusable()
            if self._backend is None:
                try:
                    await self._initialize()
                except BaseException as exc:
                    failure = _first_exception(exc)
                    await self._record_failure(failure)
                    raise failure
            return _ServiceInfo(
                state=self._state.name,
                policy_version=self._active_policy_version,
                world_size=self._train_world_size,
            )

    async def next_batch(self, request: _BatchRequest) -> TrainingBatch:
        self._validate_batch_request(request)
        async with self._condition:
            while True:
                self._raise_if_unusable()
                step = self._current_step
                if step is None or request.step_id > step.batch_id:
                    await self._condition.wait()
                    continue
                if request.step_id < step.batch_id:
                    raise ValueError("requested batch has already been released")

                batch = self._slice_batch(step, request.rank)
                step.delivered_ranks.add(request.rank)
                self._condition.notify_all()
                return batch

    async def start_weight_channel(self) -> _NcclRendezvous:
        backend = self._require_backend()
        if self._channel_task is not None:
            raise RuntimeError("weight channel was already started")

        rendezvous = _NcclRendezvous(
            master_address="127.0.0.1",
            master_port=_find_free_local_port(),
            world_size=backend.inference_world_size + 1,
        )
        init_info = {
            "master_address": rendezvous.master_address,
            "master_port": rendezvous.master_port,
            "rank_offset": rendezvous.receiver_rank_offset,
            "world_size": rendezvous.world_size,
        }
        self._channel_task = asyncio.create_task(
            backend.init_weight_channel(init_info),
            name="chito-vllm-weight-channel",
        )
        return rendezvous

    async def finish_weight_channel(self) -> None:
        if self._channel_task is None:
            raise RuntimeError("weight channel has not been started")
        try:
            await self._channel_task
        except BaseException as exc:
            await self._record_failure(exc)
            raise
        await self._start_current_step()

    async def begin_weight_update(
        self, manifest: _WeightManifest
    ) -> _WeightUpdateTicket:
        async with self._condition:
            self._raise_if_unusable()
            step = self._require_current_step()
            if self._state is not _ServiceState.RUNNING:
                raise RuntimeError("rollout service is not ready for a weight update")
            if len(step.delivered_ranks) != self._train_world_size:
                raise RuntimeError("all training ranks must fetch the batch first")
            if manifest.batch_id != step.batch_id:
                raise ValueError("weight manifest does not match the current batch")
            if self._pending_update is not None:
                raise RuntimeError("a weight update is already pending")
            expected_batch_id = (
                0
                if self._last_updated_batch_id is None
                else self._last_updated_batch_id + 1
            )
            if step.batch_id != expected_batch_id:
                raise RuntimeError("weight updates must follow batch order exactly")
            self._state = _ServiceState.DRAINING
            next_step_task = self._next_step_task

        next_step: _TrainingStep | None = None
        if self._config.rollout_mode == "async":
            if next_step_task is None:
                failure = RuntimeError("async rollout has no prefetched next batch")
                await self._record_failure(failure)
                raise failure
            try:
                next_step = await asyncio.shield(next_step_task)
            except BaseException as exc:
                failure = _first_exception(exc)
                await self._record_failure(failure)
                raise failure

        async with self._condition:
            self._raise_if_unusable()
            if self._state is not _ServiceState.DRAINING:
                raise RuntimeError("rollout service left the draining state")
            if self._config.rollout_mode == "async":
                if self._next_step_task is not next_step_task:
                    raise RuntimeError("prefetched batch changed while draining")
                assert next_step is not None
                if next_step.batch_id != step.batch_id + 1:
                    raise RuntimeError("prefetched batch is out of order")
                if next_step.policy_version != self._active_policy_version:
                    raise RuntimeError("prefetched batch has the wrong policy version")
            self._state = _ServiceState.UPDATING
            ticket = _WeightUpdateTicket(
                update_id=self._next_update_id,
                batch_id=step.batch_id,
                new_policy_version=self._active_policy_version + 1,
            )
            self._next_update_id += 1

        update_info = {
            "names": list(manifest.names),
            "shapes": [list(shape) for shape in manifest.shapes],
            "dtype_names": list(manifest.dtype_names),
            "packed": False,
        }
        try:
            receive_task = await self._require_backend().begin_weight_update(
                update_info
            )
        except BaseException as exc:
            await self._record_failure(exc)
            raise
        async with self._condition:
            self._raise_if_unusable()
            if self._state is not _ServiceState.UPDATING:
                raise RuntimeError("rollout service left the updating state")
            self._pending_update = _PendingWeightUpdate(ticket, receive_task)
        return ticket

    async def finish_weight_update(self, ticket: _WeightUpdateTicket) -> int:
        async with self._condition:
            self._raise_if_unusable()
            pending = self._require_pending_update(ticket)
            if self._state is not _ServiceState.UPDATING:
                raise RuntimeError("rollout service is not updating")
            next_step = None
            if self._config.rollout_mode == "async":
                task = self._next_step_task
                if task is None or not task.done() or task.cancelled():
                    raise RuntimeError("prefetched batch is not ready for rotation")
                next_step = task.result()

        try:
            await self._require_backend().finish_weight_update(
                pending.receive_task,
                new_policy_version=ticket.new_policy_version,
            )
        except BaseException as exc:
            await self._record_failure(exc)
            raise

        async with self._condition:
            self._raise_if_unusable()
            self._require_pending_update(ticket)
            self._pending_update = None
            self._active_policy_version = ticket.new_policy_version
            self._last_updated_batch_id = ticket.batch_id

            if self._config.rollout_mode == "async":
                assert next_step is not None
                self._current_step = next_step
                self._next_step_task = None
                self._start_next_step_locked()
            else:
                self._current_step = None
                self._start_current_step_locked()

            self._state = _ServiceState.RUNNING
            self._condition.notify_all()
            return self._active_policy_version

    async def fail_weight_update(
        self, ticket: _WeightUpdateTicket, message: str
    ) -> None:
        async with self._condition:
            self._raise_if_unusable()
            self._require_pending_update(ticket)
        failure = RuntimeError(f"trainer NCCL sender failed: {message}")
        try:
            await self._require_backend().fail_weight_update(str(failure))
        except BaseException as exc:
            cleanup_failure = _first_exception(exc)
            failure.add_note(
                "backend weight-update cleanup failed: "
                f"{type(cleanup_failure).__name__}: {cleanup_failure}"
            )
        async with self._condition:
            self._pending_update = None
        await self._record_failure(failure)

    async def close(self) -> None:
        async with self._close_lock:
            if self._close_task is None:
                async with self._condition:
                    self._state = _ServiceState.CLOSED
                    tasks = tuple(
                        task
                        for task in (
                            self._current_step_task,
                            self._next_step_task,
                            self._channel_task,
                        )
                        if task is not None
                    )
                    self._condition.notify_all()
                self._close_task = asyncio.create_task(
                    self._close_resources(tasks), name="chito-rollout-service-close"
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_resources(self, tasks: tuple[asyncio.Task[object], ...]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await self._workflow.aclose()
        finally:
            if self._backend is not None:
                await self._backend.aclose()

    async def _initialize(self) -> None:
        from .vllm_backend import VllmBackend

        await self._workflow.setup()
        options = dict(self._config.backend_kwargs)
        if self._config.rollout_gpu_ids:
            options.setdefault(
                "tensor_parallel_size", len(self._config.rollout_gpu_ids)
            )
        self._backend = VllmBackend(
            self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            engine_kwargs=options,
            seed=self._config.seed,
            initial_policy_version=self._config.initial_policy_version,
        )
        if self._config.rollout_gpu_ids and (
            self._backend.inference_world_size != len(self._config.rollout_gpu_ids)
        ):
            raise ValueError("the vLLM parallel world size must match rollout_gpu_ids")

    async def _start_current_step(self) -> None:
        async with self._condition:
            self._raise_if_unusable()
            if self._state is not _ServiceState.STARTING:
                raise RuntimeError("initial rollout step was already started")
            self._state = _ServiceState.RUNNING
            self._start_current_step_locked()
            self._condition.notify_all()

    def _start_current_step_locked(self) -> None:
        if self._current_step_task is not None:
            raise RuntimeError("current rollout step is already running")
        batch_id = self._take_batch_id()
        self._current_step_task = asyncio.create_task(
            self._build_and_publish_current(batch_id, self._active_policy_version),
            name=f"chito-rollout-batch-{batch_id}",
        )

    def _start_next_step_locked(self) -> None:
        if self._next_step_task is not None:
            raise RuntimeError("next rollout step is already running")
        batch_id = self._take_batch_id()
        self._next_step_task = asyncio.create_task(
            self._build_step_safely(batch_id, self._active_policy_version),
            name=f"chito-rollout-prefetch-{batch_id}",
        )

    def _take_batch_id(self) -> int:
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return batch_id

    async def _build_and_publish_current(
        self, batch_id: int, policy_version: int
    ) -> _TrainingStep:
        step = await self._build_step_safely(batch_id, policy_version)
        try:
            async with self._condition:
                self._raise_if_unusable()
                if self._current_step_task is not asyncio.current_task():
                    raise RuntimeError(
                        "current rollout task changed before publication"
                    )
                self._current_step = step
                self._current_step_task = None
                if self._config.rollout_mode == "async":
                    self._start_next_step_locked()
                self._condition.notify_all()
            return step
        except BaseException as exc:
            failure = _first_exception(exc)
            await self._record_failure(failure)
            raise failure

    async def _build_step_safely(
        self, batch_id: int, policy_version: int
    ) -> _TrainingStep:
        try:
            return await self._build_step(batch_id, policy_version)
        except asyncio.CancelledError:
            if self._state is not _ServiceState.CLOSED:
                await self._record_failure(
                    RuntimeError(f"rollout batch {batch_id} was cancelled")
                )
            raise
        except BaseException as exc:
            failure = _first_exception(exc)
            await self._record_failure(failure)
            raise failure

    async def _build_step(self, batch_id: int, policy_version: int) -> _TrainingStep:
        target_group_count = self._global_sample_count // self._config.group_size
        accepted_groups: list[tuple[TrainingSample, ...]] = []
        in_flight: set[asyncio.Task[tuple[TrainingSample, ...] | None]] = set()
        done: set[asyncio.Task[tuple[TrainingSample, ...] | None]] = set()

        try:
            while len(accepted_groups) < target_group_count:
                while (
                    len(in_flight) < self._config.max_concurrent_groups
                    and len(accepted_groups) + len(in_flight) < target_group_count
                ):
                    item, item_id = self._next_item()
                    in_flight.add(
                        asyncio.create_task(
                            self._process_item(item, item_id, policy_version)
                        )
                    )

                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                first_failure: BaseException | None = None
                completed_groups: list[tuple[TrainingSample, ...]] = []
                for task in done:
                    try:
                        group_samples = task.result()
                    except BaseException as exc:
                        if first_failure is None:
                            first_failure = _first_exception(exc)
                    else:
                        if group_samples is not None:
                            completed_groups.append(group_samples)

                if first_failure is not None:
                    raise first_failure
                accepted_groups.extend(completed_groups)
                done = set()

            samples = tuple(sample for group in accepted_groups for sample in group)
            if len(samples) != self._global_sample_count:
                raise RuntimeError("rollout batch has the wrong sample count")
            return _TrainingStep(batch_id, policy_version, samples, set())
        except BaseException:
            siblings = done | in_flight
            for task in siblings:
                if not task.done():
                    task.cancel()
            if siblings:
                await asyncio.gather(*siblings, return_exceptions=True)
            raise

    async def _process_item(
        self, item: object, item_id: str, policy_version: int
    ) -> tuple[TrainingSample, ...] | None:
        prompt = await self._workflow.prepare(item, item_id)
        if not isinstance(prompt, RolloutPrompt):
            raise TypeError("workflow.prepare must return RolloutPrompt")

        group = await self._generate_group(prompt, policy_version)
        group = await self._workflow.postprocess(group)
        if group is None:
            return None
        self._validate_group(group, prompt, policy_version)

        advantages = tuple(await self._workflow.compute_advantages(group))
        if len(advantages) != self._config.group_size:
            raise ValueError("workflow returned the wrong advantage count")
        if any(not math.isfinite(float(value)) for value in advantages):
            raise ValueError("workflow advantages must be finite")

        training_samples = tuple(
            TrainingSample(
                prompt_id=sample.prompt_id,
                sample_index=sample.sample_index,
                token_ids=sample.token_ids,
                loss_mask=sample.loss_mask,
                behavior_logprobs=sample.behavior_logprobs,
                reward=float(sample.reward),
                advantage=float(advantage),
                policy_version=sample.policy_version,
                metadata=group.prompt.metadata,
            )
            for sample, advantage in zip(group.samples, advantages, strict=True)
        )
        return training_samples

    async def _generate_group(
        self, prompt: RolloutPrompt, policy_version: int
    ) -> RolloutGroup:
        sample_tasks: list[asyncio.Task[RolloutSample]] = []
        async with asyncio.TaskGroup() as tasks:
            for sample_index in range(self._config.group_size):
                context = RolloutContext(
                    backend=self._require_backend(),
                    sample_index=sample_index,
                    policy_version=policy_version,
                )
                sample_tasks.append(
                    tasks.create_task(self._workflow.run(context, prompt))
                )
        samples = tuple(task.result() for task in sample_tasks)
        for sample_index, sample in enumerate(samples):
            self._validate_sample(
                sample, prompt, sample_index, policy_version, rewarded=False
            )

        reward_tasks: list[asyncio.Task[float]] = []
        async with asyncio.TaskGroup() as tasks:
            for sample in samples:
                reward_tasks.append(
                    tasks.create_task(self._workflow.reward(prompt, sample))
                )
        rewarded = tuple(
            replace(sample, reward=float(task.result()))
            for sample, task in zip(samples, reward_tasks, strict=True)
        )
        group = RolloutGroup(prompt, rewarded, policy_version)
        self._validate_group(group, prompt, policy_version)
        return group

    def _next_item(self) -> tuple[object, str]:
        if self._offset == len(self._order):
            self._epoch += 1
            self._offset = 0
            self._order = self._make_order()
        index = self._order[self._offset]
        self._offset += 1
        return self._dataset[index], f"{self._epoch}:{index}"

    def _make_order(self) -> list[int]:
        order = list(range(len(self._dataset)))
        if self._config.shuffle:
            random.Random(self._config.seed + self._epoch).shuffle(order)
        return order

    def _slice_batch(self, step: _TrainingStep, rank: int) -> TrainingBatch:
        size = self._config.per_device_train_batch_size
        start = rank * size
        return TrainingBatch(
            batch_id=step.batch_id,
            policy_version=step.policy_version,
            samples=tuple(step.samples[start : start + size]),
            global_sample_count=self._global_sample_count,
        )

    def _validate_batch_request(self, request: _BatchRequest) -> None:
        if not isinstance(request, _BatchRequest):
            raise TypeError("request must be _BatchRequest")
        if request.world_size != self._train_world_size:
            raise ValueError("training world size changed after service startup")
        if not 0 <= request.rank < self._train_world_size:
            raise ValueError("rank is outside the training world")
        if request.step_id < 0:
            raise ValueError("step_id must be non-negative")

    def _validate_group(
        self,
        group: RolloutGroup,
        prompt: RolloutPrompt,
        policy_version: int,
    ) -> None:
        if not isinstance(group, RolloutGroup):
            raise TypeError("workflow.postprocess must return RolloutGroup or None")
        if group.prompt != prompt:
            raise ValueError("workflow.postprocess changed the prompt")
        if group.policy_version != policy_version:
            raise ValueError("workflow.postprocess changed policy_version")
        if len(group.samples) != self._config.group_size:
            raise ValueError("workflow.postprocess changed the fixed group size")
        indices = [sample.sample_index for sample in group.samples]
        if indices != list(range(self._config.group_size)):
            raise ValueError("group samples must keep their fixed index order")
        for sample in group.samples:
            self._validate_sample(
                sample,
                prompt,
                sample.sample_index,
                policy_version,
                rewarded=True,
            )

    @staticmethod
    def _validate_sample(
        sample: RolloutSample,
        prompt: RolloutPrompt,
        sample_index: int,
        policy_version: int,
        *,
        rewarded: bool,
    ) -> None:
        if not isinstance(sample, RolloutSample):
            raise TypeError("workflow.run must return RolloutSample")
        if sample.prompt_id != prompt.prompt_id:
            raise ValueError("sample prompt_id does not match its prompt")
        if sample.sample_index != sample_index:
            raise ValueError("workflow.run changed sample_index")
        if sample.policy_version != policy_version:
            raise ValueError("sample policy_version is inconsistent")
        prompt_length = len(prompt.token_ids)
        if sample.token_ids[:prompt_length] != prompt.token_ids:
            raise ValueError("sample did not preserve exact prompt tokens")
        if any(sample.loss_mask[:prompt_length]):
            raise ValueError("prompt tokens must be masked from loss")
        if sample.reward is not None and not rewarded:
            raise ValueError("workflow.run must return an unrewarded sample")
        if rewarded and sample.reward is None:
            raise ValueError("accepted samples must have rewards")

    async def _record_failure(self, failure: BaseException) -> None:
        async with self._condition:
            if self._failure is None and self._state is not _ServiceState.CLOSED:
                self._failure = failure
                self._state = _ServiceState.FAILED
            self._condition.notify_all()

    def _raise_if_unusable(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._state is _ServiceState.CLOSED:
            raise RuntimeError("rollout service is closed")

    def _require_backend(self) -> VllmBackend:
        if self._backend is None:
            raise RuntimeError("rollout service is not initialized")
        return self._backend

    def _require_current_step(self) -> _TrainingStep:
        if self._current_step is None:
            raise RuntimeError("rollout service has no active training step")
        return self._current_step

    def _require_pending_update(
        self, ticket: _WeightUpdateTicket
    ) -> _PendingWeightUpdate:
        pending = self._pending_update
        if pending is None or pending.ticket != ticket:
            raise ValueError("weight update ticket is not current")
        return pending


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _first_exception(error: BaseException) -> BaseException:
    if isinstance(error, BaseExceptionGroup) and error.exceptions:
        return _first_exception(error.exceptions[0])
    return error
