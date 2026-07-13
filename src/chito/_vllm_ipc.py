"""Private single-GPU Ray runtime for vLLM CUDA IPC weight updates."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any


class VllmIpcRuntime:
    """Own one colocated Ray LLM actor and its CUDA IPC lifecycle."""

    def __init__(
        self,
        model: str,
        engine_kwargs: Mapping[str, object],
    ) -> None:
        import ray
        import torch
        from ray.util.placement_group import (
            placement_group,
            remove_placement_group,
        )
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy,
        )
        from vllm import LLM
        from vllm.config import WeightTransferConfig
        from vllm.distributed.weight_transfer.ipc_engine import (
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )
        from vllm.distributed.weight_transfer.packed_tensor import (
            DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        )

        if ray.is_initialized():
            raise RuntimeError(
                "IPC weight transfer requires an uninitialized Ray runtime"
            )
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise RuntimeError(
                "IPC weight transfer requires exactly one visible CUDA GPU; "
                f"found {visible_gpus}"
            )

        options = dict(engine_kwargs)
        self._validate_engine_options(options)

        class ColocatedLlm(LLM):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "0.4"
                os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
                os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
                super().__init__(*args, **kwargs)

        self._ray = ray
        self._torch = torch
        self._remove_placement_group = remove_placement_group
        self._send_args_type = IPCTrainerSendWeightsArgs
        self._ipc_engine = IPCWeightTransferEngine
        self._default_buffer_size = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self._placement_group: Any = None
        self._llm: Any = None
        self._owns_ray = False
        self._closed = False

        ray.init()
        self._owns_ray = True
        try:
            self._placement_group = placement_group([{"GPU": 1, "CPU": 0}])
            ray.get(self._placement_group.ready())
            colocated = PlacementGroupSchedulingStrategy(
                placement_group=self._placement_group,
                placement_group_capture_child_tasks=True,
            )
            self._llm = ray.remote(
                num_cpus=0,
                num_gpus=0,
                scheduling_strategy=colocated,
            )(ColocatedLlm).remote(
                model=model,
                tensor_parallel_size=1,
                distributed_executor_backend="ray",
                weight_transfer_config=WeightTransferConfig(backend="ipc"),
                **options,
            )
            ray.get(
                self._llm.init_weight_transfer_engine.remote(
                    dict(init_info={})
                )
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_engine_options(options: Mapping[str, object]) -> None:
        managed = {
            "distributed_executor_backend",
            "tensor_parallel_size",
            "weight_transfer_config",
        }
        provided = managed.intersection(options)
        if provided:
            name = sorted(provided)[0]
            raise ValueError(f"{name} is managed by IPC weight transfer")
        if options.get("load_format") == "dummy":
            raise ValueError("IPC weight transfer does not use dummy model loading")
        for name in (
            "data_parallel_size",
            "pipeline_parallel_size",
            "nnodes",
        ):
            if int(options.get(name, 1)) != 1:
                raise ValueError(
                    f"IPC weight transfer requires {name}=1 on one GPU"
                )

    async def generate(self, prompt: Any, sampling_params: Any) -> Any:
        outputs = await self._get(
            self._llm.generate.remote(
                prompt,
                sampling_params,
                use_tqdm=False,
            )
        )
        if len(outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of requests")
        return outputs[0]

    async def update_weights(
        self,
        named_weights: Sequence[tuple[str, Any]],
    ) -> None:
        buffer_size = max(
            self._default_buffer_size,
            max(
                int(weight.numel()) * int(weight.element_size())
                for _, weight in named_weights
            ),
        )
        await self._get(self._llm.sleep.remote(level=0, mode="wait"))
        await self._get(
            self._llm.start_weight_update.remote(
                is_checkpoint_format=True
            )
        )
        await asyncio.to_thread(
            self._send_weights,
            named_weights,
            buffer_size,
        )
        await self._get(self._llm.finish_weight_update.remote())
        await self._get(self._llm.wake_up.remote(tags=["scheduling"]))

    def _send_weights(
        self,
        named_weights: Sequence[tuple[str, Any]],
        buffer_size: int,
    ) -> None:
        device = named_weights[0][1].device
        send_args = self._send_args_type(
            send_mode="ray",
            llm_handle=self._llm,
            packed=True,
            packed_buffer_size_bytes=buffer_size,
        )
        with self._torch.cuda.device(device):
            self._torch.cuda.synchronize(device)
            self._ipc_engine.trainer_send_weights(
                iterator=iter(named_weights),
                trainer_args=send_args,
            )

    async def _get(self, object_ref: Any) -> Any:
        get_task = asyncio.create_task(
            asyncio.to_thread(self._ray.get, object_ref)
        )
        try:
            return await asyncio.shield(get_task)
        except asyncio.CancelledError as cancellation:
            try:
                await get_task
            except BaseException as operation_error:
                raise BaseExceptionGroup(
                    "Ray operation failed while it was cancelled",
                    [cancellation, operation_error],
                ) from cancellation
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._llm is not None:
                self._ray.kill(self._llm)
        finally:
            try:
                if self._placement_group is not None:
                    self._remove_placement_group(self._placement_group)
            finally:
                if self._owns_ray:
                    self._ray.shutdown()
