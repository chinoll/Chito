"""Training-process facade for the independent Ray rollout service."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from typing import Any

from .models import RolloutConfig, TrainingBatch
from .protocols import RolloutWorkflow
from .ray_service import (
    _BatchRequest,
    _NcclRendezvous,
    _RayServiceAddress,
    _RolloutService,
    _WeightManifest,
)


class RolloutEngine:
    """Connect training ranks to one rollout service started by rank 0.

    Construction is a training-process collective. Only rank 0 supplies the
    dataset and Workflow; other ranks may pass ``None`` for both.
    """

    def __init__(
        self,
        dataset: Sequence[object] | None,
        workflow: RolloutWorkflow | None,
        config: RolloutConfig,
    ) -> None:
        self._rank, self._world_size = _distributed_identity()
        self._config = config
        self._next_step_id = 0
        self._last_batch_id: int | None = None
        self._closed = False

        self._ray = _import_ray()
        self._sender: _NcclWeightSender | None = None
        address: _RayServiceAddress | None = None

        if self._rank == 0:
            if dataset is None or workflow is None:
                raise ValueError("rank 0 must provide dataset and workflow")
            address = self._start_service(dataset, workflow)

        address = _broadcast_service_address(address, self._rank, self._world_size)
        if self._rank != 0:
            self._ray.init(address=address.address, namespace=address.namespace)
            self._service = self._ray.get_actor(address.actor_name)

    async def next_batch(self) -> TrainingBatch:
        """Wait for the global buffer, then return this rank's contiguous shard."""
        self._raise_if_closed()
        request = _BatchRequest(
            rank=self._rank,
            world_size=self._world_size,
            step_id=self._next_step_id,
        )
        batch = await self._get(self._service.next_batch.remote(request))
        if not isinstance(batch, TrainingBatch):
            raise TypeError("rollout service returned an invalid TrainingBatch")
        self._next_step_id += 1
        self._last_batch_id = batch.batch_id
        return batch

    async def update_weights(self, model: Any) -> int:
        """Push rank 0's already-synchronized full parameters directly by NCCL."""
        self._raise_if_closed()
        if self._rank != 0:
            raise RuntimeError("only training rank 0 may update rollout weights")
        if self._last_batch_id is None:
            raise RuntimeError("call next_batch before update_weights")

        named_weights = _named_cuda_parameters(model)
        manifest = _make_manifest(self._last_batch_id, named_weights)
        ticket = await self._get(self._service.begin_weight_update.remote(manifest))
        try:
            assert self._sender is not None
            self._sender.send(named_weights)
        except BaseException as exc:
            await self._get(self._service.fail_weight_update.remote(ticket, str(exc)))
            raise
        return int(await self._get(self._service.finish_weight_update.remote(ticket)))

    async def aclose(self) -> None:
        """Close this client; rank 0 also closes the owned service and runtime."""
        if self._closed:
            return
        self._closed = True
        if self._rank != 0:
            self._ray.shutdown()
            return

        try:
            await self._get(self._service.close.remote())
        finally:
            if self._sender is not None:
                self._sender.close()
            self._ray.shutdown()

    def _start_service(
        self,
        dataset: Sequence[object],
        workflow: RolloutWorkflow,
    ) -> _RayServiceAddress:
        namespace = f"chito-{uuid.uuid4().hex}"
        actor_name = f"rollout-{uuid.uuid4().hex}"
        context = self._ray.init(namespace=namespace)
        cluster_address = str(context.address_info["address"])

        remote_service = self._ray.remote(
            num_cpus=1,
            max_concurrency=self._world_size + 2,
        )(_RolloutService)
        options: dict[str, object] = {"name": actor_name}
        if self._config.rollout_gpu_ids:
            options["runtime_env"] = {
                "env_vars": {
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        str(device) for device in self._config.rollout_gpu_ids
                    )
                }
            }
        self._service = remote_service.options(**options).remote(
            dataset,
            workflow,
            self._config,
            self._world_size,
        )
        self._ray.get(self._service.ready.remote())

        rendezvous = self._ray.get(self._service.start_weight_channel.remote())
        self._sender = _NcclWeightSender(rendezvous)
        self._ray.get(self._service.finish_weight_channel.remote())
        return _RayServiceAddress(cluster_address, namespace, actor_name)

    async def _get(self, reference: Any) -> Any:
        return await asyncio.to_thread(self._ray.get, reference)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("rollout engine is closed")


class _NcclWeightSender:
    """Synchronous proxy for the isolated trainer-side NCCL process."""

    def __init__(self, rendezvous: _NcclRendezvous) -> None:
        import torch
        import torch.multiprocessing as multiprocessing

        context = multiprocessing.get_context("spawn")
        connection, child_connection = context.Pipe()
        process = context.Process(
            target=_nccl_sender_main,
            args=(child_connection, rendezvous, torch.cuda.current_device()),
            name="chito-nccl-weight-sender",
            daemon=True,
        )
        self._connection = connection
        self._process = process
        self._usable = True
        self._closed = False

        process.start()
        child_connection.close()
        try:
            self._receive()
        except BaseException:
            connection.close()
            process.join()
            process.close()
            raise

    def send(self, named_weights: tuple[tuple[str, Any], ...]) -> None:
        if self._closed or not self._usable:
            raise RuntimeError("NCCL weight sender is closed")
        self._connection.send(("send", named_weights))
        self._receive()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._usable:
                self._connection.send(("close", None))
                self._receive()
        finally:
            self._connection.close()
            self._process.join()
            self._process.close()

    def _receive(self) -> None:
        try:
            error = self._connection.recv()
        except (EOFError, OSError) as exc:
            self._usable = False
            raise RuntimeError("NCCL weight sender process exited") from exc
        if error is not None:
            self._usable = False
            raise RuntimeError(f"NCCL weight sender failed: {error}")


def _nccl_sender_main(
    connection: Any,
    rendezvous: _NcclRendezvous,
    device_index: int,
) -> None:
    """Import vLLM and own its trainer-side NCCL objects in a child process."""
    group = None
    try:
        import torch

        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
        try:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLTrainerSendWeightsArgs,
                NCCLWeightTransferEngine,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "vllm":
                raise
            raise ModuleNotFoundError(
                "NCCL weight updates require chito[vllm]"
            ) from exc

        init_info = {
            "master_address": rendezvous.master_address,
            "master_port": rendezvous.master_port,
            "world_size": rendezvous.world_size,
        }
        with torch.cuda.device(device):
            group = NCCLWeightTransferEngine.trainer_init(init_info)
        connection.send(None)

        while True:
            command, payload = connection.recv()
            if command == "close":
                group.destroy()
                group = None
                connection.send(None)
                return

            named_weights = payload
            if named_weights[0][1].device != device:
                raise ValueError(
                    "rank 0 model parameters must be on the NCCL sender device"
                )
            with torch.cuda.device(device):
                NCCLWeightTransferEngine.trainer_send_weights(
                    iter(named_weights),
                    NCCLTrainerSendWeightsArgs(group=group, packed=False),
                )
            connection.send(None)
    except BaseException as exc:
        try:
            connection.send(f"{type(exc).__name__}: {exc}")
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if group is not None:
            try:
                group.destroy()
            except Exception:
                pass
        connection.close()


def _named_cuda_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is None:
        raise TypeError("model must provide named_parameters()")
    weights = tuple(named_parameters())
    if not weights:
        raise ValueError("model has no parameters")

    names = [name for name, _ in weights]
    if len(set(names)) != len(names):
        raise ValueError("model parameter names must be unique")
    first_device = weights[0][1].device
    if first_device.type != "cuda":
        raise ValueError("model parameters must be CUDA tensors")
    if any(weight.device != first_device for _, weight in weights):
        raise ValueError("all model parameters must be on one CUDA device")
    return tuple((name, weight.detach()) for name, weight in weights)


def _make_manifest(
    batch_id: int,
    named_weights: tuple[tuple[str, Any], ...],
) -> _WeightManifest:
    return _WeightManifest(
        batch_id=batch_id,
        names=tuple(name for name, _ in named_weights),
        shapes=tuple(tuple(weight.shape) for _, weight in named_weights),
        dtype_names=tuple(
            str(weight.dtype).removeprefix("torch.") for _, weight in named_weights
        ),
    )


def _distributed_identity() -> tuple[int, int]:
    import torch.distributed as distributed

    if not distributed.is_available() or not distributed.is_initialized():
        return 0, 1
    return distributed.get_rank(), distributed.get_world_size()


def _broadcast_service_address(
    address: _RayServiceAddress | None,
    rank: int,
    world_size: int,
) -> _RayServiceAddress:
    if world_size == 1:
        assert address is not None
        return address

    import torch
    import torch.distributed as distributed

    values = [address]
    device = None
    if distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    distributed.broadcast_object_list(values, src=0, device=device)
    result = values[0]
    if not isinstance(result, _RayServiceAddress):
        raise RuntimeError(f"rank {rank} received an invalid rollout service address")
    return result


def _import_ray() -> Any:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RolloutEngine requires Ray; install chito[vllm]"
        ) from exc
    return ray
