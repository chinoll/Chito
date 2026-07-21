"""Training-process facade for the independent Ray rollout service."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _ServiceStartupOutcome:
    address: _RayServiceAddress | None
    error_type: str | None = None
    error_message: str | None = None


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
        self._last_updated_batch_id: int | None = None
        self._next_batch_lock = asyncio.Lock()
        self._update_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

        self._ray: Any = None
        self._sender: _NcclWeightSender | None = None
        outcome: _ServiceStartupOutcome | None = None
        startup_cause: BaseException | None = None

        if self._rank == 0:
            try:
                if dataset is None or workflow is None:
                    raise ValueError("rank 0 must provide dataset and workflow")
                self._ray = _import_ray()
                address = self._start_service(dataset, workflow)
                outcome = _ServiceStartupOutcome(address)
            except BaseException as exc:
                startup_cause = exc
                outcome = _ServiceStartupOutcome(
                    address=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        try:
            outcome = _broadcast_startup_outcome(outcome, self._rank, self._world_size)
        except BaseException as exc:
            if self._rank == 0 and startup_cause is None:
                self._cleanup_started_service(exc)
            raise

        if outcome.error_type is not None:
            error = RuntimeError(
                "rollout service startup failed: "
                f"{outcome.error_type}: {outcome.error_message}"
            )
            if startup_cause is not None:
                raise error from startup_cause
            raise error

        address = outcome.address
        if address is None:
            raise RuntimeError("rollout service startup returned no address")
        if self._rank != 0:
            self._ray = _import_ray()
            self._ray.init(address=address.address, namespace=address.namespace)
            self._service = self._ray.get_actor(address.actor_name)

    async def next_batch(self) -> TrainingBatch:
        """Wait for the global buffer, then return this rank's contiguous shard."""
        async with self._next_batch_lock:
            self._raise_if_closed()
            if (
                self._rank == 0
                and self._last_batch_id is not None
                and self._last_updated_batch_id != self._last_batch_id
            ):
                raise RuntimeError("update_weights must follow each training batch")

            request = _BatchRequest(
                rank=self._rank,
                world_size=self._world_size,
                step_id=self._next_step_id,
            )
            batch = await self._get(self._service.next_batch.remote(request))
            if not isinstance(batch, TrainingBatch):
                raise TypeError("rollout service returned an invalid TrainingBatch")
            if batch.batch_id != self._next_step_id:
                raise RuntimeError("rollout service returned a batch out of order")
            self._next_step_id += 1
            self._last_batch_id = batch.batch_id
            return batch

    async def update_weights(self, model: Any) -> int:
        """Push rank 0's already-synchronized full parameters directly by NCCL."""
        self._raise_if_closed()
        if self._rank != 0:
            raise RuntimeError("only training rank 0 may update rollout weights")

        async with self._update_lock:
            self._raise_if_closed()
            transaction = asyncio.create_task(
                self._update_weights_transaction(model),
                name="chito-weight-update",
            )
            cancelled: asyncio.CancelledError | None = None
            while not transaction.done():
                try:
                    await asyncio.shield(transaction)
                except asyncio.CancelledError as exc:
                    cancelled = exc
                except BaseException:
                    break

            try:
                result = transaction.result()
            except BaseException:
                if cancelled is not None:
                    raise cancelled
                raise
            if cancelled is not None:
                raise cancelled
            return result

    async def _update_weights_transaction(self, model: Any) -> int:
        self._raise_if_closed()
        batch_id = self._last_batch_id
        if batch_id is None:
            raise RuntimeError("call next_batch before update_weights")
        if batch_id == self._last_updated_batch_id:
            raise RuntimeError("the current batch was already updated")
        expected_batch_id = (
            0
            if self._last_updated_batch_id is None
            else self._last_updated_batch_id + 1
        )
        if batch_id != expected_batch_id:
            raise RuntimeError("weight updates must follow batch order exactly")

        named_weights = _named_cuda_parameters(model)
        manifest = _make_manifest(batch_id, named_weights)
        ticket = await self._get(self._service.begin_weight_update.remote(manifest))
        try:
            assert self._sender is not None
            self._sender.send(named_weights)
        except BaseException as exc:
            await self._get(self._service.fail_weight_update.remote(ticket, str(exc)))
            raise

        version = int(
            await self._get(self._service.finish_weight_update.remote(ticket))
        )
        if version != ticket.new_policy_version:
            raise RuntimeError("rollout service committed an unexpected policy version")
        self._last_updated_batch_id = batch_id
        return version

    async def aclose(self) -> None:
        """Close this client; rank 0 also closes the owned service and runtime."""
        async with self._close_lock:
            if self._close_task is None:
                self._closed = True
                self._close_task = asyncio.create_task(
                    self._close_resources(), name="chito-rollout-close"
                )
            close_task = self._close_task
        await asyncio.shield(close_task)

    async def _close_resources(self) -> None:
        async with self._update_lock:
            if self._rank != 0:
                self._ray.shutdown()
                return

            try:
                await self._get(self._service.close.remote())
            finally:
                try:
                    if self._sender is not None:
                        self._sender.close()
                finally:
                    self._ray.shutdown()

    def _cleanup_started_service(self, primary: BaseException | None) -> None:
        failures: list[tuple[str, BaseException]] = []
        service = getattr(self, "_service", None)
        if service is not None:
            try:
                self._ray.get(service.close.remote())
            except BaseException as exc:
                failures.append(("rollout actor close", exc))
                kill = getattr(self._ray, "kill", None)
                if kill is not None:
                    try:
                        kill(service, no_restart=True)
                    except BaseException as kill_exc:
                        failures.append(("rollout actor kill", kill_exc))

        if self._sender is not None:
            try:
                self._sender.close()
            except BaseException as exc:
                failures.append(("NCCL sender close", exc))
            self._sender = None

        try:
            self._ray.shutdown()
        except BaseException as exc:
            failures.append(("Ray shutdown", exc))

        if primary is not None:
            for phase, failure in failures:
                primary.add_note(
                    f"startup cleanup failed during {phase}: "
                    f"{type(failure).__name__}: {failure}"
                )
            return
        if failures:
            phase, failure = failures[0]
            failure.add_note(f"startup cleanup first failed during {phase}")
            for later_phase, later_failure in failures[1:]:
                failure.add_note(
                    f"additional startup cleanup failure during {later_phase}: "
                    f"{type(later_failure).__name__}: {later_failure}"
                )
            raise failure

    def _start_service(
        self,
        dataset: Sequence[object],
        workflow: RolloutWorkflow,
    ) -> _RayServiceAddress:
        try:
            namespace = f"chito-{uuid.uuid4().hex}"
            actor_name = f"rollout-{uuid.uuid4().hex}"
            context = self._ray.init(namespace=namespace)
            cluster_address = str(context.address_info["address"])

            remote_service = self._ray.remote(
                num_cpus=1,
                max_concurrency=self._world_size + 2,
                max_restarts=0,
                max_task_retries=0,
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
        except BaseException as exc:
            self._cleanup_started_service(exc)
            raise

    async def _get(self, reference: Any) -> Any:
        return await asyncio.to_thread(self._ray.get, reference)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("rollout engine is closed")


class _NcclWeightSender:
    """Synchronous proxy for the isolated trainer-side NCCL process."""

    def __init__(self, rendezvous: _NcclRendezvous) -> None:
        import torch

        context = torch.multiprocessing.get_context("spawn")
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


def _broadcast_startup_outcome(
    outcome: _ServiceStartupOutcome | None,
    rank: int,
    world_size: int,
) -> _ServiceStartupOutcome:
    if world_size == 1:
        if outcome is None:
            raise RuntimeError("rank 0 did not provide a startup outcome")
        return outcome

    import torch
    import torch.distributed as distributed

    values = [outcome]
    device = None
    if distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    distributed.broadcast_object_list(values, src=0, device=device)
    result = values[0]
    if not isinstance(result, _ServiceStartupOutcome):
        raise RuntimeError(f"rank {rank} received an invalid rollout startup outcome")
    return result


def _import_ray() -> Any:
    try:
        import ray
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RolloutEngine requires Ray; install chito[vllm]"
        ) from exc
    return ray
