# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
import weakref
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch.distributed

from vllm.config import ParallelConfig
from vllm.distributed import (
    sched_yield,
    stateless_destroy_torch_distributed_process_group,
)
from vllm.distributed.elastic_ep.readiness import (
    new_worker_dist_init_ready_keys,
)
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.v1.engine import (
    EEPNotificationType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WorkerType = Literal["existing", "new", "removing"]
_NEW_RANK_READY_COUNT_KEY = "eep_new_rank_ready_count"
_DEFAULT_DP_STATE_SYNC_INTERVAL = 32
_CAPTURE_DP_STATE_SYNC_INTERVAL = 4
_TRANSFER_WEIGHTS_QUIESCE_PREFIX = "eep_transfer_weights_quiesce"


class ScaleUpExistingEngineState(enum.IntEnum):
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    WAIT_NEW_WORKERS_DIST_INIT = 2
    TRANSFER_EXPERT_MAPPING = 3
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 4
    TRANSFER_WEIGHTS = 5
    SYNC_KV_CACHE_MEMORY_SIZE = 6
    MATERIALIZE_NEW_COMMUNICATION_GROUPS = 7
    SWITCH_AND_PREPARE = 8
    CAPTURE_NEW_RANK_DP_SYNC = 9
    FINALIZE_SWITCH_AND_PREPARE = 10
    EPLB_RESHUFFLE = 11
    COMPLETE = 12


class ScaleUpNewEngineState(enum.IntEnum):
    PRE_KV_INIT = 0
    MATERIALIZE_NEW_COMMUNICATION_GROUPS = 1
    PREPARE = 2
    EPLB_RESHUFFLE = 3
    COMPLETE = 4


class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    SWITCH_AND_PREPARE = 2
    COMPLETE = 3


class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2


EngineState: TypeAlias = (
    ScaleUpExistingEngineState
    | ScaleUpNewEngineState
    | ScaleDownRemainingEngineState
    | ScaleDownRemovingEngineState
)


class _BarrierTimeoutError(RuntimeError):
    """
    Exception raised for timeout
    in the first stage of our two-staged
    TCPStore based barrier to synchronize the
    execution of all engines in the DP group.
    """


class ElasticEPScalingState:
    def __init__(
        self,
        model_executor: "Executor",
        engine_core: "DPEngineCoreProc",
        vllm_config: "VllmConfig",
        new_parallel_config: ParallelConfig,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ):
        self.model_executor_ref = weakref.ref(model_executor)
        self.engine_core_ref = weakref.ref(engine_core)
        self.vllm_config = vllm_config
        self.old_dp_group = self.engine_core.dp_group if worker_type != "new" else None
        self.old_dp_store = self.engine_core.dp_store if worker_type != "new" else None
        self.new_parallel_config: ParallelConfig = new_parallel_config
        self.new_dp_group = self.engine_core.dp_group if worker_type == "new" else None
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request
        self._capture_dp_companion_started = False
        self._capture_pause_requested = False
        self._capture_pause_future = None
        self._capture_pause_timing: tuple[str, float] | None = None
        self._new_rank_ready_reported = False
        self._new_rank_ready_wait_timing: tuple[str, float] | None = None
        self._drain_window_timing: tuple[str, float] | None = None
        self._transfer_weights_quiesce_started = False
        self._transfer_weights_target_epoch: int | None = None
        self._transfer_weights_ready_published = False
        self._transfer_weights_engines_running_before_quiesce: bool | None = None

        self.state: EngineState
        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewEngineState.PRE_KV_INIT
                if worker_type == "new"
                else ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
            )
        else:
            self.state = (
                ScaleDownRemovingEngineState.PREPARE
                if worker_type == "removing"
                else ScaleDownRemainingEngineState.PREPARE
            )

    @property
    def model_executor(self) -> "Executor":
        model_executor = self.model_executor_ref()
        if model_executor is None:
            raise RuntimeError("Model executor has been garbage collected")
        return model_executor

    @property
    def engine_core(self) -> "DPEngineCoreProc":
        engine_core = self.engine_core_ref()
        if engine_core is None:
            raise RuntimeError("Engine core has been garbage collected")
        return engine_core

    def progress(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self._progress_new_engine()
                if self.worker_type == "new"
                else self._progress_existing_engine()
            )
        return (
            self._progress_removing_engine()
            if self.worker_type == "removing"
            else self._progress_remaining_engine()
        )

    def should_progress_after_model_step(self) -> bool:
        return (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state == ScaleUpExistingEngineState.CAPTURE_NEW_RANK_DP_SYNC
            and self._capture_dp_companion_started
        )

    def dp_state_sync_interval(self) -> int:
        if (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state == ScaleUpExistingEngineState.CAPTURE_NEW_RANK_DP_SYNC
        ):
            return _CAPTURE_DP_STATE_SYNC_INTERVAL
        return _DEFAULT_DP_STATE_SYNC_INTERVAL

    def should_defer_dp_state_sync(self) -> bool:
        """Avoid crossing EngineCore and Worker collectives while quiescing."""
        return (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS
            and self._transfer_weights_quiesce_started
        )

    def _timing_begin(self, action: str) -> float:
        start = time.perf_counter()
        print(
            "[EEP_PAUSE_TIMING] event=BEGIN "
            f"action={action} worker_type={self.worker_type} "
            f"dp_rank={self.engine_core.dp_rank} state={self.state.name} "
            f"wall_time={time.time():.6f}",
            flush=True,
        )
        return start

    def _timing_end(self, action: str, start: float, result: str = "ok") -> None:
        print(
            "[EEP_PAUSE_TIMING] event=END "
            f"action={action} worker_type={self.worker_type} "
            f"dp_rank={self.engine_core.dp_rank} state={self.state.name} "
            f"result={result} elapsed_ms={(time.perf_counter() - start) * 1000:.3f} "
            f"wall_time={time.time():.6f}",
            flush=True,
        )

    def _timed_call(self, action: str, func, *args, **kwargs):
        start = self._timing_begin(action)
        try:
            result = func(*args, **kwargs)
        except BaseException:
            self._timing_end(action, start, "error")
            raise
        self._timing_end(action, start)
        return result

    def should_skip_dummy_batch(self) -> bool:
        if getattr(self.engine_core, "_eep_drain_batch_queue", False):
            return not getattr(self.engine_core, "_eep_force_dummy_batch", False)
        if self.scale_type != "scale_up":
            return False
        if self.worker_type == "new":
            # The old ranks remain drained until EPLB reshuffling finishes.
            # A new rank must not start a dummy model collective while the old
            # ranks are waiting for it in the EPLB barrier.
            return self.state == ScaleUpNewEngineState.EPLB_RESHUFFLE
        if self.worker_type != "existing":
            return False
        if self._old_dp_wave_is_running():
            return False
        return self.state in (
            ScaleUpExistingEngineState.MATERIALIZE_NEW_COMMUNICATION_GROUPS,
            ScaleUpExistingEngineState.SWITCH_AND_PREPARE,
            ScaleUpExistingEngineState.FINALIZE_SWITCH_AND_PREPARE,
            ScaleUpExistingEngineState.EPLB_RESHUFFLE,
        )

    def run_pre_kv_init_states(self) -> None:
        assert self.scale_type == "scale_up" and self.worker_type == "new"
        assert self.state == ScaleUpNewEngineState.PRE_KV_INIT
        while self.state != ScaleUpNewEngineState.PREPARE:
            if not self.progress():
                sched_yield()
        assert self.state == ScaleUpNewEngineState.PREPARE

    def _execute_tcp_store_barrier(
        self, dp_store, group_rank, group_size, barrier_id, timeout=None
    ):
        arrival_key = f"arrival_{barrier_id}_{group_rank}"
        dp_store.set(arrival_key, b"1")

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < group_size:
            if (
                timeout is not None
                and time.time() - start_time > timeout.total_seconds()
            ):
                raise _BarrierTimeoutError(
                    f"Barrier timed out after {timeout.total_seconds()} seconds"
                )

            for i in range(group_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                present = dp_store.check([key])
                if present:
                    processes_arrived.add(i)

            if len(processes_arrived) < group_size:
                sched_yield()

    def _staged_barrier(
        self,
        use_new_group: bool,
        barrier_name: str,
        first_stage_timeout: timedelta | None = None,
    ) -> bool:
        """
        Execute a two-staged barrier to synchronize all engines in the DP group.

        Some DP EngineCores may receive the reconfiguration notifications
        later than others, and already proceed to engine step (model forward)
        in the busy loop.
        In this case, EngineCores that already proceed to reconfiguration
        should skip reconfiguration and execute model forward for one more
        step, so in the next step, all EngineCores will be synchronized.
        We use a two-staged barrier to achieve this. The first time each
        EngineCore executes the barrier, if a timeout is reached before the
        barrier completes, that means some EngineCores have already entered
        engine step. The EngineCores that timed out will then proceed to
        engine step, and will synchronize with the other EngineCores in the
        next step with a barrier without timeout.
        """
        dp_group = self.new_dp_group if use_new_group else self.old_dp_group
        dp_store = self.new_dp_store if use_new_group else self.old_dp_store
        assert dp_group is not None and dp_store is not None

        group_rank = dp_group.rank()
        group_size = dp_group.size()
        barrier_id = f"eep_barrier_{barrier_name}"
        sync_key = f"{barrier_id}_sync"

        # TODO(yongji): figure out appropriate timeout for the barrier
        if dp_store.check([sync_key]):
            timeout = None
        else:
            timeout = first_stage_timeout or timedelta(seconds=5)
        timing_action = f"staged_barrier:{barrier_name}"
        timing_start = self._timing_begin(timing_action)
        try:
            self._execute_tcp_store_barrier(
                dp_store, group_rank, group_size, barrier_id, timeout=timeout
            )
            torch.distributed.barrier(dp_group)
            if group_rank == 0:
                dp_store.delete_key(sync_key)
                for i in range(group_size):
                    dp_store.delete_key(f"arrival_{barrier_id}_{i}")
            self._timing_end(timing_action, timing_start)
            return True
        except _BarrierTimeoutError as e:
            if timeout is None:
                self._timing_end(timing_action, timing_start, "error")
                raise RuntimeError("Unexpected timeout encountered") from e
            dp_store.compare_set(sync_key, "", b"1")
            self._timing_end(timing_action, timing_start, "timeout_retry")
            return False
        except BaseException:
            self._timing_end(timing_action, timing_start, "error")
            raise

    def _get_local_dp_collective_state(self) -> tuple[int, int]:
        get_states = getattr(
            self.model_executor, "get_elastic_ep_dp_collective_states", None
        )
        if get_states is None:
            raise RuntimeError(
                "Elastic EP batch-boundary drain requires Worker DP collective "
                "state, but the current executor does not expose it"
            )

        states = get_states()
        if not states:
            raise RuntimeError("No Worker DP collective state is available")
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError(
                "Local workers reached different DP metadata collective epochs: "
                f"{states}"
            )
        if states[0][0] < 0 or states[0][1] < 0:
            raise RuntimeError(
                "The current Worker does not publish DP metadata collective state"
            )
        return states[0]

    def _drain_inflight_model_execution(self) -> bool:
        batch_queue = getattr(self.engine_core, "batch_queue", None)
        if not getattr(self.engine_core, "_eep_drain_batch_queue", False):
            action = f"request_scheduling_drain_window:{self.state.name}"
            self._drain_window_timing = (action, self._timing_begin(action))
        self.engine_core._eep_drain_batch_queue = True
        entered_epoch, completed_epoch = self._get_local_dp_collective_state()
        local_state = torch.tensor(
            [entered_epoch, completed_epoch, int(bool(batch_queue))],
            dtype=torch.int64,
        )
        gathered_states = [
            torch.empty_like(local_state) for _ in range(self.old_dp_group.size())
        ]
        self._timed_call(
            "drain_epoch_all_gather",
            torch.distributed.all_gather,
            gathered_states,
            local_state,
            group=self.old_dp_group,
        )

        states = [state.tolist() for state in gathered_states]
        target_epoch = max(state[0] for state in states)
        local_pending = bool(batch_queue)
        # A rank already at target_epoch has a Worker participating in that
        # model step. Only a rank that has neither entered it nor queued work
        # must issue the matching dummy forward.
        needs_catch_up = not local_pending and entered_epoch < target_epoch
        self.engine_core._eep_force_dummy_batch = needs_catch_up

        all_quiesced = all(
            entered == target_epoch and completed == target_epoch and pending == 0
            for entered, completed, pending in states
        )
        if all_quiesced:
            self.engine_core._eep_force_dummy_batch = False
            return True

        logger.info_once(
            "[Elastic EP] Waiting for a DP batch boundary before communication "
            "group switch: local_epoch=%s/%s, target_epoch=%s, "
            "local_pending_batches=%s, catch_up_dummy=%s",
            completed_epoch,
            entered_epoch,
            target_epoch,
            0 if batch_queue is None else len(batch_queue),
            needs_catch_up,
        )
        return False

    def _resume_model_execution_after_drain(self) -> None:
        self.engine_core._eep_force_dummy_batch = False
        if getattr(self.engine_core, "_eep_drain_batch_queue", False):
            self.engine_core._eep_drain_batch_queue = False
        if self._drain_window_timing is not None:
            action, start = self._drain_window_timing
            self._timing_end(action, start)
            self._drain_window_timing = None

    def _transfer_weights_quiesce_key_prefix(self) -> str:
        assert self.reconfig_request is not None and self.old_dp_group is not None
        # coord_store_port is allocated for each reconfiguration and therefore
        # keeps stale keys from a previous scale operation out of this protocol.
        return (
            f"{_TRANSFER_WEIGHTS_QUIESCE_PREFIX}_"
            f"{self.reconfig_request.coord_store_port}_"
            f"{self.old_dp_group.size()}_"
            f"{self.reconfig_request.new_data_parallel_size}"
        )

    @staticmethod
    def _encode_transfer_weights_epoch_state(
        entered_epoch: int,
        completed_epoch: int,
        pending_batches: int,
    ) -> bytes:
        return f"{entered_epoch},{completed_epoch},{pending_batches}".encode()

    @staticmethod
    def _decode_transfer_weights_epoch_state(value: bytes) -> tuple[int, int, int]:
        fields = value.decode().split(",")
        if len(fields) != 3:
            raise RuntimeError(
                f"Invalid transfer-weights epoch state in TCPStore: {value!r}"
            )
        return int(fields[0]), int(fields[1]), int(fields[2])

    def _progress_transfer_weights_quiesce(self) -> bool:
        """Non-blockingly stop old ranks at a common Worker collective epoch."""
        assert self.old_dp_group is not None and self.old_dp_store is not None

        rank = self.old_dp_group.rank()
        size = self.old_dp_group.size()
        prefix = self._transfer_weights_quiesce_key_prefix()
        target_key = f"{prefix}/target"
        ready_keys = [f"{prefix}/ready/{peer_rank}" for peer_rank in range(size)]
        state_keys = [f"{prefix}/state/{peer_rank}" for peer_rank in range(size)]

        if not self._transfer_weights_quiesce_started:
            action = "request_scheduling_drain_window:TRANSFER_WEIGHTS"
            self._drain_window_timing = (action, self._timing_begin(action))
            self._transfer_weights_engines_running_before_quiesce = (
                self.engine_core.engines_running
            )
            self._transfer_weights_quiesce_started = True

        # Stop submitting real batches first. Existing queued batches still
        # drain through step_with_batch_queue(), while dummy execution is
        # controlled below after all ranks have published their state.
        self.engine_core._eep_drain_batch_queue = True
        self.engine_core._eep_force_dummy_batch = False

        batch_queue = getattr(self.engine_core, "batch_queue", None)
        pending_batches = 0 if batch_queue is None else len(batch_queue)
        entered_epoch, completed_epoch = self._get_local_dp_collective_state()
        self.old_dp_store.set(
            state_keys[rank],
            self._encode_transfer_weights_epoch_state(
                entered_epoch,
                completed_epoch,
                pending_batches,
            ),
        )

        target_epoch = self._transfer_weights_target_epoch
        if target_epoch is None:
            if rank == 0 and not self.old_dp_store.check([target_key]):
                if not self.old_dp_store.check(state_keys):
                    return False
                states = [
                    self._decode_transfer_weights_epoch_state(
                        self.old_dp_store.get(state_key)
                    )
                    for state_key in state_keys
                ]
                # A queued async batch may not have incremented entered_epoch
                # yet. Account for all queued work, then select one future
                # epoch so no rank can overshoot before observing the target.
                target_epoch = (
                    max(
                        max(entered, completed + pending)
                        for entered, completed, pending in states
                    )
                    + 1
                )
                self.old_dp_store.set(target_key, str(target_epoch).encode())
                logger.info(
                    "[Elastic EP scale-up] Published transfer-weights quiesce "
                    "target: target_epoch=%s, states=%s",
                    target_epoch,
                    states,
                )
            elif not self.old_dp_store.check([target_key]):
                return False

            target_epoch = int(self.old_dp_store.get(target_key))
            self._transfer_weights_target_epoch = target_epoch

        if entered_epoch > target_epoch or completed_epoch > target_epoch:
            raise RuntimeError(
                "Worker DP collective epoch advanced beyond the transfer-weights "
                f"quiesce target: rank={rank}, entered={entered_epoch}, "
                f"completed={completed_epoch}, target={target_epoch}"
            )

        if pending_batches:
            return False

        if entered_epoch < target_epoch:
            # The busy loop executes exactly one synchronous dummy before
            # calling progress() again and publishing the updated epoch.
            self.engine_core._eep_force_dummy_batch = True
            self.engine_core.engines_running = True
            return False

        if completed_epoch < target_epoch:
            return False

        if not self._transfer_weights_ready_published:
            self.old_dp_store.set(ready_keys[rank], str(target_epoch).encode())
            self._transfer_weights_ready_published = True
            logger.info(
                "[Elastic EP scale-up] Old rank reached transfer-weights "
                "quiesce target: rank=%s/%s, target_epoch=%s",
                rank,
                size,
                target_epoch,
            )

        if not self.old_dp_store.check(ready_keys):
            return False

        ready_epochs = [
            int(self.old_dp_store.get(ready_key)) for ready_key in ready_keys
        ]
        if any(ready_epoch != target_epoch for ready_epoch in ready_epochs):
            raise RuntimeError(
                "Old DP ranks published inconsistent transfer-weights quiesce "
                f"epochs: target={target_epoch}, ready={ready_epochs}"
            )
        return True

    def _cleanup_transfer_weights_quiesce(self) -> None:
        assert self.old_dp_group is not None and self.old_dp_store is not None
        prefix = self._transfer_weights_quiesce_key_prefix()
        if self.old_dp_group.rank() == 0:
            for peer_rank in range(self.old_dp_group.size()):
                self.old_dp_store.delete_key(f"{prefix}/state/{peer_rank}")
                self.old_dp_store.delete_key(f"{prefix}/ready/{peer_rank}")
            self.old_dp_store.delete_key(f"{prefix}/target")
        self._transfer_weights_target_epoch = None
        self._transfer_weights_ready_published = False
        self._transfer_weights_quiesce_started = False
        if (
            self._transfer_weights_engines_running_before_quiesce is False
            and not self.engine_core.scheduler.has_unfinished_requests()
        ):
            self.engine_core.engines_running = False
        self._transfer_weights_engines_running_before_quiesce = None

    def _old_dp_wave_is_running(self) -> bool:
        return (
            self.engine_core.engines_running
            or self.engine_core.scheduler.has_unfinished_requests()
        )

    def _capture_pause_complete(self) -> bool:
        """Pause old ranks at their existing DP-aware scheduling boundary.

        New-rank readiness remains a non-blocking store signal while inference
        continues. Once it is visible, ``pause_scheduler(mode="keep")`` uses
        the normal DP state-sync checkpoints to stop every old rank together.
        Ranks must keep issuing dummy batches until that consensus completes;
        adding a separate blocking collective here can deadlock with an
        in-flight Worker metadata collective.
        """
        assert self.old_dp_group is not None and self.new_dp_store is not None

        expected_new_ranks = (
            self.new_parallel_config.data_parallel_size - self.old_dp_group.size()
        )
        # A new rank reports ready only after its dedicated capture DP sync has
        # finished, which also proves that every old-rank companion completed.
        # No additional Worker RPC or old-DP collective is needed here.
        ready_count = int(self.new_dp_store.get(_NEW_RANK_READY_COUNT_KEY))
        if (
            not self._capture_pause_requested
            and self._new_rank_ready_wait_timing is None
        ):
            action = "wait_new_rank_ready"
            self._new_rank_ready_wait_timing = (action, self._timing_begin(action))

        if ready_count < expected_new_ranks:
            return False

        if (
            not self._capture_pause_requested
            and self._new_rank_ready_wait_timing is not None
        ):
            action, start = self._new_rank_ready_wait_timing
            self._timing_end(
                action,
                start,
                f"ready={ready_count}/{expected_new_ranks}",
            )
            self._new_rank_ready_wait_timing = None

        if not self._capture_pause_requested:
            action = "pause_old_dp_after_new_rank_ready"
            self._capture_pause_timing = (action, self._timing_begin(action))
            self._capture_pause_future = self.engine_core.pause_scheduler(
                mode="keep",
                clear_cache=False,
            )
            self._capture_pause_requested = True
            logger.info(
                "[Elastic EP scale-up] new ranks ready; requested old-DP "
                "batch-boundary pause: old_dp_rank=%s/%s, ready_new_ranks=%s/%s",
                self.old_dp_group.rank(),
                self.old_dp_group.size(),
                ready_count,
                expected_new_ranks,
            )

        if (
            self._capture_pause_future is not None
            and not self._capture_pause_future.done()
        ):
            return False

        if self._capture_pause_timing is not None:
            action, start = self._capture_pause_timing
            self._timing_end(action, start)
            self._capture_pause_timing = None
        logger.info_once(
            "[Elastic EP scale-up] old-DP batch-boundary pause completed"
        )
        return True

    def _progress_existing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT:
            return False

        elif state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS:
            # NOTE(yongji): wait for all existing workers to receive the request
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False,
                barrier_name="init_new_dp_group",
                first_stage_timeout=timedelta(seconds=1),
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            # Initialize the EngineCore-side Gloo group first. New EngineCores
            # are blocked in the matching initialization and cannot launch
            # their Workers until the existing EngineCores join it.
            self._init_new_dp_group()
            self.state = ScaleUpExistingEngineState.WAIT_NEW_WORKERS_DIST_INIT
            return True

        elif state == ScaleUpExistingEngineState.WAIT_NEW_WORKERS_DIST_INIT:
            if not self._new_workers_dist_init_ready():
                return False
            # Stop scheduling only after the new Workers are ready, then wait
            # for every old Worker to leave its current DP collective before
            # creating HCCL groups on the Worker command thread.
            if not self._drain_inflight_model_execution():
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="worker_create_standby_groups"
            ):
                return False
            self._clear_new_workers_dist_init_ready()
            self._create_worker_standby_groups()
            self.state = ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING
            return True

        elif state == ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING:
            self._transfer_expert_mapping()
            # Keep inference drained until every old rank has completed the
            # Worker-side mapping transfer. Otherwise a faster rank can enter
            # the next DP metadata collective while a slower rank is still
            # reconfiguring its Worker.
            self._timed_call(
                "old_dp_barrier_after_transfer_expert_mapping",
                torch.distributed.barrier,
                self.old_dp_group,
            )
            # Notification and staged-barrier retries can leave old ranks at
            # different DP-state sync phases. The barrier above is a safe
            # batch boundary, so restart the common 32-step sync cadence before
            # inference resumes.
            self.engine_core.step_counter = 0
            self._resume_model_execution_after_drain()
            self.state = ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
            return True

        elif state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            return False

        elif state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._progress_transfer_weights_quiesce():
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="transfer_weights"
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            self._transfer_weights()
            self._cleanup_transfer_weights_quiesce()
            self._resume_model_execution_after_drain()
            self.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
            return True

        elif state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            self._sync_kv_cache_memory_size()
            self.state = ScaleUpExistingEngineState.MATERIALIZE_NEW_COMMUNICATION_GROUPS
            return True

        elif state == ScaleUpExistingEngineState.MATERIALIZE_NEW_COMMUNICATION_GROUPS:
            if not self._drain_inflight_model_execution():
                return False
            if not self._staged_barrier(
                use_new_group=True,
                barrier_name="materialize_new_communication_groups",
            ):
                return False
            self._materialize_new_communication_groups()
            assert self.new_dp_group is not None
            self._timed_call(
                "new_dp_barrier_after_materialize",
                torch.distributed.barrier,
                self.new_dp_group,
            )
            self.state = ScaleUpExistingEngineState.SWITCH_AND_PREPARE
            return True

        elif state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE:
            if not self._drain_inflight_model_execution():
                return False
            if not self._staged_barrier(
                use_new_group=False,
                barrier_name="switch_and_prepare",
            ):
                return False
            old_rank = self.old_dp_group.rank()
            old_size = self.old_dp_group.size()
            logger.info(
                "[Elastic EP scale-up] entering SWITCH_AND_PREPARE: old_dp_rank=%s/%s",
                old_rank,
                old_size,
            )
            self._switch_workers_and_prepare_for_scale_up()
            self.state = ScaleUpExistingEngineState.CAPTURE_NEW_RANK_DP_SYNC
            # All old ranks must start capture-time inference with the same
            # DP-state sync cadence. Keep the Worker queue drained while the
            # state and step counter are aligned, then resume together.
            self.engine_core.step_counter = 0
            self._timed_call(
                "old_dp_barrier_before_capture_inference",
                torch.distributed.barrier,
                self.old_dp_group,
            )
            self._resume_model_execution_after_drain()
            assert self.new_dp_group is not None
            logger.info(
                "[Elastic EP scale-up] finished SWITCH_AND_PREPARE; "
                "next=CAPTURE_NEW_RANK_DP_SYNC, new_dp_rank=%s/%s",
                self.new_dp_group.rank(),
                self.new_dp_group.size(),
            )
            return True

        elif state == ScaleUpExistingEngineState.CAPTURE_NEW_RANK_DP_SYNC:
            assert self.new_dp_group is not None
            if not self._capture_dp_companion_started:
                logger.info(
                    "[Elastic EP scale-up] entering CAPTURE_NEW_RANK_DP_SYNC: "
                    "new_dp_rank=%s/%s",
                    self.new_dp_group.rank(),
                    self.new_dp_group.size(),
                )
                self.model_executor.collective_rpc(
                    "elastic_ep_execute",
                    args=("start_new_rank_capture_dp_companion",),
                )
                self._capture_dp_companion_started = True
                return False

            if not self._capture_pause_complete():
                return False

            self._timed_call(
                "old_dp_barrier_after_capture_pause",
                torch.distributed.barrier,
                self.old_dp_group,
            )

            self._timed_call(
                "activate_v3_scale_up_after_capture",
                self.model_executor.collective_rpc,
                "elastic_ep_execute",
                args=("activate_v3_scale_up_after_capture",),
            )
            logger.info(
                "[Elastic EP scale-up] finished CAPTURE_NEW_RANK_DP_SYNC: "
                "new_dp_rank=%s/%s",
                self.new_dp_group.rank(),
                self.new_dp_group.size(),
            )
            self.state = ScaleUpExistingEngineState.FINALIZE_SWITCH_AND_PREPARE
            return True

        elif state == ScaleUpExistingEngineState.FINALIZE_SWITCH_AND_PREPARE:
            assert self.new_dp_group is not None
            logger.info(
                "[Elastic EP scale-up] entering FINALIZE_SWITCH_AND_PREPARE: "
                "new_dp_rank=%s/%s",
                self.new_dp_group.rank(),
                self.new_dp_group.size(),
            )
            self._finalize_engine_core_switch_after_scale_up()
            logger.info(
                "[Elastic EP scale-up] finished FINALIZE_SWITCH_AND_PREPARE: "
                "new_dp_rank=%s/%s",
                self.new_dp_group.rank(),
                self.new_dp_group.size(),
            )
            self.state = ScaleUpExistingEngineState.EPLB_RESHUFFLE
            # The engine-state all-reduce in finalize synchronizes every new
            # DP rank. Continue directly into EPLB instead of returning to the
            # busy loop, where ranks can otherwise drift apart for one step.
            self._barrier_before_scale_up_eplb()
            self._eplb_reshuffle()
            self._resume_model_execution_after_drain()
            self._resume_scheduler_after_scale_up()
            assert self.new_dp_store is not None
            if self.new_dp_group.rank() == 0:
                self._timed_call(
                    "ready_key_cleanup",
                    self.new_dp_store.delete_key,
                    _NEW_RANK_READY_COUNT_KEY,
                )
            self.state = ScaleUpExistingEngineState.COMPLETE
            self._update_parallel_config()
            self._notify_reconfigure_finished_after_scale_up()
            return True

        else:
            assert self.state == ScaleUpExistingEngineState.COMPLETE
            return True

    def _progress_new_engine(self) -> bool:
        state = self.state
        assert self.new_dp_group is not None and self.new_dp_store is not None

        if state == ScaleUpNewEngineState.PRE_KV_INIT:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            )
            self._timed_call(
                "new_rank_receive_weights",
                self.model_executor.collective_rpc,
                "elastic_ep_execute",
                args=("receive_weights",),
            )
            self.engine_core.available_gpu_memory_for_kv_cache = self._timed_call(
                "new_rank_sync_kv_cache_memory_size",
                ParallelConfig.sync_kv_cache_memory_size,
                self.new_dp_group,
                -1,
            )
            self.state = ScaleUpNewEngineState.MATERIALIZE_NEW_COMMUNICATION_GROUPS
            return True

        elif state == ScaleUpNewEngineState.MATERIALIZE_NEW_COMMUNICATION_GROUPS:
            if not self._staged_barrier(
                use_new_group=True,
                barrier_name="materialize_new_communication_groups",
            ):
                return False
            self._materialize_new_communication_groups()
            assert self.new_dp_group is not None
            self._timed_call(
                "new_dp_barrier_after_materialize",
                torch.distributed.barrier,
                self.new_dp_group,
            )
            self._timed_call(
                "worker_prepare_new_worker",
                self.model_executor.collective_rpc,
                "elastic_ep_execute",
                args=("prepare_new_worker",),
            )
            self.state = ScaleUpNewEngineState.PREPARE
            return True

        elif state == ScaleUpNewEngineState.PREPARE:
            if not self._new_rank_ready_reported:
                self._timed_call(
                    "new_rank_ready_report",
                    self.new_dp_store.add,
                    _NEW_RANK_READY_COUNT_KEY,
                    1,
                )
                self._new_rank_ready_reported = True
            tensor = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
            self._timed_call(
                "new_rank_engine_state_all_reduce",
                torch.distributed.all_reduce,
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.new_dp_group,
            )
            data = tensor.tolist()
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self.state = ScaleUpNewEngineState.EPLB_RESHUFFLE
            # Match the existing ranks immediately after the engine-state
            # all-reduce; do not let the new rank enter the busy loop first.
            self._barrier_before_scale_up_eplb()
            assert self.new_dp_group.rank() > 0
            self._eplb_reshuffle()
            self._resume_scheduler_after_scale_up()
            self.state = ScaleUpNewEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleUpNewEngineState.COMPLETE
            return True

    def _progress_remaining_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemainingEngineState.PREPARE:
            self.state = ScaleDownRemainingEngineState.EPLB_RESHUFFLE
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            return True

        elif state == ScaleDownRemainingEngineState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="eplb_reshuffle"
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            self._eplb_reshuffle_before_scale_down()
            self.state = ScaleDownRemainingEngineState.SWITCH_AND_PREPARE
            # NOTE(yongji): currently, after EPLB reshuffle
            # that redistributes experts to remaining workers, workers
            # to be removed will immediately initiate shutdown;
            # existing workers can no longer execute forward steps using
            # the old setup. In the future, we may keep
            # the removing workers alive a bit longer,
            # e.g., to drain in-batch requests.
            self._create_standby_groups()
            self._switch_and_prepare()
            self._update_parallel_config()
            self.state = ScaleDownRemainingEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleDownRemainingEngineState.COMPLETE
            return True

    def _progress_removing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemovingEngineState.PREPARE:
            self.state = ScaleDownRemovingEngineState.EPLB_RESHUFFLE
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            return True

        if state == ScaleDownRemovingEngineState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="eplb_reshuffle"
            ):
                return False
            assert self.old_dp_group.rank() > 0
            self._eplb_reshuffle_before_scale_down()
            self._switch_and_remove()
            self.state = ScaleDownRemovingEngineState.COMPLETE
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.SHUTDOWN_COMPLETE
            )
            return True

        else:
            assert self.state == ScaleDownRemovingEngineState.COMPLETE
            return True

    def handle_notification(self, notification_type: EEPNotificationType):
        assert self.worker_type != "new"
        assert self.old_dp_store is not None
        if (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_INIT_READY
            and self.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
        ):
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            self.state = ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS
        elif (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            and self.state
            == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
        ):
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            self.state = ScaleUpExistingEngineState.TRANSFER_WEIGHTS

    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        return (
            self.state == ScaleDownRemovingEngineState.COMPLETE
            if self.worker_type == "removing"
            else self.state == ScaleDownRemainingEngineState.COMPLETE
        )

    def _init_new_dp_group(self) -> None:
        assert self.old_dp_group is not None
        self.new_dp_group, self.new_dp_store = self._timed_call(
            "init_new_dp_group",
            self.new_parallel_config.stateless_init_dp_group,
            return_store=True,
        )
        assert self.new_dp_store is not None
        if self.old_dp_group.rank() == 0:
            self._timed_call(
                "ready_key_initialize",
                self.new_dp_store.set,
                _NEW_RANK_READY_COUNT_KEY,
                b"0",
            )

    def _new_workers_dist_init_ready(self) -> bool:
        assert self.old_dp_group is not None
        ready_keys = new_worker_dist_init_ready_keys(
            self.old_dp_group.size(),
            self.new_parallel_config.data_parallel_size,
            self.new_parallel_config.world_size,
        )
        coord_store = get_cached_tcp_store_client(
            self.new_parallel_config.data_parallel_master_ip,
            self.new_parallel_config._coord_store_port,
        )
        if not coord_store.check(ready_keys):
            return False

        logger.info_once(
            "[Elastic EP scale-up] All new Workers reached distributed init; "
            "starting standby communication group creation"
        )
        return True

    def _clear_new_workers_dist_init_ready(self) -> None:
        assert self.old_dp_group is not None
        if self.old_dp_group.rank() != 0:
            return

        ready_keys = new_worker_dist_init_ready_keys(
            self.old_dp_group.size(),
            self.new_parallel_config.data_parallel_size,
            self.new_parallel_config.world_size,
        )
        coord_store = get_cached_tcp_store_client(
            self.new_parallel_config.data_parallel_master_ip,
            self.new_parallel_config._coord_store_port,
        )
        for ready_key in ready_keys:
            coord_store.delete_key(ready_key)

    def _create_worker_standby_groups(self) -> None:
        self._timed_call(
            "worker_create_standby_groups",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("create_standby_groups", self.reconfig_request),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Created standby communication groups")

    def _create_standby_groups(self) -> None:
        self._init_new_dp_group()
        self._create_worker_standby_groups()

    def _transfer_weights(self) -> None:
        assert self.reconfig_request is not None and self.old_dp_group is not None
        old_dp_size = self.old_dp_group.size()
        new_dp_size = self.reconfig_request.new_data_parallel_size

        self._timed_call(
            "transfer_model_weights",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("transfer_weights", old_dp_size, new_dp_size),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Transferred weights to new workers")

    def _transfer_expert_mapping(self):
        assert self.old_dp_group is not None
        self._timed_call(
            "transfer_expert_mapping",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("broadcast_expert_mapping",),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Broadcasted expert mapping to new workers")

    def _sync_kv_cache_memory_size(self):
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        assert self.new_dp_group is not None and self.old_dp_group is not None
        self._timed_call(
            "sync_kv_cache_memory_size",
            ParallelConfig.sync_kv_cache_memory_size,
            self.new_dp_group,
            self.engine_core.available_gpu_memory_for_kv_cache,
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Synced KV cache memory size to new workers")

    def _materialize_new_communication_groups(self):
        self._timed_call(
            "worker_materialize_new_communication_groups",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("materialize_new_communication_groups",),
        )
        assert self.new_dp_group is not None
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Materialized new communication groups")

    def _switch_workers_and_prepare_for_scale_up(self):
        self._timed_call(
            "worker_switch_and_prepare",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("switch_and_prepare",),
        )

    def _barrier_before_scale_up_eplb(self) -> None:
        assert self.new_dp_group is not None
        self._timed_call(
            "new_dp_barrier_before_eplb",
            torch.distributed.barrier,
            self.new_dp_group,
        )

    def _resume_scheduler_after_scale_up(self) -> None:
        self._timed_call(
            "resume_scheduler_after_scale_up",
            self.engine_core.resume_scheduler,
        )

    def _notify_reconfigure_finished_after_scale_up(self) -> None:
        assert self.new_dp_group is not None
        if (
            self.new_dp_group.rank() == 0
            or self.vllm_config.parallel_config.data_parallel_external_lb
        ):
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _finalize_engine_core_switch_after_scale_up(self):
        old_dp_group = self.old_dp_group
        self._timed_call(
            "destroy_old_dp_group",
            stateless_destroy_torch_distributed_process_group,
            old_dp_group,
        )
        assert self.new_dp_group is not None
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        tensor = torch.tensor(
            [engines_running, current_wave, step_counter],
            dtype=torch.int32,
            device="cpu",
        )
        self._timed_call(
            "new_dp_engine_state_all_reduce",
            torch.distributed.all_reduce,
            tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=new_dp_group,
        )
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])

    def _switch_and_prepare(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("switch_and_prepare",)
        )
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        assert self.new_dp_group is not None
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        tensor = torch.tensor(
            [engines_running, current_wave, step_counter],
            dtype=torch.int32,
            device="cpu",
        )
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=new_dp_group
        )
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if (
            new_dp_group.rank() == 0
            or self.vllm_config.parallel_config.data_parallel_external_lb
        ):
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )
        if new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _eplb_reshuffle(self):
        self._timed_call(
            "eplb_expert_reshuffle",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("perform_eplb_reshuffle",),
        )
        # Reshuffle changes per-rank token routing; the locked MoE workspace
        # may now be too small. Rewarm covers both new and existing engines.
        self._timed_call(
            "post_eplb_workspace_rewarm",
            self.model_executor.collective_rpc,
            "elastic_ep_execute",
            args=("rewarm_workspace",),
        )
        assert self.new_dp_group is not None
        if self.new_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _eplb_reshuffle_before_scale_down(self):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        self.model_executor.collective_rpc(
            "elastic_ep_execute",
            args=(
                "perform_scale_down_eplb_reshuffle",
                self.reconfig_request.new_data_parallel_size,
            ),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _switch_and_remove(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("switch_and_remove",)
        )

    def _update_parallel_config(self):
        assert self.reconfig_request is not None
        reconfig_request = self.reconfig_request
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        parallel_config._coord_store_port = reconfig_request.coord_store_port
