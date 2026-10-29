# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
import torch
import zmq.asyncio
from torch.distributed.distributed_c10d import Backend, _get_default_timeout

from vllm.config import ParallelConfig
from vllm.distributed import utils as distributed_utils
from vllm.distributed.elastic_ep import elastic_state
from vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleUpExistingEngineState,
)
from vllm.utils.network_utils import get_open_zmq_inproc_path, make_zmq_socket
from vllm.v1.engine.core_client import DPAsyncMPClient


@pytest.mark.asyncio
async def test_idle_request_notifies_coordinator_after_repeated_migration():
    with zmq.asyncio.Context() as ctx:
        client = object.__new__(DPAsyncMPClient)
        client.ctx = ctx
        client.resources = SimpleNamespace(stats_update_task=None)
        client.stats_update_address = None
        client.engine_ranks_managed = [3]
        client.core_engine = (3).to_bytes(2, "little")
        client.client_index = 0
        client.current_wave = 7
        client.call_utility_async = AsyncMock()
        sent = asyncio.get_running_loop().create_future()
        sent.set_result(None)
        client._send_input = MagicMock(return_value=sent)
        client._ensure_output_queue_task = MagicMock()
        client.first_req_sock_addr = get_open_zmq_inproc_path()
        client.first_req_send_socket = make_zmq_socket(
            ctx, client.first_req_sock_addr, zmq.PAIR, bind=True
        )
        try:
            for _ in range(3):
                with ctx.socket(zmq.XPUB) as coordinator:
                    address = get_open_zmq_inproc_path()
                    coordinator.bind(address)
                    await client.reconnect_dp_coordinator("input", "output", address)
                    assert await asyncio.wait_for(coordinator.recv(), 2) == b"\x01"
                    client.engines_running = False
                    await asyncio.wait_for(
                        client.add_request_async(SimpleNamespace()), 2
                    )
                    notification = await asyncio.wait_for(coordinator.recv(), 2)
                    assert msgspec.msgpack.decode(notification) == [
                        client.core_engine,
                        7,
                    ]
        finally:
            task = client.resources.stats_update_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            client.first_req_send_socket.close(linger=0)


@pytest.mark.parametrize("steady_state", [False, True])
def test_standby_dp_init_can_wait_for_joiners_without_changing_serving_timeout(
    monkeypatch, steady_state
):
    parallel = object.__new__(ParallelConfig)
    parallel.enable_fault_tolerance = True
    parallel.cpu_distributed_timeout_seconds = 60
    parallel.data_parallel_master_ip = "127.0.0.1"
    parallel.data_parallel_rank = 3
    parallel.data_parallel_size = 4
    parallel._data_parallel_master_rank = 1
    parallel._pick_stateless_dp_port = lambda: (29500, None)
    monkeypatch.setattr(distributed_utils, "steady_state_entered", lambda: steady_state)
    init_group = MagicMock(return_value=(object(), object()))
    monkeypatch.setattr(
        distributed_utils, "stateless_init_torch_distributed_process_group", init_group
    )
    state = object.__new__(ElasticEPScalingState)
    state.new_parallel_config = parallel

    assert state._init_new_dp_group() == init_group.return_value
    assert init_group.call_args.kwargs["timeout"] == _get_default_timeout(Backend.GLOO)
    assert init_group.call_args.kwargs["store_master_rank"] == 1
    assert parallel.cpu_distributed_timeout_seconds == 60
    parallel.stateless_init_dp_group()
    expected = timedelta(seconds=60) if steady_state else None
    assert init_group.call_args.kwargs["timeout"] == expected


def test_commit_restores_serving_timeout_on_replacement_dp_group(monkeypatch):
    state = object.__new__(ElasticEPScalingState)
    state.old_dp_group = object()
    state.new_dp_group = SimpleNamespace(rank=lambda: 3)
    state.new_dp_store = object()
    state.new_parallel_config = SimpleNamespace(cpu_distributed_timeout_seconds=60)
    engine = SimpleNamespace(engines_running=True, current_wave=7, step_counter=5)
    state.engine_core_ref = lambda: engine
    monkeypatch.setattr(
        elastic_state, "stateless_destroy_torch_distributed_process_group", MagicMock()
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", MagicMock())
    restore_timeout = MagicMock()
    monkeypatch.setattr(elastic_state, "set_gloo_backend_timeout", restore_timeout)

    state._commit_new_dp_group()

    assert engine.dp_group is state.new_dp_group
    assert engine.dp_store is state.new_dp_store
    restore_timeout.assert_called_once_with(state.new_dp_group, timedelta(seconds=60))


def test_existing_engine_waits_for_precommit_capture_before_commit():
    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
    state.old_dp_group = object()
    state.reconfig_request = SimpleNamespace(operation_id="scale-1")
    state._sync_kv_cache_memory_size = MagicMock(return_value=True)
    state._uses_precommit_graph_capture = MagicMock(return_value=True)
    state._execute_async = MagicMock(side_effect=[False, True])
    state._mark_ready_for_switch = MagicMock()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.CAPTURE_NEW_RANKS
    state._mark_ready_for_switch.assert_not_called()

    assert not state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.CAPTURE_NEW_RANKS
    state._mark_ready_for_switch.assert_not_called()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state._execute_async.assert_called_with(
        "run_new_rank_capture_companion",
        "scale-1",
    )
    state._mark_ready_for_switch.assert_called_once_with()


def test_existing_engine_without_fast_capture_keeps_normal_transition():
    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
    state.old_dp_group = object()
    state._sync_kv_cache_memory_size = MagicMock(return_value=True)
    state._uses_precommit_graph_capture = MagicMock(return_value=False)
    state._mark_ready_for_switch = MagicMock()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state._mark_ready_for_switch.assert_called_once_with()


def test_ready_keys_preserve_rank_slots_after_fault_tolerance_scale_down():
    keys = []
    for old_rank, new_rank in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        state = object.__new__(ElasticEPScalingState)
        engine_core = SimpleNamespace(dp_rank=old_rank)
        state.engine_core_ref = lambda engine_core=engine_core: engine_core
        state.new_parallel_config = SimpleNamespace(data_parallel_rank=new_rank)
        keys.append(state.ready_key)

    assert keys == ["eep_ready/0", "eep_ready/1", "eep_ready/2", "eep_ready/3"]


def _commit_state_after_fault(old_rank, new_rank, new_size=4):
    from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    parallel_config = SimpleNamespace(
        data_parallel_rank=old_rank,
        data_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_rank=0,
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
    )
    engine = SimpleNamespace(
        engine_index=old_rank,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        output_queue=Queue(),
        dp_store=MagicMock(),
        model_executor=SimpleNamespace(failure_callback=None),
    )
    sentinel = engine.ft_sentinel = EngineCoreSentinel(engine, parallel_config)
    sentinel._recover_and_vote = MagicMock()
    sentinel.scale_down(
        FaultToleranceRequest(
            "scale_down", {"removed_dp_ranks": [2]}, request_id="shrink-1"
        )
    )
    engine.output_queue.get_nowait()
    sentinel._dp_reinit_epoch = 1

    state = object.__new__(ElasticEPScalingState)
    state.engine_core_ref = lambda: engine
    state.vllm_config = engine.vllm_config
    state.scale_type = "scale_up"
    state.old_dp_group = object()
    state.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state.commit_requested = True
    state.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_size,
        new_data_parallel_rank=new_rank,
        new_data_parallel_rank_local=0,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29860,
        new_data_parallel_master_port_list=[],
        coord_store_port=29861,
        existing_dp_ranks=[0, 1, 3],
        data_parallel_master_rank=0,
    )
    state._commit_new_dp_group = MagicMock()
    state._collective_rpc = MagicMock()
    state._send_reconfigure_finished = MagicMock()
    return state, sentinel


@pytest.mark.parametrize(
    "old_rank,new_rank,new_size,removed_rank",
    [(3, 3, 4, 2), (0, 0, 4, 2), (3, 3, 5, 4)],
)
def test_scale_up_rebases_fault_state_for_the_next_failure(
    old_rank, new_rank, new_size, removed_rank
):
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(old_rank, new_rank, new_size)
    assert state._progress_existing_engine()
    request = FaultToleranceRequest(
        "scale_down", {"removed_dp_ranks": [removed_rank]}, request_id="shrink-2"
    )
    sentinel.scale_down(request)

    # A replacement may reuse a previously dead rank. Only this cycle's
    # failure belongs in the worker recovery request.
    assert request.params["dead_dp_ranks"] == [removed_rank]
    assert sentinel._dp_reinit_epoch == 0
    _, output = sentinel.engine.output_queue.get_nowait()
    assert output.engine_index == old_rank
    assert output.utility_output.result.result == {
        "id": old_rank,
        "dp_rank": new_rank,
        "status": "healthy",
    }


def test_failed_scale_up_keeps_fault_state():
    state, sentinel = _commit_state_after_fault(3, 3)
    state._collective_rpc.side_effect = RuntimeError("worker commit failed")
    with pytest.raises(RuntimeError, match="worker commit failed"):
        state._progress_existing_engine()
    assert sentinel._dead_dp_ranks == {2}
    assert sentinel._dp_reinit_epoch == 1
    assert sentinel.parallel_config.data_parallel_rank == 3
    assert sentinel.engine.output_queue.empty()


def test_scale_up_does_not_allow_removing_the_survivor_itself():
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(3, 3)
    state._progress_existing_engine()
    request = FaultToleranceRequest(
        "scale_down", {"removed_dp_ranks": [3]}, request_id="shrink-1"
    )
    with pytest.raises(ValueError, match=r"dp_rank=3, dead_dp_ranks=\[\]"):
        sentinel.scale_down(request)
