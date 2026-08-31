# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleUpExistingEngineState,
)


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
