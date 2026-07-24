# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ParallelConfig

logger = init_logger(__name__)


_WORKER_DIST_INIT_READY_KEY_PREFIX = "eep_new_worker_dist_init_ready"


def worker_dist_init_ready_key(global_rank: int) -> str:
    return f"{_WORKER_DIST_INIT_READY_KEY_PREFIX}/{global_rank}"


def new_worker_dist_init_ready_keys(
    old_dp_size: int,
    new_dp_size: int,
    worker_world_size: int,
) -> list[str]:
    return [
        worker_dist_init_ready_key(dp_rank * worker_world_size + worker_rank)
        for dp_rank in range(old_dp_size, new_dp_size)
        for worker_rank in range(worker_world_size)
    ]


def publish_worker_dist_init_ready(
    parallel_config: "ParallelConfig", worker_rank: int
) -> None:
    """Publish that a scale-up Worker is about to enter distributed init."""
    global_rank = parallel_config.data_parallel_rank * parallel_config.world_size
    global_rank += worker_rank
    store = get_cached_tcp_store_client(
        parallel_config.data_parallel_master_ip,
        parallel_config._coord_store_port,
    )
    store.set(worker_dist_init_ready_key(global_rank), b"1")
    logger.info(
        "[Elastic EP scale-up] Worker reached distributed init: "
        "global_rank=%s, dp_rank=%s, worker_rank=%s",
        global_rank,
        parallel_config.data_parallel_rank,
        worker_rank,
    )
