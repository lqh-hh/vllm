#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Helper script for external Elastic EP scaling.

This script only coordinates the old-rank control plane:

1. Fan out /scale_elastic_ep to all old ranks concurrently.
2. Wait for every old-rank API request to return success.

For scale-up, new ranks should be launched separately while this helper is
waiting. For scale-down, remove the target ranks from the external load
balancer before running this helper.
"""

import argparse
import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import requests


DEFAULT_REQUEST_PATH = "/scale_elastic_ep"


@dataclass
class ScaleRequestResult:
    endpoint: str
    ok: bool
    status_code: int | None
    response_text: str
    error: str | None = None


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"http://{endpoint}"


def send_scale_request(
    endpoint: str,
    request_path: str,
    new_dp_size: int,
    drain_timeout: int,
    request_timeout: int,
) -> ScaleRequestResult:
    url = f"{normalize_endpoint(endpoint)}{request_path}"
    payload = {
        "new_data_parallel_size": new_dp_size,
        "drain_timeout": drain_timeout,
    }
    headers = {"Content-Type": "application/json"}

    print(f"[old-rank] POST {url}")
    print(f"[old-rank] payload={json.dumps(payload, ensure_ascii=True)}")

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=request_timeout,
        )
    except requests.exceptions.RequestException as e:
        return ScaleRequestResult(
            endpoint=endpoint,
            ok=False,
            status_code=None,
            response_text="",
            error=str(e),
        )

    return ScaleRequestResult(
        endpoint=endpoint,
        ok=response.status_code == 200,
        status_code=response.status_code,
        response_text=response.text,
    )


def wait_for_old_rank_results(
    futures: list[Future[ScaleRequestResult]],
) -> list[ScaleRequestResult]:
    results: list[ScaleRequestResult] = []
    for future in futures:
        result = future.result()
        results.append(result)

        print(
            f"[old-rank] endpoint={result.endpoint} "
            f"status={result.status_code} ok={result.ok}"
        )
        if result.response_text:
            print(f"[old-rank] response={result.response_text}")
        if result.error:
            print(f"[old-rank] error={result.error}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trigger external Elastic EP scaling on all old ranks."
    )
    parser.add_argument(
        "--old-rank",
        action="append",
        required=True,
        help=(
            "Old-rank API endpoint, e.g. host:port or "
            "http://host:port. Repeat for every old rank."
        ),
    )
    parser.add_argument(
        "--new-dp-size",
        type=int,
        required=True,
        help="Target data parallel size after scaling.",
    )
    parser.add_argument(
        "--drain-timeout",
        type=int,
        default=120,
        help="drain_timeout passed to /scale_elastic_ep.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=1800,
        help="HTTP timeout in seconds for each old-rank scale request.",
    )
    parser.add_argument(
        "--request-path",
        default=DEFAULT_REQUEST_PATH,
        help="Scale API path. Defaults to /scale_elastic_ep.",
    )

    args = parser.parse_args()

    if args.new_dp_size == len(args.old_rank):
        parser.error("--new-dp-size must be different from the old rank count")

    request_path = args.request_path
    if not request_path.startswith("/"):
        request_path = f"/{request_path}"

    print(
        "Starting external Elastic EP scaling: "
        f"old_ranks={len(args.old_rank)} -> new_dp_size={args.new_dp_size}"
    )

    with ThreadPoolExecutor(max_workers=len(args.old_rank)) as executor:
        futures = [
            executor.submit(
                send_scale_request,
                endpoint,
                request_path,
                args.new_dp_size,
                args.drain_timeout,
                args.request_timeout,
            )
            for endpoint in args.old_rank
        ]
        old_rank_results = wait_for_old_rank_results(futures)

    old_ranks_ok = all(result.ok for result in old_rank_results)
    if old_ranks_ok:
        print("External Elastic EP scaling helper completed successfully.")
        return 0

    print("External Elastic EP scaling helper failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
