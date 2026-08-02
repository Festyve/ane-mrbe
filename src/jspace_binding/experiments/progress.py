"""Progress logging for the long sweeps.

The runners print nothing between "generated N families" and the final JSON, so
a real-backend run looks identical to a hung one for tens of minutes. That is
tolerable at RQ1's ~4,800 reads and not at RQ2's ~21,600 per site, where the
gap is hours — and it is actively harmful on a metered or time-limited box,
where the useful question is not "is it alive" but "will it finish before the
machine goes away, and should I cut the sweep short".

Writes to STDERR so `> results.json` still captures clean JSON.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def track(
    items: Iterable[T],
    label: str,
    total: int | None = None,
    every: int | None = None,
    stream: object = None,
) -> Iterator[T]:
    """Yield `items`, logging rate and ETA to stderr.

    `every` defaults to roughly twenty updates over the whole sweep, floored at
    1 — frequent enough to extrapolate from early on, sparse enough not to bury
    a traceback in a log someone has to read later.

    The rate is measured from the FIRST item's completion, not from entry, so
    one-off setup inside the loop (a lazy model load, a cache warm) does not
    drag the estimate down for the rest of the run.
    """
    out = stream if stream is not None else sys.stderr
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    if every is None:
        every = max(1, (total // 20) if total else 50)

    started = time.monotonic()
    first_done: float | None = None
    for index, item in enumerate(items, start=1):
        yield item
        if first_done is None:
            first_done = time.monotonic()
        if index % every and index != total:
            continue
        elapsed = time.monotonic() - started
        # Rate over items 2..n, so item 1 carries the setup cost alone.
        steady = time.monotonic() - first_done
        rate = (index - 1) / steady if index > 1 and steady > 0 else 0.0
        message = f"[{label}] {index}"
        if total:
            message += f"/{total} ({100 * index / total:.0f}%)"
        message += f"  {_duration(elapsed)} elapsed"
        if rate > 0:
            message += f"  {rate:.1f}/s"
            if total and index < total:
                message += f"  ETA {_duration((total - index) / rate)}"
        print(message, file=out, flush=True)
