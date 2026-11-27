"""Cron scheduler background infrastructure (spec §5.3 / §7 SCHED-001).

``run_scheduler_tick`` is the REAL production scheduler entry point: one pass
over due crons (occurrence claiming, overlap skips, execution dispatch). The
behavioral suite drives it directly — concurrency scenarios await it from
competing tasks, mirroring two workers; in production the lifespan-owned
background loop invokes the same entry point on a fixed interval.

Occurrence claiming and overlap handling are enforced by database constraints
and atomic state transitions in ``PostgresCronRunRepository`` (CRN-006) —
this module holds no in-memory locking.
"""

import asyncio
import logging

from src.domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

# Production cadence for the background loop. Occurrences are minute-granular
# (five-field expressions) so a 30s interval bounds the skip latency.
DEFAULT_TICK_INTERVAL_SECONDS = 30.0


async def run_scheduler_tick() -> None:
    """Invoke one production scheduler pass over due crons."""
    from src.dependencies import get_scheduler_tick_use_case

    await get_scheduler_tick_use_case().execute()


def start_scheduler_loop(tick, interval: float = DEFAULT_TICK_INTERVAL_SECONDS) -> asyncio.Task:
    """Start the background scheduler loop (lifespan ownership).

    Args:
        tick: Awaitable-producing callable (the SchedulerTickUseCase.execute).
        interval: Seconds between passes.

    Returns:
        The running :class:`asyncio.Task` — cancellable via
        :func:`stop_scheduler_loop`.
    """
    logger.info(LogMessage.CRON_SCHEDULER_LOOP_STARTED, interval)

    async def _loop() -> None:
        while True:
            try:
                await tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a failing pass must not stop the loop
                logger.exception(LogMessage.CRON_SCHEDULER_TICK_FAILED)
            await asyncio.sleep(interval)

    return asyncio.create_task(_loop())


async def stop_scheduler_loop(task: asyncio.Task | None) -> None:
    """Cancel the background loop and await its termination.

    Cancellation is the expected termination path on shutdown, so the cancelled
    child is awaited via ``gather(return_exceptions=True)`` without an
    ``except asyncio.CancelledError`` swallow clause (Sonar S7497: a swallowed
    ``CancelledError`` may mask enclosing cancellation). Real loop failures
    surface as errors; a genuine ``CancelledError`` reaching this point — e.g.
    process shutdown cancelling *this* coroutine — propagates to the caller.
    """
    if task is None:
        return
    task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)
    for result in results if isinstance(results, list) else [results]:
        if isinstance(result, asyncio.CancelledError):
            # Expected: the loop was cancelled on shutdown (this pass).
            logger.info(LogMessage.CRON_SCHEDULER_LOOP_STOPPED)
        elif isinstance(result, BaseException):
            logger.error(LogMessage.CRON_SCHEDULER_TICK_FAILED, repr(result))
        else:
            logger.info(LogMessage.CRON_SCHEDULER_LOOP_STOPPED)
