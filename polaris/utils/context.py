from contextlib import contextmanager
from contextvars import ContextVar
from itertools import cycle

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Singleton Progress instance to be used for all calls to `track_progress`
progress_instance = ContextVar(
    "progress",
    default=Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ),
)

colors = cycle(
    {
        "green",
        "cyan",
        "magenta",
    }
)


@contextmanager
def track_progress(description: str, total: float | None = 1.0):
    """
    Use the Progress instance to track a task's progress
    """
    progress = progress_instance.get()

    # Make sure the Progress is started
    progress.start()

    task = progress.add_task(f"[{next(colors)}]{description}", total=total)

    try:
        # Yield the task and Progress instance, for more granular control
        yield progress, task

        # Mark the task as completed
        progress.update(task, completed=total, refresh=True)
        progress.log(f"[green] Success: {description}")
    except Exception:
        progress.log(f"[red] Error: {description}")
        raise
    finally:
        # Remove the task from the UI, and stop the progress bar if all tasks are completed
        progress.remove_task(task)
        if progress.finished:
            progress.stop()
