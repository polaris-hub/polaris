from contextlib import contextmanager
from contextvars import ContextVar
from itertools import cycle

from halo import Halo
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from polaris.mixins import FormattingMixin

# Singleton Progress instance to be used for all calls to `track_progress`
progress_instance = ContextVar(
    "progress",
    default=Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
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


class ProgressIndicator(FormattingMixin):
    def __init__(self, success_msg: str, error_msg: str, start_msg: str = "In progress..."):
        self._start_msg = start_msg
        self._success_msg = success_msg
        self._error_msg = error_msg

        self._spinner = Halo(text=self._start_msg, spinner="dots")

    def __enter__(self):
        self._spinner.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._spinner.text = ""

        if exc_type:
            self._spinner.text_color = "red"
            self._spinner.fail(f"ERROR: {self._error_msg}")
        else:
            self._spinner.text_color = "green"
            self._spinner.succeed(f"SUCCESS: {self.format(self._success_msg, self.BOLD)}\n")

        self._spinner.stop()

    def update_success_msg(self, msg: str):
        self._success_msg = msg


@contextmanager
def track_progress(description: str, total: float | None = 100.0):
    """
    Use the Progress instance to track a task's progress
    """
    progress = progress_instance.get()

    # Make sure the Progress is started
    progress.start()

    task = progress.add_task(f"[{next(colors)}]{description}", total=total)

    try:
        # time.sleep(5)
        # Yield the progress instance, for more granular control
        yield task, progress
        # time.sleep(5)
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
