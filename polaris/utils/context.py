from halo import Halo

from polaris.mixins import FormattingMixin


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
