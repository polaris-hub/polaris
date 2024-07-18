from yaspin import yaspin
from yaspin.spinners import Spinners


class ProgressIndicator:
    def __init__(self):
        self._spinner = yaspin()
        self._spinner.spinner = Spinners.dots
        self._spinner.text = "In progress... "

    def start_spinner(self, text: str):
        if text:
            self._spinner.text = text

        self._spinner.start()

    def stop_spinner(self, success: bool, text: str = ""):
        if success:
            self._spinner.color = "green"
            self._spinner.text = ""
            self._spinner.ok(f"✅ SUCCESS: \033[1m{text}\033[0m\n")
        else:
            self._spinner.color = "red"
            self._spinner.fail("💥 ERROR: ")

        self._spinner.stop()
