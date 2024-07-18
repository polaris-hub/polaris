from contextlib import contextmanager
from yaspin import yaspin
from yaspin.spinners import Spinners


@contextmanager
def progress_indicator(start_msg: str = "In progress...", success_msg: str = "", error_msg: str = ""):
    spinner = yaspin()
    spinner.spinner = Spinners.dots
    spinner.text = start_msg
    result_message = {"message": ""}

    try:
        spinner.start()
        yield result_message

        # For using a message generated in the manager's enclosed function call
        if result_message["message"] != "":
            success_msg = result_message["message"]

        spinner.color = "green"
        spinner.text = ""
        spinner.ok(f"✅ SUCCESS: \033[1m{success_msg}\033[0m\n")

    except Exception as err:
        spinner.color = "red"
        spinner.text = ""
        spinner.fail(f"💥 ERROR: {error_msg}")
        raise err

    finally:
        spinner.stop()
