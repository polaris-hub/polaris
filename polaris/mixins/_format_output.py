class FormattingMixin:
    """Mixin class for formatting strings to be output in the console"""

    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    _END_CODE = "\033[0m"

    def format(self, text: str, codes: str | list[str]):
        if not isinstance(codes, list):
            codes = [codes]

        return "".join(codes) + text + self._END_CODE
