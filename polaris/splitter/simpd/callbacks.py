from pymoo.core.callback import Callback


class ProgressCallback(Callback):
    def __init__(
        self,
        n_gen: int,
        progress: bool = True,
        leave: bool = False,
        auto_tqdm: bool = True,
        description: str = "Optimization",
    ):
        super().__init__()

        if auto_tqdm:
            from tqdm.auto import tqdm as tqdm_module
        else:
            from tqdm import tqdm as tqdm_module

        self.pbar = tqdm_module(
            total=n_gen,
            disable=not progress,
            leave=leave,
            desc=description,
        )

    def notify(self, algorithm):
        self.pbar.update(1)

        if self.pbar.total == algorithm.n_iter:
            self.pbar.close()
