import os
from contextlib import contextmanager, redirect_stdout


@contextmanager
def suppress_print():
    """Suppress stdout temporarily."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            yield