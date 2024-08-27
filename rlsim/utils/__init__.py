import os
from contextlib import (ExitStack, contextmanager, redirect_stderr,
                        redirect_stdout)


@contextmanager
def suppress_print(out: bool = True, err: bool = False):
    """Suppress stdout and/or stderr."""

    with ExitStack() as stack:
        devnull = stack.enter_context(open(os.devnull, "w"))
        if out:
            stack.enter_context(redirect_stdout(devnull))
        if err:
            stack.enter_context(redirect_stderr(devnull))
        yield