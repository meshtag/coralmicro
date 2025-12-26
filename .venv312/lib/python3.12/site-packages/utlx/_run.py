# Copyright (c) 2012 Adam Karpierz
# SPDX-License-Identifier: Zlib

from typing import TYPE_CHECKING, TypeAlias, Any
from collections.abc import Iterable
import subprocess
import logging

__all__ = ('run',)

log = logging.getLogger(__name__)


class run:
    """Unified subprocess interface with namespaced constants and callable execution."""

    if TYPE_CHECKING:
        CompletedProcess:      TypeAlias = subprocess.CompletedProcess[str | bytes]
        CompletedTextProcess:  TypeAlias = subprocess.CompletedProcess[str]
        CompletedBytesProcess: TypeAlias = subprocess.CompletedProcess[bytes]
    else:
        CompletedProcess      = subprocess.CompletedProcess
        CompletedTextProcess  = subprocess.CompletedProcess
        CompletedBytesProcess = subprocess.CompletedProcess

    PIPE    = subprocess.PIPE
    STDOUT  = subprocess.STDOUT
    DEVNULL = subprocess.DEVNULL

    SubprocessError    = subprocess.SubprocessError
    TimeoutExpired     = subprocess.TimeoutExpired
    CalledProcessError = subprocess.CalledProcessError

    class SafeString(str):
        """Marks sensitive arguments to be masked in logs."""

    def __new__(cls, *args: Any,  # type: ignore[misc]
                start_terminal_window: bool = False, **kwargs: Any) -> CompletedProcess:
        """Runs the command described by `args`.

        Waits for the command to complete and returns a run.CompletedProcess instance.

        Args:
            args: Command-line arguments passed to `subprocess.run`.
            start_terminal_window: If True, starts the command in a separate console
                                   window (server mode).
            kwargs: Additional keyword arguments controlling subprocess execution.

        Returns:
            run.CompletedProcess: Result of the executed command.
        """
        if start_terminal_window:  # pragma: no cover
            args = ("cmd.exe", "/C", "start", *args)
        output: run.CompletedProcess = subprocess.run(
            [str(arg) for arg in args], check=kwargs.pop("check", True), **kwargs)
        print_cmd = [("*****" if isinstance(arg, cls.SafeString) else arg) for arg in args]
        log.debug("cmd=%s, returncode=%d", str(print_cmd), int(output.returncode))
        return output

    @staticmethod
    def split_kwargs(kwargs: dict[str, Any], forbidden_kwargs: Iterable[str]) \
            -> tuple[dict[str, Any], dict[str, Any]]:
        allowed_kwargs  = {key: val for key, val in kwargs.items()
                           if key not in forbidden_kwargs}
        reserved_kwargs = {key: val for key, val in kwargs.items()
                           if key in forbidden_kwargs}
        return (allowed_kwargs, reserved_kwargs)
