from typing import Protocol

class HasHistory(Protocol):
    """
    Tracks the operation history applied to the pAnnData object.

    Functions:
        _append_history: Adds a string to the transformation history log.
        print_history: Prints the current transformation history.
    """
    _history: list

class HistoryMixin:
    def _append_history(self: HasHistory, action):
        self._history.append(action)

    def print_history(self: HasHistory):
        formatted_history = "\n".join(f"{i}: {action}" for i, action in enumerate(self._history, 1))
        print("-------------------------------\nHistory:\n-------------------------------\n"+formatted_history)
