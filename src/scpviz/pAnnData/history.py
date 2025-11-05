from typing import Protocol
from datetime import datetime

class HasHistory(Protocol):
    """
    Tracks the operation history applied to the pAnnData object.

    Functions:
        _append_history: Adds a string to the transformation history log.
        print_history: Prints the current transformation history.
    """
    _history: list

class HistoryMixin:
    """
    Mixin for tracking the history of operations performed on a pAnnData object.

    This mixin provides simple utilities to log and review transformations or
    analysis steps performed on the data object. Each action is stored as a 
    string in the internal `_history` list.

    Features:

    - Track transformations or analysis steps as text entries
    - Print chronological history of actions performed

    Functions:
        _append_history: Add a custom string to the internal history list  
        print_history: Print the full history in a formatted list
    """
    def _append_history(self: HasHistory, action):
        """
        Append a timestamped entry to the internal transformation history log.

        Each entry records the current date and time alongside a string description 
        of the action performed. Useful for tracking the sequence and timing of 
        transformations on the pAnnData object.

        Args:
            action (str): Description of the action performed.
        """
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self._history.append(f"{timestamp} {action}")

    def print_history(self: HasHistory): 
        """
        Print the current transformation history in a numbered format.

        Each logged action is printed with its index, showing the chronological
        sequence of operations applied to the object.
        """
        formatted_history = "\n".join(f"{i}: {action}" for i, action in enumerate(self._history, 1))
        print("-------------------------------\nHistory:\n-------------------------------\n" + formatted_history)
