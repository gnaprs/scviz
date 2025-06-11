import pandas as pd

class TrackedDataFrame(pd.DataFrame):
    """
    A subclass of DataFrame that marks its parent pAnnData object as 'stale'
    when it's modified directly.
    """
    _metadata = ["_parent", "_mark_stale_fn"]

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, parent=None, mark_stale_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent
        self._mark_stale_fn = mark_stale_fn

    def __repr__(self):
        base = super().__repr__()
        stale_msg = ""
        if getattr(self, "_mark_stale_fn", None) and getattr(self._parent, "_summary_is_stale", False):
            stale_msg = "\n⚠️  [TrackedDataFrame] This summary has been modified and is not synced back to .obs."
        return base + stale_msg

    def _mark_stale(self):
        if self._mark_stale_fn is not None:
            self._mark_stale_fn()

    def __setitem__(self, key, value):
        self._mark_stale()
        return super().__setitem__(key, value)

    def drop(self, *args, **kwargs):
        self._mark_stale()
        return super().drop(*args, **kwargs)

    def assign(self, *args, **kwargs):
        self._mark_stale()
        return super().assign(*args, **kwargs)

    def pop(self, *args, **kwargs):
        self._mark_stale()
        return super().pop(*args, **kwargs)
