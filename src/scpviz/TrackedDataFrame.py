import pandas as pd

class TrackedDataFrame(pd.DataFrame):
    """
    A subclass of :class:`pandas.DataFrame` that integrates with a parent 
    :class:`pAnnData` object to track when derived tables (e.g., `.summary`) 
    have been modified outside the canonical workflow.

    Any in-place modifications automatically mark the parent object as "stale"
    using the provided callback (`mark_stale_fn`). This ensures downstream
    code can detect unsynchronized changes and prompt recomputation if needed.

    Attributes:
        _parent (pAnnData): The parent object associated with this DataFrame.
        _mark_stale_fn (callable): Function called when the DataFrame is modified.
        _raw_loc, _raw_iloc: Direct accessors for untracked indexing (safe use).

    !!! warning "Internal Utility"
        `TrackedDataFrame` is primarily intended for internal use within `pAnnData`.  
        Direct use in analysis code is not recommended, as stale-tracking may
        interfere with expected pandas behaviors.

    !!! tip
        Use `.raw_loc` and `.raw_iloc` to bypass stale-marking when read-only
        access is explicitly desired.

    """
    _metadata = ["_parent", "_mark_stale_fn"]

    @property
    def _constructor(self):
        return TrackedDataFrame

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, parent=None, mark_stale_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent
        self._mark_stale_fn = mark_stale_fn
        # Bind raw .loc/.iloc for safe use
        self._raw_loc = super().loc
        self._raw_iloc = super().iloc

    def __repr__(self):
        base = super().__repr__()
        stale_msg = ""
        if getattr(self, "_mark_stale_fn", None) and getattr(self._parent, "_summary_is_stale", False):
            stale_msg = "\n⚠️  [TrackedDataFrame] This summary has been modified and is not synced back to .obs."
        return base + stale_msg

    def _mark_stale(self):
        """
        Mark the parent pAnnData object as stale.

        Called internally before any modifying operations (e.g. `__setitem__`,
        `drop`, `assign`). Triggers the parent’s `_mark_stale_fn` if provided.
        """
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

    @property
    def loc(self):
        self._mark_stale()
        return super().loc

    @property
    def iloc(self):
        self._mark_stale()
        return super().iloc