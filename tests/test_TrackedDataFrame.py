import pandas as pd
import pytest
from scviz.TrackedDataFrame import TrackedDataFrame

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

class DummyParent:
    def __init__(self):
        self._summary_is_stale = False
        self.stale_called = False

    def mark_stale(self):
        self._summary_is_stale = True
        self.stale_called = True

@pytest.fixture
def tdf():
    parent = DummyParent()
    return TrackedDataFrame(
        {"A": [1, 2, 3], "B": [4, 5, 6]},
        parent=parent,
        mark_stale_fn=parent.mark_stale
    )

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_mark_stale_function_called(tdf):
    """_mark_stale() should trigger parent's callback."""
    parent = tdf._parent
    tdf._mark_stale()
    assert parent.stale_called
    assert parent._summary_is_stale

def test_setitem_triggers_mark_stale(tdf):
    parent = tdf._parent
    tdf["C"] = [7, 8, 9]
    assert parent.stale_called
    assert "C" in tdf.columns

def test_drop_triggers_mark_stale(tdf):
    parent = tdf._parent
    tdf.drop(columns=["A"])
    assert parent.stale_called

def test_assign_triggers_mark_stale(tdf):
    parent = tdf._parent
    tdf.assign(D=tdf["A"] * 2)
    assert parent.stale_called

def test_pop_triggers_mark_stale(tdf):
    parent = tdf._parent
    tdf.pop("A")
    assert parent.stale_called

def test_loc_triggers_mark_stale(tdf):
    parent = tdf._parent
    _ = tdf.loc[0]  # access
    assert parent.stale_called

def test_iloc_triggers_mark_stale(tdf):
    parent = tdf._parent
    _ = tdf.iloc[0]
    assert parent.stale_called

def test_repr_includes_warning_when_stale(tdf):
    parent = tdf._parent
    parent._summary_is_stale = True
    rep = repr(tdf)
    assert "⚠️" in rep
    assert "not synced back" in rep

def test_raw_loc_and_iloc_do_not_mark_stale(tdf):
    """Ensure ._raw_loc and ._raw_iloc allow read-only access without marking stale."""
    parent = tdf._parent
    parent.stale_called = False

    # Access a row and a cell through raw loc/iloc
    _ = tdf._raw_loc[0, "A"]
    _ = tdf._raw_iloc[0, 0]

    # Parent should not have been marked stale
    assert not parent.stale_called, "raw_loc or raw_iloc should not trigger stale marking"
