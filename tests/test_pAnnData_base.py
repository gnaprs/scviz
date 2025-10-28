import pytest
import copy
from datetime import datetime

def test_has_data_true(pdata):
    """Ensure _has_data returns True when .prot or .pep is populated."""
    assert pdata._has_data(), "Expected _has_data() to be True for initialized pAnnData"

def test_copy_includes_cached_maps(pdata):
    """Test that .copy() deep-copies optional cached maps if present."""
    # Add dummy cached maps
    pdata._gene_maps_protein = {"A0A0": "GENE_A", "B0B0": "GENE_B"}
    pdata._protein_maps_peptide = {"P1": ["PEP1", "PEP2"]}

    # Make a copy
    pdata_copy = pdata.copy()

    # Check new object is distinct
    assert pdata_copy is not pdata
    assert pdata_copy.prot is not pdata.prot
    assert pdata_copy.pep is not pdata.pep

    # Both cached maps should exist and be equal but independent
    assert hasattr(pdata_copy, "_gene_maps_protein")
    assert hasattr(pdata_copy, "_protein_maps_peptide")
    assert pdata_copy._gene_maps_protein == pdata._gene_maps_protein
    assert pdata_copy._protein_maps_peptide == pdata._protein_maps_peptide

    # Ensure they are deep copies, not same reference
    assert pdata_copy._gene_maps_protein is not pdata._gene_maps_protein
    assert pdata_copy._protein_maps_peptide is not pdata._protein_maps_peptide

    # Changing the copy should not affect the original
    pdata_copy._gene_maps_protein["A0A0"] = "CHANGED"
    assert pdata._gene_maps_protein["A0A0"] == "GENE_A"

def test_copy_handles_missing_cached_maps(pdata):
    """Test .copy() still works safely when cached maps are absent."""
    # Ensure attributes are missing
    if hasattr(pdata, "_gene_maps_protein"):
        del pdata._gene_maps_protein
    if hasattr(pdata, "_protein_maps_peptide"):
        del pdata._protein_maps_peptide

    pdata_copy = pdata.copy()
    assert not hasattr(pdata_copy, "_gene_maps_protein")
    assert not hasattr(pdata_copy, "_protein_maps_peptide")

def test_history_initialization(pdata):
    """Ensure _history attribute initializes properly."""
    assert hasattr(pdata, "_history"), "❌ pdata object should have a _history attribute"
    assert isinstance(pdata._history, list), "❌ _history should be a list"
    assert all(isinstance(e, str) for e in pdata._history), "❌ All _history entries should be strings"
    assert any("Imported" in e for e in pdata._history), "❌ Expected import action in initial history"
    print(f"✅ Initial history has {len(pdata._history)} entries (e.g. import recorded)")

def test_history_appends_after_operations(pdata):
    """Test that normalize() and impute() append to the history."""
    prev_len = len(pdata._history)
    pdata.normalize()
    pdata.impute()

    # Ensure history grew
    assert len(pdata._history) > prev_len, "❌ Expected new history entries after normalize() and impute()"

    # Allow both timestamped and plain entries
    new_entries = pdata._history[prev_len:]
    for entry in new_entries:
        assert isinstance(entry, str), "❌ History entries should be strings"
        # Either timestamped or contains expected keywords
        assert (entry.startswith("[") and "]" in entry) or any(
            kw in entry.lower() for kw in ["normalize", "impute", "layer"]
        ), f"❌ Unexpected history format: {entry}"

    print(f"✅ {len(new_entries)} new entries added after normalize() + impute()")

def test_print_history_output(pdata, capsys):
    """Test that print_history prints all actions with correct formatting."""
    pdata.normalize()
    pdata.impute()
    pdata.print_history()

    captured = capsys.readouterr()
    out = captured.out

    # ✅ Check for header
    assert "History:" in out, "❌ print_history() missing header"
    # ✅ Check entries are numbered
    assert "1:" in out, "❌ print_history() should number entries"
    # ✅ Check actions are listed
    assert "normalize" in out.lower() or "impute" in out.lower(), "❌ Expected operation names in printed history"

def test_describe_rs_returns_dataframe(pdata):
    """Ensure describe_rs() returns a DataFrame with expected columns and shape."""
    import pandas as pd
    import numpy as np
    
    summary = pdata.describe_rs()

    # Basic structure
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {"peptides_per_protein", "unique_peptides"}
    assert len(summary) == pdata.prot.shape[1]

    # Columns should be numeric
    assert np.issubdtype(summary["peptides_per_protein"].dtype, np.number)
    assert np.issubdtype(summary["unique_peptides"].dtype, np.number)

    # Index matches protein var names
    assert all(summary.index == pdata.prot.var_names)


def test_describe_rs_handles_missing_rs(pdata_nopep, capsys):
    """If rs is None, should print warning and return None."""
    result = pdata_nopep.describe_rs()
    captured = capsys.readouterr().out

    assert result is None
    assert "No RS matrix" in captured

import matplotlib.pyplot as plt

def test_plot_rs_runs_without_error(pdata):
    """Ensure plot_rs() executes successfully with valid RS."""
    import matplotlib.pyplot as plt
    plt.close("all")
    pdata.plot_rs(figsize=(4, 2))

def test_plot_rs_handles_missing_rs(pdata_nopep, capsys):
    """If rs is None, should print warning and not plot."""
    plt.close("all")
    pdata_nopep.plot_rs()
    captured = capsys.readouterr().out

    assert "No RS matrix" in captured
    assert not plt.get_fignums(), "No figure should be created when RS is None"