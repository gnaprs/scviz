import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# unit tests for .summary staleness, manual edits, auto-sync behaviour:
def test_summary_stale_flag_after_manual_edit(pdata):
    # update_summary() clears stale flag
    pdata.update_summary(recompute=True)
    assert not pdata._summary_is_stale

    # Manual edit triggers stale flag
    pdata.summary["new_col"] = np.arange(pdata.summary.shape[0])
    assert pdata._summary_is_stale

def test_summary_setter_triggers_sync_back(pdata):
    # Re-assigning .summary = ... triggers sync back
    pdata.update_summary()
    new_summary = pdata.summary.copy()
    new_summary["test_col"] = ["X"] * new_summary.shape[0]
    pdata.summary = new_summary  # triggers sync_back via setter

    # Check that the column was pushed into obs
    assert "test_col" in pdata.prot.obs.columns
    assert (pdata.prot.obs["test_col"] == "X").all()

def test_auto_sync_back_to_obs(pdata):
    # Stale summary triggers auto-sync
    pdata.update_summary()

    # Modify .summary manually
    pdata.summary["condition"] = ["A"] * pdata.summary.shape[0]

    # This should trigger auto sync (check .prot.obs)
    pdata.update_summary(recompute=False)

    assert "condition" in pdata.prot.obs.columns
    assert (pdata.prot.obs["condition"] == "A").all()
    assert not pdata._summary_is_stale

def test_no_sync_when_not_stale(pdata):
    # Clean summary does not sync again
    pdata.update_summary()
    prot_obs_before = pdata.prot.obs.copy()

    # No manual edits, no recompute
    pdata.update_summary(recompute=False)

    pd.testing.assert_frame_equal(prot_obs_before, pdata.prot.obs)
