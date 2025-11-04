from scpviz.TrackedDataFrame import TrackedDataFrame 
import pandas as pd
import anndata as ad

from scpviz.utils import format_log_prefix

class SummaryMixin:
    """
    Handles creation, synchronization, and metric updates for the `.summary` attribute.

    This mixin maintains a unified sample-level summary table by merging `.prot.obs` and `.pep.obs`,
    with automatic flagging and update mechanisms to track when recomputation or syncing is needed.

    Features:
    
    - Merges sample-level metadata from `.prot.obs` and `.pep.obs` into `.summary`
    - Tracks when `.summary` becomes out of sync via `_summary_is_stale`
    - Supports recomputing per-sample metrics and syncing edits back to `.obs`
    - Enables passive refresh of `.summary` after filtering or manual editing

    Functions:
        update_summary: Rebuild and optionally recompute or sync `.summary`
        _update_summary: Legacy alias for `update_summary()`
        _merge_obs: Internal merge logic for `.prot.obs` and `.pep.obs`
        _push_summary_to_obs: Sync edited `.summary` values back into `.obs`
        _mark_summary_stale: Mark the summary as stale for downstream tracking
    """
    def update_summary(self, recompute=True, sync_back=False, verbose=True):
        """
        Update the `.summary` DataFrame to reflect current state of `.obs` and metadata.

        This function ensures `.summary` stays synchronized with sample-level metadata
        stored in `.prot.obs` / `.pep.obs`. You can choose to recompute metrics,
        sync edits back to `.obs`, or simply refresh the merged view.

        Args:
            recompute (bool): If True, re-calculate protein/peptide stats.
            sync_back (bool): If True, push edited `.summary` values back to `.prot.obs` / `.pep.obs`.
                False by default, as `.summary` is derived.
            verbose (bool): If True, print action messages.

        ??? example "Typical Usage Scenarios"
            | Scenario                        | Call                                | recompute | sync_back | _summary_is_stale | Effect                                                           |
            |---------------------------------|-------------------------------------|-----------|-----------|-------------------|------------------------------------------------------------------|
            | Filtering `.prot` or `.pep`     | `.update_summary(recompute=True)`   | ✅        | ❌        | ❌                | Recalculate protein/peptide stats and merge into `.summary`.     |
            | Filtering samples               | `.update_summary(recompute=False)`  | ❌        | ❌        | ❌                | Refresh `.summary` view of `.obs` without recomputation.         |
            | Manual `.summary[...] = ...`    | `.update_summary()`                 | ✅/❌     | ✅        | ✅                | Push edited `.summary` values back to `.obs`.                    |
            | After setting `.summary = ...`  | `.update_summary()`                 | ✅        | ✅        | ✅                | Sync back and recompute stats from new `.summary`.               |
            | No changes                      | `.update_summary()`                 | ❌        | ❌        | ❌                | No-op other than passive re-merge.                               |

        Note:
            - For most typical use cases, we auto-detect which flags need to be applied.
                You usually don’t need to set `recompute` or `sync_back` manually.
            - `recompute=True` triggers `_update_metrics()` from `.prot` / `.pep` data.
            - `sync_back=True` ensures changes to `.summary` are reflected in `.obs`.
            - `.summary_is_stale` is automatically set when `.summary` is edited directly
            (e.g. via `TrackedDataFrame`) or when assigned via the setter.
        """

        # 1. Push back first if summary was edited by the user
        if sync_back or getattr(self, "_summary_is_stale", False):
            updated_prot, updated_pep = self._push_summary_to_obs()
            updated_cols = list(set(updated_prot + updated_pep))
            updated_str = f" Columns updated: {', '.join(updated_cols)}." if updated_cols else ""

            if verbose:
                reason = " (marked stale)" if not sync_back else ""
                print(f"{format_log_prefix('update',indent=1)} Updating summary [sync_back]: pushed edits from `.summary` to `.obs`{reason}.\n{format_log_prefix('blank',indent=2)}{updated_str}")

            self._summary_is_stale = False  # reset before recompute

        # 2. Recompute or re-merge afterward
        if recompute:
            self._update_metrics() # type: ignore #, in MetricsMixin
        self._merge_obs()
        self._update_summary_metrics() # type: ignore #, in MetricsMixin
        self.refresh_identifier_maps() # type: ignore #, in IdentifierMixin

        # 3. Final messaging
        if verbose and not (sync_back or self._summary_is_stale):
            if recompute:
                print(f"{format_log_prefix('update',indent=3)} Updating summary [recompute]: Recomputed metrics and refreshed `.summary` from `.obs`.")
            else:
                print(f"{format_log_prefix('update',indent=3)} Updating summary [refresh]: Refreshed `.summary` view (no recompute).")

        # 4. Final cleanup
        self._summary_is_stale = False

    def _update_summary(self):
        """
        Legacy method for updating the `.summary` table.

        This method is retained for backward compatibility and simply calls the newer
        `update_summary()` function with default arguments:
        `recompute=True`, `sync_back=False`, and `verbose=False`.

        Note:
            This method is deprecated and may be removed in a future version.
            Use `update_summary()` instead.
        """
        print("⚠️  Legacy _update_summary() called — consider switching to update_summary()")
        self.update_summary(recompute=True, sync_back=False, verbose=False)

    def _merge_obs(self):
        """
        Merge `.prot.obs` and `.pep.obs` into a unified sample-level summary.

        This function combines metadata from protein-level and peptide-level `.obs` tables
        into a single summary DataFrame. Shared columns (e.g., 'gradient', 'condition') 
        are taken from `.prot.obs` by default if present in both.

        Returns:
            pandas.DataFrame: Merged observation metadata for all samples.
        """
        if self.prot is not None:
            summary = self.prot.obs.copy()
            if self.pep is not None:
                for col in self.pep.obs.columns:
                    if col not in summary.columns:
                        summary[col] = self.pep.obs[col]
        elif self.pep is not None:
            summary = self.pep.obs.copy()
        else:
            summary = pd.DataFrame()

        
        self._summary = TrackedDataFrame(
            summary, parent=self, mark_stale_fn=self._mark_summary_stale)
        self._previous_summary = summary.copy()

    def _push_summary_to_obs(self, skip_if_contains='pep', verbose=False):
        """
        Push changes from `.summary` back into `.prot.obs` and `.pep.obs`.

        This function updates `.prot.obs` and `.pep.obs` with any modified columns
        in `.summary`. To avoid overwriting incompatible fields, columns containing
        `skip_if_contains` are excluded when updating `.prot.obs`, and similarly,
        columns containing 'prot' are excluded when updating `.pep.obs`.

        Args:
            skip_if_contains (str): Substring used to skip incompatible columns for `.prot.obs`.
                                    Defaults to 'pep'.
            verbose (bool): If True, print updates being pushed to `.obs`.

        Note:
            This is typically called internally by `update_summary(sync_back=True)`.
        """
        if not self._has_data():
            return

        def update_obs_with_summary(obs, summary, skip_if_contains):
            skipped, updated = [], []
            for col in summary.columns:
                if skip_if_contains in str(col):
                    skipped.append(col)
                    continue
                if col not in obs.columns or not obs[col].equals(summary[col]):
                    updated.append(col)
                obs[col] = summary[col]
            return skipped, updated

        if self.prot is not None:
            if not self.prot.obs.index.equals(self._summary.index):
                raise ValueError("Mismatch: .summary and .prot.obs have different sample indices.")
            skipped_prot, updated_prot = update_obs_with_summary(self.prot.obs, self._summary, skip_if_contains)
        else:
            skipped_prot, updated_prot = None, []

        if self.pep is not None:
            if not self.pep.obs.index.equals(self._summary.index):
                raise ValueError("Mismatch: .summary and .pep.obs have different sample indices.")
            skipped_pep, updated_pep = update_obs_with_summary(self.pep.obs, self._summary, skip_if_contains='prot')
        else:
            skipped_pep, updated_pep = None, []

        msg = "Pushed summary values back to obs. "
        if skipped_prot:
            msg += f"Skipped for prot: {', '.join(skipped_prot)}. "
        if skipped_pep:
            msg += f"Skipped for pep: {', '.join(skipped_pep)}. "

        self._append_history(msg)
        if verbose:
            print(msg)

        return updated_prot, updated_pep

    def _mark_summary_stale(self):
        """
        Mark the `.summary` as stale.

        This sets the `_summary_is_stale` flag to True, indicating that the
        summary is out of sync with `.obs` or metrics and should be updated
        using `update_summary()`.

        Note:
            This is typically triggered automatically when `.summary` is edited
            (e.g., via `TrackedDataFrame`) or reassigned.
        """
        self._summary_is_stale = True