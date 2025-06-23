from scviz.TrackedDataFrame import TrackedDataFrame 
import pandas as pd
import anndata as ad

from scviz.utils import format_log_prefix

class SummaryMixin:
    """
    Manages `.summary` generation and synchronization with sample metadata.

    Functions:
        update_summary: Recomputes `.summary` with current metrics.
        _update_summary: Builds the internal `.summary` table.
        _merge_obs: Merges `.obs` from `.prot` and `.pep` into a unified summary.
        _push_summary_to_obs: Pushes `.summary` back into `.obs`.
        update_obs_with_summary: Updates `.obs` selectively with new `.summary` columns.
    """
    def update_summary(self, recompute=True, sync_back=False, verbose=True):
        """
        Update the `.summary` DataFrame to reflect current state of `.obs` and metadata.

        This function ensures `.summary` stays synchronized with sample-level metadata
        stored in `.prot.obs` / `.pep.obs`. You can choose to recompute metrics,
        sync edits back to `.obs`, or simply refresh the merged view.

        Parameters:
        - recompute (bool): If True, re-calculate protein/peptide stats.
        - sync_back (bool): If True, push edited .summary values back to .prot.obs / .pep.obs. False by default, as .summary is derived.
        - verbose (bool): If True, print action messages.

        Typical Usage Scenarios:
        ------------------------

        | Scenario                        | Call                         | recompute | sync_back | _summary_is_stale | Effect                                                           |
        |---------------------------------|------------------------------|-----------|-----------|-------------------|------------------------------------------------------------------|
        | Filtering `.prot` or `.pep`     | `.update_summary(True)`      | True      | False     | False             | Recalculate protein/peptide stats and merge into `.summary`.     |
        | Filtering samples               | `.update_summary(False)`     | False     | False     | False             | Refresh `.summary` view of `.obs` without recomputation.         |
        | Manual `.summary[...] = ...`    | `.update_summary()`          | True/False| True      | True              | Push edited `.summary` values back to `.obs`.                    |
        | After setting `.summary = ...`  | `.update_summary()`          | True      | True      | True              | Sync back and recompute stats from new `.summary`.               |
        | No changes                      | `.update_summary()`          | False     | False     | False             | No-op other than passive re-merge.                               |

        Notes:
        ------
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
            self._update_metrics()
        self._merge_obs()
        self._update_summary_metrics()
        self.refresh_identifier_maps()

        # 3. Final messaging
        if verbose and not (sync_back or self._summary_is_stale):
            if recompute:
                print(f"{format_log_prefix('update',indent=3)} Updating summary [recompute]: Recomputed metrics and refreshed `.summary` from `.obs`.")
            else:
                print(f"{format_log_prefix('update',indent=3)} Updating summary [refresh]: Refreshed `.summary` view (no recompute).")

        # 4. Final cleanup
        self._summary_is_stale = False

    def _update_summary(self):
        print("⚠️  Legacy _update_summary() called — consider switching to update_summary()")
        self.update_summary(recompute=True, sync_back=False, verbose=False)

    def _merge_obs(self):
        """
        Merge .prot.obs and .pep.obs into a single summary DataFrame.
        Shared columns (e.g. 'gradient', 'condition') are kept from .prot by default.
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
        Push changes from .summary back into .prot.obs and .pep.obs.
        Columns containing `skip_if_contains` are ignored for .prot; same for 'prot' in .pep.
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

    def update_obs_with_summary(self):
        pass

    def _mark_summary_stale(self):
        self._summary_is_stale = True