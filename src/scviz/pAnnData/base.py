import copy

from scviz.utils import format_log_prefix

class BaseMixin:
    """
    Core base methods for pAnnData.

    This mixin provides essential utility and management functions for cloning, 
    checking, and managing core attributes of a `pAnnData` object. These methods
    serve as foundational building blocks for other mixins and functions.

    Features:
    
    - Checks presence of data (.prot or .pep)
    - Safe object copying with state preservation
    - Internal metadata management (stats, history, summary)

    Functions:
        _has_data: Check whether .prot and/or .pep data are present  
        copy: Return a new `pAnnData` object with retained internal state
    """
    def _has_data(self) -> bool:
        """
        Check whether the pAnnData object contains either protein or peptide data.

        Returns:
            bool: True if either .prot or .pep is not None; otherwise False.
        """
        return self.prot is not None or self.pep is not None # type: ignore[attr-defined]

    def copy(self):
        """
        Return a new `pAnnData` object with the current state of all components.

        This method performs a shallow copy of core data (.prot, .pep) and a deep copy of internal attributes
        (e.g., RS matrix, summary, stats, and cached maps). It avoids full deepcopy for efficiency and retains
        the current filtered or processed state of the object.

        Returns:
            pAnnData: A new object containing copies of the current data and metadata.
        """
        new_obj = self.__class__.__new__(self.__class__)

        # Copy core AnnData components
        new_obj.prot = self.prot.copy() if self.prot is not None else None # type: ignore[attr-defined]
        new_obj.pep = self.pep.copy() if self.pep is not None else None # type: ignore[attr-defined]
        new_obj._rs = copy.deepcopy(self._rs) # type: ignore[attr-defined]

        # Copy summary and stats
        new_obj._stats = copy.deepcopy(self._stats) # type: ignore[attr-defined]
        new_obj._history = copy.deepcopy(self._history) # type: ignore[attr-defined]
        new_obj._previous_summary = copy.deepcopy(self._previous_summary) # type: ignore[attr-defined]
        new_obj._suppress_summary_log = True # type: ignore[attr-defined]
        new_obj.summary = self._summary.copy(deep=True) if self._summary is not None else None # go through setter to mark as stale, # type: ignore[attr-defined]
        del new_obj._suppress_summary_log # type: ignore[attr-defined]

        # Optional: cached maps
        if hasattr(self, "_gene_maps_protein"):
            new_obj._gene_maps_protein = copy.deepcopy(self._gene_maps_protein) # type: ignore[attr-defined]
        if hasattr(self, "_protein_maps_peptide"):
            new_obj._protein_maps_peptide = copy.deepcopy(self._protein_maps_peptide) # type: ignore[attr-defined]

        return new_obj

    def compare_current_to_raw(self, on="protein"):
        """
        Compare current pdata object to original raw data, showing how many samples and features were dropped.
        Compares current obs/var names to the original raw data (stored in .uns).

        Args:
            on (str): Dataset to compare ('protein' or 'peptide').

        Returns:
            dict: Dictionary summarizing dropped samples and features.
        """
        print(f"{format_log_prefix('user', 1)} Comparing current pdata to X_raw [{on}]:")

        adata = getattr(self, "prot" if on == "protein" else "pep", None)
        if adata is None:
            print(f"{format_log_prefix('warning', 2)} No {on} data found.")
            return None

        orig_obs = set(adata.uns.get("X_raw_obs_names", []))
        orig_var = set(adata.uns.get("X_raw_var_names", []))
        current_obs = set(adata.obs_names)
        current_var = set(adata.var_names)

        dropped_obs = sorted(list(orig_obs - current_obs))
        dropped_var = sorted(list(orig_var - current_var))

        print(f"   → Samples dropped: {len(dropped_obs)}")
        print(f"   → Features dropped: {len(dropped_var)}")

        return {"dropped_samples": dropped_obs, "dropped_features": dropped_var}

    def get_X_raw_aligned(self, on="protein"):
        """
        Return X_raw subset aligned to current obs/var order.

        Returns:
            np.ndarray: Subset of X_raw matching current AnnData.
        """
        import numpy as np
        print(f"{format_log_prefix('user_only', 1)} Returning X_raw subset aligned to current obs/var order [{on}]:")
        adata = getattr(self, "prot" if on == "protein" else "pep", None)
        if adata is None or "X_raw" not in adata.layers:
            raise ValueError(f"No raw layer found for {on} data.")

        X_raw = adata.layers["X_raw"]
        orig_obs = adata.uns["X_raw_obs_names"]
        orig_var = adata.uns["X_raw_var_names"]

        obs_idx = [orig_obs.index(o) for o in adata.obs_names if o in orig_obs]
        var_idx = [orig_var.index(v) for v in adata.var_names if v in orig_var]

        return X_raw[np.ix_(obs_idx, var_idx)]
