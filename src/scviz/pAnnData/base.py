import copy

class BaseMixin:
    """
    Core base functions for the pAnnData object.

    Functions:
        _has_data: Returns True if either `.prot` or `.pep` is not None.
        _copy: Returns a deep copy of the pAnnData object.
    """
    def _has_data(self) -> bool:
        """Check whether .prot and .pep data are available."""
        return self.prot is not None or self.pep is not None # type: ignore[attr-defined]

    def copy(self):
        """
        Return a new pAnnData object with the current state of all components.
        This avoids full deepcopy and ensures filtered state is retained.
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

