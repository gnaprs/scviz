from .base import BaseMixin
from .validation import ValidationMixin
from .summary import SummaryMixin
from .metrics import MetricsMixin
from .identifier import IdentifierMixin
from .history import HistoryMixin
from .editing import EditingMixin
from .filtering import FilterMixin
from .analysis import AnalysisMixin
from .enrichment import EnrichmentMixin
from .io import import_data

import pandas as pd
import anndata as ad
from scviz.TrackedDataFrame import TrackedDataFrame 

class pAnnData(BaseMixin, ValidationMixin, SummaryMixin, MetricsMixin,
               IdentifierMixin, HistoryMixin, EditingMixin, FilterMixin,
               AnalysisMixin, EnrichmentMixin):
    """
    Unified data container for protein and peptide expression in single-cell and bulk proteomics.

    `pAnnData` integrates matched protein-level and peptide-level `AnnData` objects, along with an optional
    binary relational structure (`rs`) that maps peptides to their parent proteins. It is designed for single-cell and bulk
    proteomics data, and supports a range of analysis workflows, including filtering, normalization, imputation, differential
    expression, enrichment, and visualization.

    This class is composed of modular mixins to enhance maintainability and organization.

    ## Mixins

    - **BaseMixin**: Core internal utilities, copying, and simple logic.
    - **ValidationMixin**: Ensures structural and dimensional consistency across `.prot`, `.pep`, `.summary`, and `rs`.
    - **SummaryMixin**: Maintains `.summary`, synchronizes metadata, and caches per-sample metrics.
    - **MetricsMixin**: Computes descriptive statistics from expression data and relational structure (RS matrix).
    - **IdentifierMixin**: Manages bidirectional gene/accession mappings and handles missing gene resolution via UniProt.
    - **HistoryMixin**: Tracks all operations performed on the object for transparency.
    - **EditingMixin**: Supports in-place editing, direct manipulation of expression matrices, and data export.
    - **FilterMixin**: Provides flexible filtering of samples and proteins/peptides based on metadata, presence, or quantification.
    - **AnalysisMixin**: Core statistical operations: differential expression, imputation, PCA, clustering, etc.
    - **EnrichmentMixin**: Runs STRING-based enrichment analyses (GO, pathways, PPI) using ranked or unranked protein sets.

    Args:
        prot (AnnData): Protein-level expression matrix, with `.obs` containing sample metadata and `.var` describing protein features.

        pep (AnnData): Peptide-level expression matrix, structured analogously to `prot`.

        rs (np.ndarray or sparse.spmatrix, optional):
            Binary relational matrix (proteins × peptides), where non-zero entries indicate a parent-protein relationship.

        summary (pd.DataFrame, optional):
            Sample-level metadata table, merged from `.prot.obs` and `.pep.obs`, with support for additional metrics.

        stats (dict, optional):
            Dictionary for storing analysis outputs such as DE results, imputation metadata, and enrichment summaries.

        history (list of str, optional):
            Chronological list of user-invoked operations, automatically tracked for reproducibility.

    Todo:
        Decide whether to standardize internal terminology to `classes` or `class_types` for sample-level grouping.
    """
    def __init__(self, 
                 prot = None, # np.ndarray | sparse.spmatrix 
                 pep = None, # np.ndarray | sparse.spmatrix
                 rs = None): # np.ndarray | sparse.spmatrix, protein x peptide relational data

        self._prot = ad.AnnData(prot) if prot is not None else None
        self._pep = ad.AnnData(pep) if pep is not None else None
        self._rs = None
        if rs is not None:
            self._set_RS(rs) # Defined in the EditingMixin

        # Internal attributes
        self._history = []
        self._summary = pd.DataFrame()
        self._stats = {}
        self._summary_is_stale = False

    @classmethod
    def import_data(cls, *args, **kwargs):
        """
        Import data from a file or URL.
        """
        return import_data(*args, **kwargs)
    
    def __repr__(self):
        def format_ann(adata, label):
            if adata is None:
                return f"{label}: None"

            shape_str = f"{adata.shape[0]} files × {adata.shape[1]} {label.lower()}s"
            obs_cols = ', '.join(adata.obs.columns[:5]) + ('...' if len(adata.obs.columns) > 5 else '')
            var_cols = ', '.join(adata.var.columns[:5]) + ('...' if len(adata.var.columns) > 5 else '')
            obsm_keys = ', '.join(adata.obsm.keys())
            layers_keys = ', '.join(adata.layers.keys())

            return (f"{label} (shape: {shape_str})\n"
                    f"  obs:    {obs_cols or '—'}\n"
                    f"  var:    {var_cols or '—'}\n"
                    f"  obsm:   {obsm_keys or '—'}\n"
                    f"  layers: {layers_keys or '—'}")
            
        def format_rs(rs):
            if rs is None:
                return "RS: None"
            return f"RS (shape: {rs.shape[0]} proteins × {rs.shape[1]} peptides)"
        
        lines = [
            "pAnnData object",
            format_ann(self.prot, "Protein"),
            format_ann(self.pep, "Peptide"),
            format_rs(self._rs)
        ]
        return "\n\n".join(lines)

    # -----------------------------
    # Properties (GETTERS)
    @property
    def prot(self):
        return self._prot

    @property
    def pep(self):
        return self._pep

    @property
    def rs(self):
        return self._rs

    @property
    def history(self):
        return self._history

    @property
    def summary(self):
        if not hasattr(self, "_summary"):
            raise AttributeError("Summary has not been initialized.")
        if getattr(self, "_summary_is_stale", False):
            print("[summary] ⚠️ Warning: .summary has been modified. Run `(pdata).update_summary()` to sync changes back to .obs.")
        return self._summary

    @property
    def stats(self):
        return self._stats

    @property
    def metadata(self):
        if self.prot is not None and 'metadata' in self.prot.uns:
            return self.prot.uns['metadata']
        elif self.pep is not None and 'metadata' in self.pep.uns:
            return self.pep.uns['metadata']
        return {}

    @property
    def _cached_identifier_maps_protein(self):
        if not hasattr(self, "_gene_maps_protein"):
            self._gene_maps_protein = self._build_identifier_maps(self.prot)
        return self._gene_maps_protein

    @property
    def _cached_identifier_maps_peptide(self):
        if not hasattr(self, "_protein_maps_peptide"):
            self._protein_maps_peptide = self._build_identifier_maps(self.pep)
        return self._protein_maps_peptide

    # -----------------------------
    # Properties (SETTERS)
    @prot.setter
    def prot(self, value: ad.AnnData):
        self._prot = value

    @pep.setter
    def pep(self, value: ad.AnnData):
        self._pep = value

    @rs.setter
    def rs(self, value):
        self._set_RS(value)  # From EditingMixin

    @history.setter
    def history(self, value):
        self._history = value

    @summary.setter
    def summary(self, value: pd.DataFrame):
        self._summary = TrackedDataFrame(
            value,
            parent=self,
            mark_stale_fn=self._mark_summary_stale  # You likely have this defined in SummaryMixin or base
        )
        self._summary_is_stale = True
        suppress = getattr(self, "_suppress_summary_log", False)
        self.update_summary(recompute=True, sync_back=True, verbose=not suppress)

    @stats.setter
    def stats(self, value):
        self._stats = value