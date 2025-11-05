from scpviz.pAnnData.plot import PlotMixin
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
from .io import IOMixin

import pandas as pd
import anndata as ad
from scpviz.TrackedDataFrame import TrackedDataFrame 

class pAnnData(BaseMixin, ValidationMixin, SummaryMixin, MetricsMixin,
               IdentifierMixin, HistoryMixin, EditingMixin, FilterMixin,
               AnalysisMixin, EnrichmentMixin, IOMixin, PlotMixin):
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
    - **PlotMixin**: Used to plot and visualize attributes of the pAnnData object, frequently for QC.

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

    def __repr__(self):
        def format_summary(summary):
            if summary is None:
                return "Summary:\n  None"

            default_cols = {"protein_count", "peptide_count", "protein_quant", "peptide_quant"}
            grouping_cols = [col for col in summary.columns
                            if col not in default_cols and summary[col].nunique() < len(summary)]

            avg_prot = summary["protein_count"].mean() if "protein_count" in summary else None
            avg_quant = summary["protein_quant"].mean() if "protein_quant" in summary else None
            low_quant = (summary["protein_quant"] < 0.5).sum() if "protein_quant" in summary else None

            lines = []
            if grouping_cols:
                lines.append(f"Groups: {', '.join(grouping_cols)}")
            if avg_prot is not None:
                lines.append(f"Avg proteins/sample: {avg_prot:.1f}")
            if avg_quant is not None:
                lines.append(f"Avg protein quant: {avg_quant:.2f}")
            if low_quant is not None:
                lines.append(f"Samples < 50% quant: {low_quant}")

            return "Summary:\n" + "\n".join(f"  {line}" for line in lines) if lines else "Summary:\n  —"

        def format_ann(adata, label, shared_obs=False):
            if adata is None:
                return f"{label}: None"

            shape_str = f"{adata.shape[0]} files × {adata.shape[1]} {label.lower()}s"
            obs_cols = ', '.join(adata.obs.columns[:5]) + ('...' if len(adata.obs.columns) > 5 else '')
            var_cols = ', '.join(adata.var.columns[:5]) + ('...' if len(adata.var.columns) > 5 else '')
            obsm_keys = ', '.join(adata.obsm.keys())
            layers_keys = ', '.join(adata.layers.keys())

            obs_line = "  obs:    (same as protein)" if shared_obs and label == "Peptide" else f"  obs:    {obs_cols or '—'}"

            return (f"{label} (shape: {shape_str})\n"
                    f"{obs_line}\n"
                    f"  var:    {var_cols or '—'}\n"
                    f"  obsm:   {obsm_keys or '—'}\n"
                    f"  layers: {layers_keys or '—'}")

        def format_rs_summary(rs):
            if rs is None:
                return "RS:\n  None"

            peptides_per_protein = rs.getnnz(axis=1)
            unique_mask = rs.getnnz(axis=0) == 1
            unique_counts = rs[:, unique_mask].getnnz(axis=1)

            mean_pep = peptides_per_protein.mean()
            mean_uniq = unique_counts.mean()
            pct_highconf = (unique_counts >= 2).mean() * 100

            return (
                f"RS (shape: {rs.shape[0]} proteins × {rs.shape[1]} peptides)\n"
                f"  Avg peptides/protein: {mean_pep:.2f}\n"
                f"  Avg unique peptides : {mean_uniq:.2f}\n"
                f"  Proteins with ≥2 unique peptides: {pct_highconf:.1f}%"
            )

        def format_enrichments(stats):
            functional = stats.get("functional", {})
            ppi_keys = stats.get("ppi", {})
            de_keys = {k for k in stats if "vs" in k and not k.endswith(("_up", "_down"))}

            enriched_de = {
                meta.get("input_key", k)
                for k, meta in functional.items()
                if "vs" in k and meta.get("input_key", None) in de_keys
            }

            n_de = len(de_keys)
            n_func = len(functional)
            n_ppi = len(ppi_keys)
            n_unenriched = n_de - len(enriched_de)

            lines = []
            if n_de:
                lines.append(f"DE comparisons: {n_de}")
                if n_func:
                    examples = sorted(k for k in functional if "vs" in k)[:3]
                    lines.append(f"Functional enrichment: {n_func} result(s) (e.g. {', '.join(examples)})")
                if n_ppi:
                    lines.append(f"PPI enrichment: {n_ppi} result(s)")
                if n_unenriched > 0:
                    lines.append(f"Pending enrichment: {n_unenriched}")
            elif n_func or n_ppi:
                if n_func:
                    lines.append(f"Functional enrichment: {n_func} result(s)")
                if n_ppi:
                    lines.append(f"PPI enrichment: {n_ppi} result(s)")

            if lines:
                lines.append("↪ Use `pdata.list_enrichments()` for details")

            return "STRING Enrichment:\n" + "\n".join(f"  {line}" for line in lines) if lines else "STRING Enrichment:\n  —"

        shared_obs = self.prot is not None and self.pep is not None and self.prot.obs.equals(self.pep.obs)

        lines = ["pAnnData object"]
        lines.append("")  # Spacer

        # Summary
        lines.append(format_summary(self._summary))
        lines.append("")  # Spacer

        # Protein and Peptide
        lines.append(format_ann(self.prot, "Protein", shared_obs=shared_obs))
        lines.append("")  # Spacer
        lines.append(format_ann(self.pep, "Peptide", shared_obs=shared_obs))
        lines.append("")  # Spacer

        # RS Matrix
        lines.append(format_rs_summary(self._rs))
        lines.append("")  # Spacer

        # Enrichment Summary
        enrichment_info = format_enrichments(self.stats)
        lines.append(enrichment_info)

        return "\n".join(lines)
    
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