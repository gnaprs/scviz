# Hidden Functions

Hidden functions for all MixIns.

!!! warning "Advanced / Internal"
    The functions in this section are internal utilities. They may change 
    without notice and are not guaranteed to remain stable across releases. 
    Use only if you understand the internal architecture of `pAnnData`.

---

::: src.scpviz.pAnnData.analysis.AnalysisMixin
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _normalize_helper
        - _normalize_helper_directlfq

::: src.scpviz.pAnnData.base
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _has_data

::: src.scpviz.pAnnData.editing
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _set_RS

::: src.scpviz.pAnnData.enrichment
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _pretty_vs_key
        - _resolve_de_key

::: src.scpviz.pAnnData.filtering
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _detect_ambiguous_input
        - _filter_sync_peptides_to_proteins
        - _filter_sample_condition
        - _filter_sample_values
        - _filter_sample_query
        - _cleanup_proteins_after_sample_filter
        - _apply_rs_filter
        - _format_filter_query
        - _annotate_found_samples
        - _annotate_significant_samples

::: src.scpviz.pAnnData.history
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _append_history

::: src.scpviz.pAnnData.identifier
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _build_identifier_maps

::: src.scpviz.pAnnData.io
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _import_proteomeDiscoverer
        - _import_diann
        - _create_pAnnData_from_parts

::: src.scpviz.pAnnData.metrics
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _update_metrics
        - _update_summary_metrics

::: src.scpviz.pAnnData.summary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _update_summary
        - _merge_obs
        - _push_summary_to_obs
        - _mark_summary_stale

::: src.scpviz.pAnnData.validation
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - _check_data
        - _check_rankcol