# `pAnnData` Overview

::: src.scviz.pAnnData.pAnnData.pAnnData
    options:
      show_root_heading: true
      heading_level: 2

## Import Functions

::: src.scviz.pAnnData.io.import_data
    options:
      show_root_heading: true
      heading_level: 2
::: src.scviz.pAnnData.io.suggest_obs_columns
    options:
      show_root_heading: true
      heading_level: 2
...

---

## Mixins

The `pAnnData` class inherits functionality from the following mixins:

- [BaseMixin](mixins/base.md)
- [ValidationMixin](mixins/validation.md)
- [SummaryMixin](mixins/summary.md)
- [MetricsMixin](mixins/metrics.md)
- [IdentifierMixin](mixins/identifier.md)
- [HistoryMixin](mixins/history.md)
- [EditingMixin](mixins/editing.md)
- [FilterMixin](mixins/filtering.md)
- [AnalysisMixin](mixins/analysis.md)
- [EnrichmentMixin](mixins/enrichment.md)

---

## Filtering Examples 

### Check API
::: src.scviz.pAnnData.filtering.FilterMixin
    options:
      filters:
        - "^filter_"