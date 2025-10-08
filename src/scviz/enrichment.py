# scviz/enrichment.py

"""
Shim module to expose enrichment functions at top-level `scviz.enrichment`.
Delegates to `scviz.pAnnData.enrichment`.
"""

from scviz.pAnnData.enrichment import (
    enrichment_functional,
    enrichment_ppi,
    _resolve_de_key,
    _pretty_vs_key,
    EnrichmentMixin,
)

import requests  # so patch("scviz.enrichment.requests.post") still works
