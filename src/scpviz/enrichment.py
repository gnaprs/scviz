# scpviz/enrichment.py

"""
Shim module to expose enrichment functions at top-level `scpviz.enrichment`.
Delegates to `scpviz.pAnnData.enrichment`.
"""

from scpviz.pAnnData.enrichment import (
    enrichment_functional,
    enrichment_ppi,
    _resolve_de_key,
    _pretty_vs_key,
    EnrichmentMixin,
)

import requests  # so patch("scpviz.enrichment.requests.post") still works
