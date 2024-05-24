## To-Do List
1. Add more visualization features (correlation matrix)
2. Extend support for GSEA and GO analysis
3. Add more comprehensive documentation.
4. Implement unit tests for better code reliability.

# TODO For marion
Add sharedPeptides function on get_CV()
Check out scprep repo for possible utility functions
Implement DE utility + volcano plotting function
Add venn diagram and upsetplot function
Add Type hints to all functions, e.g. 
```python
from typing import Any, Dict, Optional

def get_abundance(
    data: Any,
    cases: Any,
    prot_list: Optional[Any] = None,
    list_type: str = 'accession',
    abun_type: str = 'average'
) -> Optional[Dict[Any, Any]]:
    # function body
```