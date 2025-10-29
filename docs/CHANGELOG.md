# Full Changelog\n
# Changelog
All notable changes to this project are documented here.


<details open>
<summary><b>Unreleased</b></summary>


#### CI

- Upload changelog.md (2be787e…)


#### Other

- Fix github workflow bugs (1ef2b00…)

- (auto) Paper PDF Draft (6aee1de…)

- (auto) Paper PDF Draft (2894f44…)

- Fix workflow yml (cc2544f…)

- Update pyproject.toml (18af439…)

- Update pyproject.toml and workflow bug (19f4c7a…)

- Fix changelog yml tag fetch error (5b20d69…)

- Fix changelog yml workflow (cb956b1…)

- Fix changelog toml format (7c5dcc8…)

- Fix attempt for workflow (506109b…)

- Changelog sync to docs (84e991d…)

- Update full changelog [skip ci] (b8b5634…)



</details>


<details open>
<summary><b>0.4.0-alpha</b> – 2025-10-28</summary>


#### Changed

- Update joss-draft-pdf.yml (feac9d9…)

- Update python-package.yml (344e82a…)

- Update python-package.yml (5760023…)

- Update README.md (c709f2f…)


#### Other

- Add git-cliff config and changelog workflows (0159d05…)



</details>


<details open>
<summary><b>0.3.0-alpha</b> – 2025-10-08</summary>


#### Added

- Upload package to testpypi for testing (ea69dba…)

- Create python-package.yml (99e5e5f…)

- Add install scviz to github action workflow (995e408…)

- Create separate filter and plotting functions (7755f80…)

- Add get_classlist() (8495225…)

- Add cv function, tidy up plotting code (93a9992…)

- Add get_upset, fix pca errors (f717b08…)

- Add missing values handle (abb7578…)

- Add venn and upset, fix color (b0552e6…)

- Add columntransformer for imputing on classes (9ed2ba0…)

- Add validate function, simplify import functions with support for future formats (ef57f98…)

- Add rs related metrics, including updates to .summary and .prot.var (b66076d…)

- Added pairwise method, print utility functions, adata utility functions (9cb3348…)

- Implement get_abundance(), added pep_pairwise_median to de() for peptide pair-wise median fc calculation (5a79ec5…)

- Implement .summary as TrackedDataFrame, more verbose update_summary(), implement caching for gene/protein mapping, speed up computation by using sparse functions, fix filter prot/filter rs functions (94fef94…)

- Implement update_identifier_maps() and relevant tests, fixed some third party pytest warnings (d8c834a…)

- Implemented gsea with string (reminder - also add PPI) (4228119…)

- Added normalize by classes, fixed non_exact_cases in grouped imputation (8d3a87f…)

- Add plot_clustermap function (3297fe3…)

- Implement harmony, neighbor, fix normalize print order, update dependencies after harmony/leiden (5eed452…)

- Add stale catch for iloc/loc assignments on trackedDataFrame (26b2244…)

- Added export_layer function (7b02f65…)

- Add search_annotations function, clean up repr() (31e3d7f…)

- Added uniprot string search (faster), fixed taxon_id caching bug (7a549b8…)

- Add any match to filters, add annotation and filtering functions for fdr significance (56e8359…)

- Add fdr parsing to io functions, improve io documentation by implementing ioMixin (e412756…)

- Add tutorial pages to documentation (677ec01…)

- Create joss-draft-pdf.yml (f73ff5a…)

- Add directlfq normalization method (2e76f29…)

- Add precursor counts for directlfq (f9ed9c1…)

- Upload coverage action and readme (949448e…)

- Upload skeleton tutorials (3217b7e…)

- Added ambiguity checks for filter_prot_found and filter_prot_significant when user inputs classes, or mix of values and files (af16877…)

- Add error catching for if missing significance data entirely, catch old variable, and respective tests (8290169…)

- Add import tests and fix some bugs on import (23d6a24…)

- Add plotmixin to pAnnData class (a31e47a…)

- Upload tests for TrackedDataFrame (39aaa49…)

- Added tests for string network to enrichment test suite (4a08f4d…)

- Add uniprot schema actions check (daff24b…)

- Add fixtures for venn tests (7761af6…)


#### Changed

- Update get_cv() (ebb7001…)

- Update with more plotting functions (bd5646a…)

- Update data utility (8c06c34…)

- Update plot_heatmap_annotated with flexible search terms and return matches (d7e780d…)

- Update plot_heatmap_annotated() to accept list of search terms (dd94368…)

- Update package details (e60b04f…)

- Update functions with support for volcano plots and umap (35af9a6…)

- Update pAnnData class structure (1b4e97a…)

- Update .gitignore (1d6efdb…)

- Update with pAnnData object (name pending) (0484437…)

- Update pca and double-check if sc returns equivalent (a1a9326…)

- Update to 0.1.1 with some major pAnnData updates (2c9c9e1…)

- Update import functions to account for NaN obs columns (14bc74d…)

- Update get_uniprot_fields to work in batches (accepts >1024 proteins now) (2ca215e…)

- Update before merge (0b4f62b…)

- Update import function for new DIANN format (3e9a8d3…)

- Update for diann 2.0 (cd930bc…)

- Update versions (1c9cc64…)

- Update panndata filter functions to account for sample AND obs (9c9b842…)

- Overhaul of filter functions in class definition, legacy filter function now uses class filter functions (0bb8e93…)

- Updates to filter, fixing flags, modularize summary - obs push pull (04afff5…)

- Updates to plot_pca (supports ellipses, multi-class, abundance), plot_volcano (supports multiple fc modes for DE) (239a670…)

- Updates to panndata object, supporting missing gene search, de pairwise method (peptide coming soon), annotate and filter found, cleaned up import prints (632a3ae…)

- Update plot_abundace() to use get_abundance() function, reduce code clutter (c33165d…)

- Update test suite structure (96463b0…)

- Update import tests (3832b7d…)

- Update pca (dc68dd7…)

- Update to v 0.2.1 (5cf9034…)

- Update missing docs folder (cf9e117…)

- Update issue templates (62136cf…)

- Update documentation files (85c8726…)

- Updated get_color docstring to fit mkdocstring style (4c91090…)

- Update filter_prot docstring for mkdocstring style, clean up prints for de, impute, update_summary, bug fix for import_data (856ce2f…)

- Update notes and todo (5283e03…)

- Overhaul panndata.py into module (5e3f24c…)

- Update documentation w module (3839beb…)

- Update docstrings to mkdocstring format (f5178d0…)

- Update todo, add housekeeping plot function, update print statements (5803091…)

- Update plot_umap to handle more color types, refactor resolve_pca_colors into resolve_plot_colors helper function (3eb7aa0…)

- Update filter_sapmle docstring for more details on query mode (8554a87…)

- Update to latest version of string api (744e62f…)

- Update plot_abundance for ordered dictionary input, update plot_clustermap for ordered dictionary input (6037ac8…)

- Update todos (d5b9f10…)

- Update todo, add new utils page on documentation, remove blank space on init (bc5f98d…)

- Update mkdocs serve instructions (4646f47…)

- Update filter_sample_metadata() to filter_sample_condition(), fix print statements (e6afd40…)

- Update ci.yml (00c2d29…)

- Update python-package.yml (26457d1…)

- Update pyproject.toml (fe862a1…)

- Update README.md (3d26cc0…)

- Update README.md (5018ee2…)

- Update mkdocs.yml, update readme (0e4c5e2…)

- Update test suite for filtering (3210cfc…)

- Update conftest with new fixtures and helper functions (194344e…)

- Editing test suite and update to get_string_mappings after fixing uniprot API calls (61519a8…)

- Updates to pytest.ini for the new tests (c064839…)

- Update ci.yml (eb91917…)

- Update mkdocs.yml (7191610…)


#### Documentation

- Mkdocs for documentation, ci workflow for hosting documentation (bd0955b…)

- Documentation for plotting done (93a46d8…)

- Include changelog in docs (9d7dbc0…)


#### Fixed

- Fixing folder structure (9c00681…)

- Fix issue with adjacent imports #1 (9dccbed…)

- Fixing github actions workflow (049158c…)

- Fix lint errors (a45a33d…)

- Fix lint error (b09b9a2…)

- Fix lint error no.3 (2d6709e…)

- Fixed diann import (b882a6e…)

- Fix utility functions, add summary functionality to class (3965ef4…)

- Fix plot_rankquant() (b49969a…)

- Fix bug in filter function (499a2e2…)

- Fix volcano, integrate de into panndata and remove from utils (d487b4d…)

- Fix pca importance (31d9525…)

- Fix unintentional sort on import (da09ee3…)

- Fix a whole bunch of issues with filter, plotting, updated volcano (plot + mark) with genes, fixed de (c904a23…)

- Fix imputation between samples, overhaul simpleimputer (a6bb563…)

- Bug fixes on upset_contents (791f8ab…)

- Fix normalization, update plot, upgrade to v 0.1.6 (64375eb…)

- Bug fix for violinplot, matplotlib doesn't handle -inf (log10 0) values (422a741…)

- Fix rankquant bug when wrong cmap/color given (164d5c7…)

- Fixed format_class_filter bug with multi_class extract exact list (1ff4a52…)

- Fixed naive copy function, minor update to validate and initialization, implement clean_X to get ready for scanpy functions, implement obs suggestion utility function (7937cea…)

- Fix up UI printouts for a lot of user functions, updated todo list with known bugs (fee0ad9…)

- Fix corruption to config due to pc crash (c9aee23…)

- Fix bug in update_missing_genes from error out in uniprot mapping function (e3e6c71…)

- Fix impute nan bug (left as 0 instead of nan) (4ad09d3…)

- Fix bad html commenting (ce39a77…)

- Fix legacy mode to handle class_type is string, values is list of 2 strings (078c961…)

- Fix python raw string warning (9930c18…)

- Fix broken links due to enrichment mixin moving to pAnnData module, added enrichment shim file to maintain testing structure (03e4b69…)

- Fix print statements for filter_prot_significant (68d5e9f…)

- Bug fix on default neighbor params, update print statement on io (81b1f95…)

- Bugfix for prot/protein and pep/peptide matching (6310a93…)

- Fix fixtures over-riding actual scutils module (f9efae0…)

- Fix test for Leiden in py3.11 merges tiny graphs into one cluster (5e2c396…)


#### Other

- Initial commit (d3881e0…)

- Remove unnecessary files (3219b3c…)

- Push version 0.0.4 with tests and updated docs (9d36517…)

- Check github test workflow (12cacf4…)

- Install optional dependency openpyxl (1fbb5e4…)

- Clean up volcano functions (8002951…)

- Setup background gene sets for mouse and human (eb7183d…)

- Full fix of plot and mark rankquant() (8844c9f…)

- Tidy up plotting functions, move repeated code into helper functions (56a1292…)

- Trying to fix dependencies (a512607…)

- Finished plot_abundance function for protein/gene (list), including usable dataframe for user-specific plotting (ed35434…)

- Tidy up files and update package version (1724bb6…)

- Clean up abundance function with new helper resolve_accessions() using gene/protein mapping (b57b47c…)

- Forgot to stage pAnnData.py in previous commit oops (98ba646…)

- Quick fix for py3.8/old np errors (3e6dcef…)

- Pointer in plotting for enrichment svg, bunch of cleanup for old functions in utils (57321a3…)

- Final updates to enrichments (0a1528c…)

- Final fixes for enrichment, added support for URL to string (18f6e60…)

- Group-wise imputation done (59138bc…)

- Miscellaneous updates (93ff19c…)

- Quick fix for single cmap (d957169…)

- Quick fix for imputation check (keeping all empty features) (d844141…)

- Clean up filter tests, implement helper function for pep-prot syncing (e1f79fb…)

- Move extra functions to obs_util, update print statements (019e0e0…)

- Udpate test_import (0fc1079…)

- (forgot to commit) clean up prints for de, impute, update_summary, bug fix for import_data (df7bfc3…)

- Allow pca to accept palette dictionary (457e4c0…)

- Clean export functions, clean plot_volcano docstring (246af6a…)

- Include number of samples dropped in filter message (6cc8e22…)

- Clean print statements in filtering (e3e394d…)

- Clean up repr() (18e8181…)

- Initial commit for joss (537d6e4…)

- (auto) Paper PDF Draft (c782ba1…)

- Joss draft update (fcb268f…)

- (auto) Paper PDF Draft (3622341…)

- Utils docstring complete (0c19bfb…)

- Tidy up docstrings for documentation api (41ca53a…)

- Tidy up documentation layout for API and tutorials (aead14d…)

- Quick plot function for pdata objects (c1ca59e…)

- Import break due to kwargs pop (f210fd4…)

- Analysis test suite and bug fixes to analysis code (7440906…)

- Base test suite, moved plot functions to a new PlottingMixin, bug fixes on update_missing_genes (9dba386…)

- Big overhaul of test_utils and utils function, implement standardize_uniprot_columns and fallbacks for if uniprot changes column names (again) (08bb21b…)

- Barebones test for plotting suite, including bug fixes to plotting functions (a089c44…)

- Quick update to readme (a6a2d0c…)

- Small bug on cv error (1d81389…)

- Matplotlib-venn pushed update that broke plotting logic, added version-safe fallback (5cd6726…)

- (auto) Paper PDF Draft (4928e4b…)


#### Tests

- Unit tests for enrichment analysis (326f999…)

- Test joss action workflow (c99b2b7…)

- Test joss workflow (09224c1…)



</details>
<!-- generated by git-cliff -->
