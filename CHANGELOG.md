All notable changes to this project are documented here.

<details open>
<summary><b>Unreleased</b></summary>


#### Added


##### (Filtering)


- Add exclude_file_list argument (2f592f1…)



##### (Plot_cv)


- Add palette flag (f92100b…)



#### Build System



- Bump scpviz to v0.5.0-alpha (1942e28…)



#### CI


##### (Changelog)


- Update .git-cliff.toml (85f2304…)



##### (Joss)


- Rename joss commit message to match conventional commit style (83c4935…)




- Update test CI to run only on changes to /src or /tests (06b3dc6…)



#### Chores


##### (Assets)


- Move assets out of src (917be4c…)



##### (Todo)


- Update todo list (388fdf3…)




- Update changelogs [skip ci] (8e09ce2…)

- Update changelogs [skip ci] (dfff921…)

- Update changelogs [skip ci] (6e3f746…)

- Update changelogs [skip ci] (8a99267…)



#### Documentation


##### (Assets)


- Upload logo and test assets (e1ee49b…)



##### (Package rename)


- Rename all links in docs (cfc8344…)

- Update mkdocs.yml (838fa33…)



##### (Quickstart)


- Update tutorial files (422ae4f…)

- Upload diann_report.parquet, using git lfs (32501ee…)



##### (Readme)


- Update readme (b8637fb…)

- Update broken links and logo (8d7b64a…)



##### (Setup)


- Update mkdocs.yml, add dev and js for navigation (6427731…)



##### (Tutorial)


- Update quickstart, some tutorials in works (91e10bf…)

- Update tutorial home page (f1a0cb9…)




- Update readme to proper image (7b82bea…)

- Remove conda installation from readme, update gitignore (0e5e5ac…)

- Update joss paper.md (11151fc…)



#### Fixed


##### (Io)


- Implement handler for diann files when using suggest_obs_columns (93cfac5…)



#### Other


##### (Paper)


- Update Paper PDF Draft (2fe3401…)



#### Style


##### (Cv)


- Default to false verbose on cv resolve_class_filter (ee346ca…)



##### (Filtering)


- Fix print statements to be more verbose (2fdeae3…)

- Updated print statements to include exclude_file_list (3efd2c6…)

- Add print for cleanup with no empty prots (b71aab6…)

- Fix typo in print statement of annotate_significant_prot (9445f0a…)



##### (Readme)


- Update version on docs badge (ad74fa6…)




</details>

<details open>
<summary><b>0.4.1-alpha</b> – 2025-11-04</summary>


#### Added


##### (Base)


- Add compare_current_to_raw and get_X_raw_aligned functionality (043251b…)



##### (Filtering)


- Add valid_genes, unique_profiles to filter_prot and cleanup (nans) to filter_sample (e251c8a…)

- Add handling of duplicate gene name in filter_prot (01d750e…)



##### (Import)


- Add cleanup after import (b290ec8…)

- Add support for pd3.2 import (592a814…)



##### (Plot)


- Add plot_abundnace wrapper from pdata (8e2a01c…)



#### Build System



- Update pyproject.toml (18af439…)

- Update pyproject.toml and workflow bug (19f4c7a…)

- Fix changelog yml tag fetch error (5b20d69…)



#### CI


##### (Changelog)


- Upload changelog.md (2be787e…)



##### (Pytest.ini)


- Add test "slow" marker (8280133…)



#### Chores


##### (Changelog)


- Changelog sync to docs (84e991d…)

- Final updates (6b94c5b…)

- Edit changelog workflow (0ec6ab5…)



##### (Docs)


- Update deploy workflow to use committed changelog (9fa1705…)




- Fix github workflow bugs (1ef2b00…)

- Fix workflow yml (cc2544f…)

- Fix changelog yml workflow (cb956b1…)

- Fix changelog toml format (7c5dcc8…)

- Fix attempt for workflow (506109b…)

- Update full changelog [skip ci] (b8b5634…)

- Update changelogs [skip ci] (c3a8704…)

- Update changelogs [skip ci] (2356fff…)

- Update changelogs [skip ci] (9910211…)

- Update changelogs [skip ci] (b5d04c7…)

- Update changelogs [skip ci] (7e5d8b6…)

- Update changelogs [skip ci] (a688e34…)

- Update changelogs [skip ci] (265982c…)

- Update changelogs [skip ci] (ca558ad…)



##### (Workflow)


- Build and deploy runs after changelog is finished (97491c3…)



#### Documentation


##### (Coc)


- Add contributor covenant code of conduct (72fd27e…)



##### (Contributing)


- Update contributing.md file (4002003…)




- Update markdown files (d72e6fa…)



#### Fixed


##### (Base)


- Anndata automatically aligns X_raw, removed function and tests (ce82a52…)



##### (Export)


- Handle export of .X (a3028bd…)



##### (Filtering)


- Fix bug on import with non-matching obs/summary after nan protein cleanup (f61c0b5…)



##### (Import)


- Renaming scheme for pd prot var (9bcbf44…)



##### (Plotting)


- Fix bug for volcano_df handling of "not comparable" (1d3d87b…)



#### Other


##### (Identifier)


- Fix indentation on update_identifiers tip (90b7bec…)



##### (Package name)


- Rename scviz to scpviz (9bc6347…)



#### Style


##### (Changelog)


- Edit markdown formatting for docs changelog (ff92df7…)

- Update parsers to match conventional commit format (a870d32…)



##### (De)


- Add comment on metaboanalyst median normalization (42c0260…)



##### (Summary)


- Edit table of usage scenarios for update_summary to be clearer (ffd2a37…)



#### Tests


##### (Filtering,import)


- Add tests for duplicate gene handling, import pd32 (4cf1b86…)



##### (Test files)


- Add test for pd3.2 import with prot and pep, upload pd3.2 mock files (3c96c45…)




</details>

<details open>
<summary><b>0.4.0-alpha</b> – 2025-10-28</summary>


#### Changed


#### Chores



- Add git-cliff config and changelog workflows (0159d05…)




</details>

<details open>
<summary><b>0.3.0-alpha</b> – 2025-10-08</summary>


#### Added


#### Changed


#### Documentation



- Include changelog in docs (9d7dbc0…)



#### Fixed


#### Other


#### Tests



</details>
<!-- generated by git-cliff -->
