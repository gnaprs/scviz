# about versions
1. Declared constraint is `scikit-learn>=1.0` in `pyproject.toml`. Sparse `Imputer` needs 1.2+; tighten the pin if that still matters. 

# rebuild docs
`pip install -e ".[docs]"`    # after changing pyproject / first time
1. open command prompt
2. conda activate scpviz-docs
3. mkdocs serve

# run coverage tests
1. open powershell
2. pytest --cov=src/scpviz --cov-branch --cov-report=xml:htmlcov/coverage.xml --cov-report=html:htmlcov --cov-report=term-missing tests/

# for commits
Use format:
## core
* feat: introduce a new feature for the user or API [`feat(cli): add support for --dry-run flag`]
* fix: Resolve a bug in existing code [`fix(parser): handle empty commit messages gracefully`]
* refactor: Modify code structure without changing behavior [`refactor(core)!: change internal API to use async/await`]
* perf: Improve performance (speed, efficiency, memory) [`perf(loop): reduce time complexity from O(n²) to O(n)`]

## support
* chore: Routine maintenance, tooling, dependencies [`chore(deps): bump numpy to 1.26.4`]
* build: Changes to build system, packaging, or dependencies [`build(setup): add scpviz to pyproject.toml dependencies`]
* ci: Continuous integration / deployment configurations [`ci(github): add test coverage badge update to workflow`]
* style: Code style, formatting, or linting only [`style(core): apply black autoformatter`]
* test: Add, update, or fix tests [`test(filtering): add test for duplicate gene handling`]

# version release
## Things to update
1. pyproject.toml -> version
2. readme.md -> documentation version badge
3. Push tag -> automatically updates changelog and deploys docs
4. Draft and publish GitHub Release -> automatically uploads to PyPI (`publish.yml`; do not twine locally)

## Add tag (open repo in git bash)
git fetch origin
VERSION=$(grep -m1 version pyproject.toml | cut -d'"' -f2)
git tag -a "v$VERSION" origin/main -m "Release v$VERSION"
CHECK: name of tag with `git tag -n` (q to quit)
git push origin "v$VERSION"

## PyPI release

Go to GitHub -> Releases -> "Draft a new release" -> select tag -> publish.
This triggers the publish.yml workflow which builds and uploads to PyPI automatically.

### If we need to tag delete
git tag -d v0.4.2-alpha v0.4.1-alpha
git push --delete origin v0.4.2-alpha v0.4.1-alpha