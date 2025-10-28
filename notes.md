# about versions
1. For scikit-learn, need ver >1.2 so that we can handle sparse arrays in Imputer 

# rebuild docs
1. open cmd
2. conda deactivate
3. .\venv\Scripts\activate
4. mkdocs serve

# run coverage tests
1. open powershell
2. pytest --cov=src/scviz --cov-branch --cov-report=xml:htmlcov/coverage.xml --cov-report=html:htmlcov --cov-report=term-missing tests/

# for commits
Use format:
fix(parser): handle empty commit messages gracefully
feat(cli): add support for --dry-run flag
refactor(core)!: change internal API to use async/await
chore():
build():
ci():
style():
perf():
test():

# for tag (in git bash)
git tag "v$(grep -m1 version pyproject.toml | cut -d'"' -f2)" -m "Release v$(grep -m1 version pyproject.toml | cut -d'"' -f2)"
git push origin "v$(grep -m1 version pyproject.toml | cut -d'"' -f2)"

## delete
git tag -d v0.4.2-alpha v0.4.1-alpha
git push --delete origin v0.4.2-alpha v0.4.1-alpha