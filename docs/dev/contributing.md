# `scpviz` Contribution Guidelines

Thank you for your interest in contributing to **scpviz**, an open-source Python package for visualizing and analyzing single-cell and bulk proteomics data.  
We welcome contributions from the community to help improve, expand, and document the functionality of scpviz.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant](CODE_OF_CONDUCT.md).  
Please be respectful and considerate in your interactions with others.

---

## How to Contribute

To get an overview of the project, read the [README](https://github.com/gnaprs/scpviz/blob/main/README.md) file.

There are several ways you can contribute to scpviz, including but not limited to:

* asking and answering questions in [Discussions](https://github.com/gnaprs/scpviz/discussions),
* reporting bugs and requesting features by submitting [new issues](https://github.com/gnaprs/scpviz/issues/new),
* adding new features and fixing bugs by creating pull requests (PRs),
* improving and maintaining consistency in the documentation (including docstrings and tutorials), and
* providing reproducible examples and workflows in Jupyter notebooks.

---

## Getting Started

### Issues

#### Open a New Issue

Before reporting a bug or requesting a feature, search to see if a related issue already exists.  
If not, you can [submit a new issue](https://github.com/gnaprs/scpviz/issues/new) — make sure to include:

- a clear and descriptive title,
- relevant environment or dataset information (if applicable), and
- a minimal, reproducible example (if possible).

#### Solve an Issue

Browse through the existing [issues](https://github.com/gnaprs/scpviz/issues) to find one that interests you.  
You can filter by labels (e.g., *feature*, *bug*, *enhancement*).  
If you find an issue you’d like to work on, comment to let maintainers know and open a PR when ready.

---

### Make Changes

To contribute to scpviz, use the **fork and pull request** workflow described below.

1. [Fork the repository.](https://github.com/gnaprs/scpviz/fork)
2. Clone your fork locally and navigate to it:

       git clone https://github.com/gnaprs/scpviz.git
       cd scpviz

3. Create a new branch for your feature or fix:

       git checkout -b <branch-name>

4. Install scpviz and its development dependencies:

       pip install -e .[dev]

   You may also want to create and activate a virtual environment before installing dependencies:

       python3 -m venv .venv
       source .venv/bin/activate

5. Run tests to confirm everything works before editing:

       pytest tests/ -v

---

### Development Guidelines

Please follow these best practices when contributing code:

* Follow the [PEP 8](https://peps.python.org/pep-0008/) style guide.
* Write clear, consistent docstrings in the [Google style](https://google.github.io/styleguide/pyguide.html#381-docstrings).
* Add [pytest](https://docs.pytest.org/) unit tests for new functions and features.
* Use meaningful variable names and comments where appropriate.
* Keep imports organized and minimal.
* When modifying documentation, ensure that `mkdocs build --strict` completes successfully.

---

### Commit Your Update

When your changes are ready:

1. Ensure that all unit tests pass:

       pytest tests/ -v

2. Stage and commit your changes:

       git add .
       git commit -m "<short-description-of-your-changes>"

3. Push your branch to your fork:

       git push origin <branch-name>

---

### Pull Request

To contribute your changes to the main scpviz repository, [create a pull request](https://github.com/gnaprs/scpviz/compare).  
The project maintainers will review your PR and provide feedback.  
If your changes align with the project goals and pass all tests, they will be merged into the main branch.

---

## Documentation Contributions

The documentation for scpviz is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

To build and preview locally:

```bash
mkdocs serve
```

Your changes will automatically rebuild the site in your browser.

When merged to `main`, the documentation is automatically deployed to  
[https://gnaprs.github.io/scpviz/](https://gnaprs.github.io/scpviz/)  
via GitHub Actions (`ci.yml`).

---

## Release Process (Maintainers)

When ready to publish a new version:

1. Update the version number in `pyproject.toml`.
2. Commit, tag, and push:
   ```bash
   git commit -am "Bump version to v0.X.Y"
   git tag -a v0.X.Y -m "Release v0.X.Y"
   git push origin main --tags
   ```
3. Build and upload to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```
4. Verify:
   - PyPI: https://pypi.org/project/scpviz/
   - Docs: https://gnaprs.github.io/scpviz/
   - Coverage: https://codecov.io/gh/gnaprs/scpviz

---

## Additional Resources

- [PEP 621](https://peps.python.org/pep-0621/): `pyproject.toml` metadata format  
- [MkDocs Material Documentation](https://squidfunk.github.io/mkdocs-material/)  
- [pytest Documentation](https://docs.pytest.org/)  
- [Codecov Integration Guide](https://docs.codecov.com/docs)

---

Thank you for helping make **scpviz** a reliable, open, and community-driven platform for single-cell and spatial proteomics research.
