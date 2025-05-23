# ✅ CI/CD Integration with GitHub Actions

## 🎯 Objective
To automatically run all QA tests on every push or pull request using GitHub Actions — ensuring that code changes never break the EEG pipeline.

---

## ⚙️ Setup Summary

The workflow file is located at:

```
.github/workflows/ci.yml
```

This workflow:
- Triggers on pushes and pull requests to the `main` branch
- Installs all dependencies listed in `requirements.txt`
- Runs the `pytest` suite for full test coverage
- Fails visibly if any test fails — promoting early bug detection

---

## ❌ Issues Encountered and Fixes

### 1. **Error: No event triggers defined in `on`**
- **Cause:** The workflow file was missing the `on:` block.
- **Fix:** Added:
  ```yaml
  on:
    push:
      branches: [ main ]
    pull_request:
      branches: [ main ]
  ```

### 2. **Error: No jobs defined in `jobs`**
- **Cause:** GitHub Actions expected at least one job to run, but none were defined.
- **Fix:** Added a `jobs:` section with a `test` job to run Python tests:
  ```yaml
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v5
          with:
            python-version: '3.10'
        - run: |
            pip install -r requirements.txt
            pip install pytest
        - run: pytest tests/
  ```

---

## ✅ Outcome

- CI is live and functional
- Full test suite runs on every push and PR
- Confidence in code quality is automated
