# Contributing

Thanks for taking the time to contribute.

## Development setup

MLX, `mlx-lm` and `coremltools` need Apple Silicon. On other machines the MLX tests are skipped or fail.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v --cov=src --cov-report=term-missing
# lint: run the exact flake8 command from the Lint step in .github/workflows/ci.yml
```

CI (`.github/workflows/ci.yml`) runs on Linux without MLX, so its test step cannot gate a PR. Run the tests
on Apple Silicon before opening one, and say in the PR that you did.

## Ground rules

- **One logical change per PR.** Keep diffs reviewable.
- **Tests are not optional.** New behaviour needs a test; bug fixes need a test that fails before the fix.
- **Measured vs target.** The README separates design targets from measured results. A number moves into
  "measured" only with the command and hardware that produced it.
- **No secrets in code.** Tokens and keys come from the environment.

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Reporting bugs / proposing features

Use the issue templates. For anything security-related, follow [SECURITY.md](SECURITY.md) instead of opening
a public issue.
