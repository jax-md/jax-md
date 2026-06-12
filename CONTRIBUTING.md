# How to Contribute

We'd love to accept your patches and contributions to JAX MD. Development
happens in the open on GitHub at
[jax-md/jax-md](https://github.com/jax-md/jax-md), and we welcome bug
reports, feature requests, and pull requests. Contributions are licensed,
like the rest of the project, under the [Apache License 2.0](LICENSE).

## Reporting issues and requesting features

Opening [issues](https://github.com/jax-md/jax-md/issues) with bug reports
or feature requests helps us guide development, even if you don't plan to
implement the change yourself.

## Development setup

JAX MD uses [uv](https://docs.astral.sh/uv/) for environment management:

```sh
git clone https://github.com/<your-username>/jax-md.git
cd jax-md
uv sync --no-default-groups --group lint --group testing
```

This installs the package with the linting (`ruff`, `ty`) and testing
(`pytest`, `ase`, `netCDF4`) tools at locked versions. A plain `uv sync`
additionally installs the heavier `dev` group (PyTorch reference stack).

## Code style

Formatting and linting are enforced with [ruff](https://docs.astral.sh/ruff/),
and type checking with [ty](https://docs.astral.sh/ty/) (configuration for
both lives in `pyproject.toml`). Before sending a change, run:

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format .
uv run --no-sync ty check --error-on-warning
```

(`--no-sync` keeps `uv run` from re-syncing the environment to the default
`dev` group, which pulls in the heavy PyTorch reference stack.)

`ty check --error-on-warning` must exit cleanly: CI treats warning-level
diagnostics as failures, including stale `ty: ignore` comments. The one rule
still being ratcheted up is listed in `[tool.ty.rules]`; avoid adding new
suppressions.

Or install the [pre-commit](https://pre-commit.com/) hooks once and let them
run on every commit:

```sh
pre-commit install
```

## Running tests

Tests live in `tests/` as one file per module and run in double precision:

```sh
JAX_ENABLE_X64=1 uv run --no-sync pytest tests/<suite>_test.py
```

Run the suites affected by your change (for example
`tests/partition_test.py` for changes to `jax_md/partition.py`). Continuous
integration runs the full matrix on Python 3.10–3.13, along with the ruff
and ty checks, on every pull request.

## Code reviews

All submissions, including submissions by project members, require review.
Open a pull request against `main` from a fork; see
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for
more information on using pull requests. Please keep pull requests focused
and include tests for new functionality.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
