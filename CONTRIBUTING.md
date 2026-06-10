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
uv venv
uv pip install -e .
uv pip install pytest ase
```

## Code style

Formatting and linting are enforced with [ruff](https://docs.astral.sh/ruff/)
(configuration lives in `pyproject.toml`). Before sending a change, run:

```sh
uv run ruff check .
uv run ruff format .
```

Or install the [pre-commit](https://pre-commit.com/) hooks once and let them
run on every commit:

```sh
pre-commit install
```

## Running tests

Tests live in `tests/` as one file per module and run in double precision:

```sh
JAX_ENABLE_X64=1 uv run pytest tests/<suite>_test.py
```

Run the suites affected by your change (for example
`tests/partition_test.py` for changes to `jax_md/partition.py`). Continuous
integration runs the full matrix on Python 3.10–3.13, along with the ruff
checks, on every pull request.

## Code reviews

All submissions, including submissions by project members, require review.
Open a pull request against `main` from a fork; see
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for
more information on using pull requests. Please keep pull requests focused
and include tests for new functionality.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
