"""Pretrained checkpoint resolution and download.

Checkpoints are not part of the PyPI package. They live in the repo as Git
LFS files and as GitHub release assets under the ``weights-v1`` tag. Lookup
order: in-tree first, then the cache dir. Fetch missing checkpoints with
``python -m jax_md._nn.weights <model>``. The digests below equal the repo's
LFS pointer digests: ``sha256sum jax_md/_nn/*/*.eqx``.
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_RELEASE_TAG = 'weights-v1'
WEIGHTS_BASE_URL = (
  f'https://github.com/jax-md/jax-md/releases/download/{WEIGHTS_RELEASE_TAG}'
)

# filename -> sha256, size in bytes
WEIGHTS = {
  'orb-v3-conservative-omol.eqx': (
    '5879019afdc06b42bc6b59abe161cc4e900233ee363b6748f89007f1a622625f',
    97804680,
  ),
  'ani2x_ensemble.eqx': (
    '7885718884108ae9e4bfccfeb46ad76d084a8437edc4399d9687f6e56cfcc262',
    75868388,
  ),
  'ani2x_model0.eqx': (
    'b61bcbd57c01125c9873e572f25ac09d5cddb2e7af3a865ba76dd3e88b16e24d',
    9485344,
  ),
  'aimnet2.eqx': (
    '27666e8dfd733b03df6fc69c4ca7339f4c718a8b753520025585735ba4533cd1',
    8835119,
  ),
  'aceff_v2.0.eqx': (
    '9e83ac0791738980250d3f5454a0a0fa069705f2948a00c9a0c485beae077797',
    3988508,
  ),
  'aceff_v1.1.eqx': (
    '069d0cff78e61a723c755aae54f61e981dfad33a174890c27d77a18357726286',
    2148642,
  ),
  'so3lr.eqx': (
    '57ae5e8e35a2690e219d8ab08fea95a6c7c133ea48643ee004ba8fe040fb4d3c',
    2139519,
  ),
}

# family -> checkpoint filenames
MODELS = {
  'orb': ('orb-v3-conservative-omol.eqx',),
  'ani': ('ani2x_ensemble.eqx', 'ani2x_model0.eqx'),
  'aimnet2': ('aimnet2.eqx',),
  'aceff': ('aceff_v1.1.eqx', 'aceff_v2.0.eqx'),
  'so3lr': ('so3lr.eqx',),
}


def cache_dir() -> Path:
  override = os.environ.get('JAX_MD_CACHE')
  if override:
    return Path(override)
  xdg = os.environ.get('XDG_CACHE_HOME')
  base = Path(xdg) if xdg else Path.home() / '.cache'
  return base / 'jax_md'


def is_lfs_pointer(path: Path) -> bool:
  # A checkout without LFS content leaves a small text stub behind.
  try:
    if path.stat().st_size > 512:
      return False
    with path.open('rb') as handle:
      return handle.read(24).startswith(b'version https://git-lfs')
  except OSError:
    return False


def family_of(filename: str) -> str | None:
  for family, filenames in MODELS.items():
    if filename in filenames:
      return family
  return None


def resolve_checkpoint(
  path: os.PathLike | str, *, allow_cache: bool = True
) -> Path:
  """Return an on-disk path for the checkpoint expected at ``path``.

  Falls back to the cache directory, then raises ``FileNotFoundError``
  with fetch instructions. Set ``allow_cache`` to False for an explicit
  user-supplied path so a missing or LFS-pointer override is not silently
  replaced by an unrelated cache file.
  """
  path = Path(path)
  if path.is_file() and not is_lfs_pointer(path):
    return path

  cached = cache_dir() / path.name
  expected = WEIGHTS.get(path.name)
  if (
    allow_cache
    and expected is not None
    and cached.is_file()
    and not is_lfs_pointer(cached)
    and hashlib.sha256(cached.read_bytes()).hexdigest() == expected[0]
  ):
    return cached

  family = family_of(path.name) or path.name
  raise FileNotFoundError(
    f'Checkpoint {path.name!r} was not found. Pretrained weights are not '
    f'bundled with the jax-md PyPI package. Fetch it with:\n'
    f'  python -m jax_md._nn.weights {family}\n'
    f'or manually:\n'
    f'  curl -L --create-dirs -o {cached} {WEIGHTS_BASE_URL}/{path.name}\n'
    f'The cache location is set by the JAX_MD_CACHE environment variable. '
    f'A git clone with LFS content also works.'
  )


def fetch(url: str, dest: Path, sha256: str, progress: bool = True) -> Path:
  dest.parent.mkdir(parents=True, exist_ok=True)
  digest = hashlib.sha256()
  with urllib.request.urlopen(url, timeout=30) as response:
    total = int(response.headers.get('Content-Length') or 0)
    with tempfile.NamedTemporaryFile(
      dir=dest.parent, prefix=dest.name + '.', delete=False
    ) as tmp:
      try:
        done = 0
        while True:
          chunk = response.read(1 << 20)
          if not chunk:
            break
          digest.update(chunk)
          tmp.write(chunk)
          done += len(chunk)
          if progress and total:
            print(
              f'\r  {dest.name}: {done / 1048576:.0f}/{total / 1048576:.0f} MiB',
              end='',
              file=sys.stderr,
            )
      except BaseException:
        Path(tmp.name).unlink(missing_ok=True)
        raise
  if progress and total:
    print(file=sys.stderr)
  if digest.hexdigest() != sha256:
    Path(tmp.name).unlink(missing_ok=True)
    raise RuntimeError(
      f'SHA-256 mismatch for {url}: got {digest.hexdigest()}, '
      f'expected {sha256}. Refusing to keep the corrupt file.'
    )
  os.replace(tmp.name, dest)
  return dest


def download(filename: str, force: bool = False, progress: bool = True) -> Path:
  """Download one checkpoint by filename into the cache; returns its path."""
  if filename not in WEIGHTS:
    known = ', '.join(sorted(WEIGHTS))
    raise ValueError(f'Unknown checkpoint {filename!r}. Known: {known}')
  sha256, _ = WEIGHTS[filename]
  dest = cache_dir() / filename
  if (
    dest.is_file()
    and not force
    and not is_lfs_pointer(dest)
    and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256
  ):
    return dest
  return fetch(f'{WEIGHTS_BASE_URL}/{filename}', dest, sha256, progress)


def download_models(names, force: bool = False, progress: bool = True):
  """Download checkpoints for model families or filenames; returns paths."""
  filenames = []
  for name in names:
    if name == 'all':
      filenames.extend(fn for fns in MODELS.values() for fn in fns)
    elif name in MODELS:
      filenames.extend(MODELS[name])
    elif name in WEIGHTS:
      filenames.append(name)
    else:
      known = ', '.join(['all', *MODELS])
      raise ValueError(f'Unknown model {name!r}. Known: {known}')
  return [download(fn, force=force, progress=progress) for fn in filenames]


def main(argv=None) -> int:
  import argparse

  parser = argparse.ArgumentParser(
    prog='python -m jax_md._nn.weights',
    description='Fetch pretrained checkpoints into the cache directory '
    'where model loaders find them. Downloads are verified against '
    'pinned SHA-256 digests.',
  )
  parser.add_argument(
    'models',
    nargs='*',
    metavar='MODEL',
    help=f'model families to fetch: all, {", ".join(MODELS)} '
    f'or an exact checkpoint filename',
  )
  parser.add_argument(
    '--list', action='store_true', help='list checkpoints and their status'
  )
  parser.add_argument(
    '--force', action='store_true', help='re-download even if cached'
  )
  args = parser.parse_args(argv)

  if args.list or not args.models:
    cache = cache_dir()
    print(f'cache directory: {cache}')
    for family, filenames in MODELS.items():
      for filename in filenames:
        _, size = WEIGHTS[filename]
        status = 'cached' if (cache / filename).is_file() else 'not downloaded'
        print(
          f'  {family:8s} {filename:36s} {size / 1048576:6.1f} MiB  {status}'
        )
    if not args.models:
      parser.print_usage()
      return 0

  try:
    paths = download_models(args.models, force=args.force)
  except (ValueError, RuntimeError, OSError) as error:
    print(f'error: {error}', file=sys.stderr)
    return 1
  for path in paths:
    print(path)
  return 0


if __name__ == '__main__':
  sys.exit(main())
