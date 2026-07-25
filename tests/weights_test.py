"""Tests for jax_md._nn.weights checkpoint resolution. No network access."""

import hashlib
from pathlib import Path

import pytest

from jax_md._nn import weights

FAKE_BYTES = b'not a real checkpoint, but real enough for resolution\n'

LFS_POINTER = (
  b'version https://git-lfs.github.com/spec/v1\n'
  b'oid sha256:' + b'0' * 64 + b'\n'
  b'size 2129435\n'
)


def test_registry_consistency():
  model_files = {fn for fns in weights.MODELS.values() for fn in fns}
  assert model_files == set(weights.WEIGHTS)
  for sha256, size in weights.WEIGHTS.values():
    assert len(sha256) == 64 and int(sha256, 16) >= 0
    assert size > 0


def test_resolves_in_tree_file(tmp_path):
  target = tmp_path / 'so3lr.eqx'
  target.write_bytes(FAKE_BYTES)
  assert weights.resolve_checkpoint(target) == target


def test_lfs_pointer_stub_is_not_a_checkpoint(tmp_path, monkeypatch):
  monkeypatch.setenv('JAX_MD_CACHE', str(tmp_path / 'cache'))
  stub = tmp_path / 'so3lr.eqx'
  stub.write_bytes(LFS_POINTER)
  with pytest.raises(FileNotFoundError):
    weights.resolve_checkpoint(stub)


def test_falls_back_to_cache(tmp_path, monkeypatch):
  cache = tmp_path / 'cache'
  cache.mkdir()
  (cache / 'so3lr.eqx').write_bytes(FAKE_BYTES)
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  missing = tmp_path / 'pkg' / 'so3lr.eqx'
  assert weights.resolve_checkpoint(missing) == cache / 'so3lr.eqx'


def test_missing_checkpoint_error_is_actionable(tmp_path, monkeypatch):
  monkeypatch.setenv('JAX_MD_CACHE', str(tmp_path / 'cache'))
  missing = tmp_path / 'pkg' / 'so3lr.eqx'
  with pytest.raises(FileNotFoundError) as error:
    weights.resolve_checkpoint(missing)
  message = str(error.value)
  assert 'python -m jax_md._nn.weights so3lr' in message
  assert f'{weights.WEIGHTS_BASE_URL}/so3lr.eqx' in message
  assert 'JAX_MD_CACHE' in message


def test_cache_dir_env_override(tmp_path, monkeypatch):
  monkeypatch.setenv('JAX_MD_CACHE', str(tmp_path))
  assert weights.cache_dir() == tmp_path
  monkeypatch.delenv('JAX_MD_CACHE')
  monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
  assert weights.cache_dir() == tmp_path / 'xdg' / 'jax_md'


def test_fetch_verifies_sha256(tmp_path):
  source = tmp_path / 'asset.eqx'
  source.write_bytes(FAKE_BYTES)
  url = source.as_uri()
  good = hashlib.sha256(FAKE_BYTES).hexdigest()

  dest = tmp_path / 'cache' / 'asset.eqx'
  assert weights.fetch(url, dest, good, progress=False) == dest
  assert dest.read_bytes() == FAKE_BYTES

  bad_dest = tmp_path / 'cache' / 'asset2.eqx'
  with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
    weights.fetch(url, bad_dest, '0' * 64, progress=False)
  assert not bad_dest.exists()
  assert not list(bad_dest.parent.glob('asset2.eqx.*'))


def test_download_rejects_unknown_names():
  with pytest.raises(ValueError, match='Unknown checkpoint'):
    weights.download('nonexistent.eqx')
  with pytest.raises(ValueError, match='Unknown model'):
    weights.download_models(['nonexistent'])


def test_download_models_expands_families(tmp_path, monkeypatch):
  cache = tmp_path / 'cache'
  cache.mkdir()
  for filename in weights.WEIGHTS:
    (cache / filename).write_bytes(FAKE_BYTES)
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  paths = weights.download_models(['all'])
  assert sorted(p.name for p in paths) == sorted(weights.WEIGHTS)
  paths = weights.download_models(['aceff'])
  assert sorted(p.name for p in paths) == ['aceff_v1.1.eqx', 'aceff_v2.0.eqx']


# The tests below use the real checkpoint files and skip without them.

NN_DIR = Path(weights.__file__).resolve().parent


def real_checkpoint(family: str, filename: str) -> Path | None:
  path = NN_DIR / family / filename
  if path.is_file() and not weights.is_lfs_pointer(path):
    return path
  return None


def all_real_checkpoints():
  return [
    (family, filename, real_checkpoint(family, filename))
    for family, filenames in weights.MODELS.items()
    for filename in filenames
  ]


needs_weights = pytest.mark.skipif(
  all(path is None for _, _, path in all_real_checkpoints()),
  reason='checkpoint files not fetched',
)


@needs_weights
def test_registry_matches_real_files():
  checked = 0
  for _, filename, path in all_real_checkpoints():
    if path is None:
      continue
    sha256, size = weights.WEIGHTS[filename]
    assert path.stat().st_size == size, filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, filename
    checked += 1
  assert checked > 0


@needs_weights
def test_resolves_real_in_tree_checkpoint():
  path = real_checkpoint('so3lr', 'so3lr.eqx')
  if path is None:
    pytest.skip('so3lr.eqx not present')
  assert weights.resolve_checkpoint(path) == path


@needs_weights
def test_load_so3lr_from_cache_like_pypi_install(tmp_path, monkeypatch):
  # A PyPI install has no in-tree file and real bytes in the cache.
  import shutil

  from jax_md._nn.so3lr import model as so3lr_model

  path = real_checkpoint('so3lr', 'so3lr.eqx')
  if path is None:
    pytest.skip('so3lr.eqx not present')
  cache = tmp_path / 'cache'
  cache.mkdir()
  shutil.copy(path, cache / 'so3lr.eqx')
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  monkeypatch.setitem(
    so3lr_model.SO3LR_MODEL_PATHS, 'so3lr', tmp_path / 'absent' / 'so3lr.eqx'
  )
  model = so3lr_model.load_model('so3lr')
  assert type(model).__name__ == 'SO3LR'
