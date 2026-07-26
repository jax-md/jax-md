"""Tests for jax_md._nn.weights checkpoint resolution. No network access."""

import hashlib
import re
from pathlib import Path

import pytest

from jax_md._nn import weights

FAKE_BYTES = b'not a real checkpoint, but real enough for resolution\n'

LFS_POINTER = (
  b'version https://git-lfs.github.com/spec/v1\n'
  b'oid sha256:' + b'0' * 64 + b'\n'
  b'size 2129435\n'
)

NN_DIR = Path(weights.__file__).resolve().parent


def test_registry_consistency():
  model_files = {fn for fns in weights.MODELS.values() for fn in fns}
  assert model_files == set(weights.WEIGHTS)
  for sha256, size in weights.WEIGHTS.values():
    assert len(sha256) == 64 and int(sha256, 16) >= 0
    assert size > 0


def test_registry_matches_checkpoints():
  checked = 0
  for family, filenames in weights.MODELS.items():
    for filename in filenames:
      path = NN_DIR / family / filename
      if not path.is_file():
        continue
      sha256, size = weights.WEIGHTS[filename]
      if weights.is_lfs_pointer(path):
        text = path.read_text()
        assert re.search(r'sha256:([0-9a-f]+)', text).group(1) == sha256, (
          filename
        )
        assert int(re.search(r'size (\d+)', text).group(1)) == size, filename
      else:
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256, filename
        assert len(data) == size, filename
      checked += 1
  assert checked > 0


def test_resolve_checkpoint(tmp_path, monkeypatch):
  cache = tmp_path / 'cache'
  cache.mkdir()
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  monkeypatch.setitem(
    weights.WEIGHTS,
    'so3lr.eqx',
    (weights.WEIGHTS['so3lr.eqx'][0], len(FAKE_BYTES)),
  )
  in_tree = tmp_path / 'pkg' / 'so3lr.eqx'
  in_tree.parent.mkdir()

  in_tree.write_bytes(FAKE_BYTES)
  assert weights.resolve_checkpoint(in_tree) == in_tree

  in_tree.write_bytes(LFS_POINTER)
  (cache / 'so3lr.eqx').write_bytes(FAKE_BYTES)
  assert weights.resolve_checkpoint(in_tree) == cache / 'so3lr.eqx'

  (cache / 'so3lr.eqx').write_bytes(FAKE_BYTES + b'!')
  with pytest.raises(FileNotFoundError) as error:
    weights.resolve_checkpoint(in_tree)
  message = str(error.value)
  assert 'python -m jax_md._nn.weights so3lr' in message
  assert f'{weights.WEIGHTS_BASE_URL}/so3lr.eqx' in message
  assert 'JAX_MD_CACHE' in message


def test_resolve_checkpoint_explicit_override(tmp_path, monkeypatch):
  cache = tmp_path / 'cache'
  cache.mkdir()
  (cache / 'so3lr.eqx').write_bytes(FAKE_BYTES)
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  missing = tmp_path / 'pkg' / 'so3lr.eqx'
  with pytest.raises(FileNotFoundError):
    weights.resolve_checkpoint(missing, allow_cache=False)
  override = tmp_path / 'custom.eqx'
  override.write_bytes(FAKE_BYTES)
  assert weights.resolve_checkpoint(override, allow_cache=False) == override


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


def test_download(tmp_path, monkeypatch):
  (tmp_path / 'so3lr.eqx').write_bytes(FAKE_BYTES)
  monkeypatch.setenv('JAX_MD_CACHE', str(tmp_path / 'cache'))
  monkeypatch.setattr(weights, 'WEIGHTS_BASE_URL', tmp_path.as_uri())
  monkeypatch.setitem(
    weights.WEIGHTS,
    'so3lr.eqx',
    (hashlib.sha256(FAKE_BYTES).hexdigest(), len(FAKE_BYTES)),
  )
  dest = weights.download('so3lr.eqx', progress=False)
  assert dest.read_bytes() == FAKE_BYTES
  assert weights.download('so3lr.eqx', force=True, progress=False) == dest
  with pytest.raises(ValueError, match='Unknown checkpoint'):
    weights.download('nonexistent.eqx')


def test_download_models(tmp_path, monkeypatch):
  cache = tmp_path / 'cache'
  cache.mkdir()
  for filename in list(weights.WEIGHTS):
    (cache / filename).write_bytes(FAKE_BYTES)
    monkeypatch.setitem(
      weights.WEIGHTS,
      filename,
      (weights.WEIGHTS[filename][0], len(FAKE_BYTES)),
    )
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  assert sorted(p.name for p in weights.download_models(['all'])) == sorted(
    weights.WEIGHTS
  )
  assert sorted(p.name for p in weights.download_models(['aceff'])) == [
    'aceff_v1.1.eqx',
    'aceff_v2.0.eqx',
  ]
  assert [p.name for p in weights.download_models(['so3lr.eqx'])] == [
    'so3lr.eqx'
  ]
  with pytest.raises(ValueError, match='Unknown model'):
    weights.download_models(['nonexistent'])


def test_main_cli(tmp_path, monkeypatch, capsys):
  monkeypatch.setenv('JAX_MD_CACHE', str(tmp_path / 'cache'))
  assert weights.main(['--list']) == 0
  out = capsys.readouterr().out
  assert 'cache directory:' in out
  assert 'so3lr' in out
  assert weights.main([]) == 0
  capsys.readouterr()
  assert weights.main(['nonexistent']) == 1
  assert 'error:' in capsys.readouterr().err


def test_load_so3lr_from_cache_like_pypi_install(tmp_path, monkeypatch):
  import shutil

  from jax_md._nn.so3lr import model as so3lr_model

  src = NN_DIR / 'so3lr' / 'so3lr.eqx'
  if not src.is_file() or weights.is_lfs_pointer(src):
    pytest.skip('so3lr.eqx not fetched')
  cache = tmp_path / 'cache'
  cache.mkdir()
  shutil.copy(src, cache / 'so3lr.eqx')
  monkeypatch.setenv('JAX_MD_CACHE', str(cache))
  monkeypatch.setitem(
    so3lr_model.SO3LR_MODEL_PATHS, 'so3lr', tmp_path / 'absent' / 'so3lr.eqx'
  )
  model = so3lr_model.load_model('so3lr')
  assert type(model).__name__ == 'SO3LR'
