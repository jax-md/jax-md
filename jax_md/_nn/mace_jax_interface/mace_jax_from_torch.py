"""Convert a pre-trained Torch MACE foundation model to MACE-JAX parameters."""

from __future__ import annotations

import argparse
import importlib
import json
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings(
  'ignore',
  message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*',
  category=UserWarning,
)


def load_torch_model_from_foundations(source: str, model: str | None):
  foundations_models = importlib.import_module(
    'mace.calculators.foundations_models'
  )

  source = source.lower()
  if source not in {'mp', 'off', 'anicc', 'omol'}:
    raise ValueError(
      "Unknown foundation source. Supported values are 'mp', 'off', 'anicc', 'omol'."
    )

  loader_kwargs: dict[str, Any] = {'device': 'cpu'}
  loader = None
  if source in {'mp', 'off', 'omol'}:
    loader = getattr(
      foundations_models, f'mace_{"mp" if source == "mp" else source}'
    )
    if model is not None:
      loader_kwargs['model'] = model
  else:  # anicc
    loader = getattr(foundations_models, 'mace_anicc')
    if model is not None:
      loader_kwargs['model_path'] = model

  try:
    return loader(return_raw_model=True, **loader_kwargs)
  except Exception:
    calc = loader(return_raw_model=False, **loader_kwargs)
    torch_model = getattr(calc, 'model', None)
    if torch_model is None:
      models_attr = getattr(calc, 'models', None)
      if models_attr:
        torch_model = models_attr[0]
    if torch_model is None:
      raise
    return torch_model


def maybe_update_hidden_irreps_from_torch(
  torch_model, config: dict[str, Any]
) -> None:
  from jax_md._nn.mace_jax_interface.model_builder import as_irreps

  try:
    num_interactions = int(config.get('num_interactions', 0))
  except Exception:
    return
  if num_interactions != 1:
    return

  if 'hidden_irreps' not in config:
    return

  torch_hidden = None
  try:
    products = getattr(torch_model, 'products', None)
    if products:
      linear = getattr(products[0], 'linear', None)
      if linear is not None:
        torch_hidden = getattr(linear, 'irreps_out', None)
    if torch_hidden is None:
      torch_hidden = getattr(torch_model, 'hidden_irreps', None)
  except Exception:
    torch_hidden = None

  if torch_hidden is None:
    return

  try:
    torch_irreps = as_irreps(torch_hidden)
    config_irreps = as_irreps(config['hidden_irreps'])
  except Exception:
    return

  # Torch started collapsing single-interaction hidden irreps in
  # mace commit f599b0e ("fix the 1 layer model cueq"); switch to legacy
  # mode when the Torch model still carries multiple irreps.
  collapse_hidden = len(torch_irreps) <= 1
  config['collapse_hidden_irreps'] = collapse_hidden

  if torch_irreps != config_irreps:
    config['hidden_irreps'] = str(torch_irreps)


def serialize_for_json(value: Any) -> Any:
  if is_dataclass(value):
    return {str(k): serialize_for_json(v) for k, v in asdict(value).items()}
  if isinstance(value, dict):
    return {str(k): serialize_for_json(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [serialize_for_json(v) for v in value]
  if isinstance(value, set):
    return [serialize_for_json(v) for v in sorted(value)]
  if (
    hasattr(value, 'detach')
    and hasattr(value, 'cpu')
    and hasattr(value, 'numpy')
  ):
    arr = value.detach().cpu().numpy()
    if arr.ndim == 0:
      return arr.item()
    return arr.tolist()
  value_module = getattr(value.__class__, '__module__', '')
  value_class = value.__class__.__name__
  if value_module.startswith('torch') and value_class == 'device':
    return str(value)
  if value_module.startswith('torch') and value_class == 'dtype':
    return str(value).replace('torch.', '')
  if value_module.startswith('torch') and value_class == 'Size':
    return list(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.integer, np.floating)):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, type):
    return value.__name__
  if callable(value):
    name = getattr(value, '__name__', None)
    if name:
      return name
    cls = getattr(value, '__class__', None)
    if cls is not None:
      return getattr(cls, '__name__', str(value))
  return value


def convert_model(
  torch_model,
  config: dict[str, Any],
  *,
  cueq_config: Any | None = None,
  n_template: int | None = None,
  e_template: int | None = None,
):
  from flax import nnx
  from mace_jax.nnx_utils import state_to_serializable_dict
  from mace_jax.tools.import_from_torch import import_from_torch
  from jax_md._nn.mace_jax_interface.model_builder import (
    build_jax_model,
    prepare_template_data,
  )

  # avoid mutating caller config
  config = dict(config)

  if not hasattr(nnx.Variable, 'get_value'):
    nnx.Variable.get_value = lambda self: self.raw_value  # pylint: disable=protected-access
  if not hasattr(nnx.Variable, 'set_value'):
    nnx.Variable.set_value = lambda self, value: setattr(
      self, 'raw_value', value
    )

  maybe_update_hidden_irreps_from_torch(torch_model, config)

  jax_model = build_jax_model(
    config,
    cueq_config=cueq_config,
    rngs=nnx.Rngs(0),
  )

  template_data = prepare_template_data(
    config,
    n_node=n_template,
    n_edge=e_template,
  )

  # Split NNX module into graphdef/state
  graphdef, state = nnx.split(jax_model)

  # Import torch weights into NNX state (in-place mutation of `state`)
  import_from_torch(jax_model, torch_model, state)

  # Extract normalize2mom consts (if present) into config copy
  variables = state_to_serializable_dict(state)
  consts_loaded = None
  if isinstance(variables, dict) or hasattr(variables, 'get'):
    consts_loaded = variables.get('_normalize2mom_consts_var')
  if consts_loaded:
    config['normalize2mom_consts'] = {
      key: float(np.asarray(val)) for key, val in consts_loaded.items()
    }

  # Return NNX artifacts (recommended)
  return graphdef, state, template_data, config


def main():
  import torch
  from flax import serialization
  from mace_jax.nnx_utils import state_to_pure_dict

  scripts_utils = importlib.import_module('mace.tools.scripts_utils')
  extract_config_mace_model = getattr(
    scripts_utils, 'extract_config_mace_model'
  )

  parser = argparse.ArgumentParser(
    description='Convert Torch MACE model to JAX parameters'
  )
  parser.add_argument(
    '--torch-model',
    help='Optional path to a Torch checkpoint. If omitted, a foundation model is downloaded.',
  )
  parser.add_argument(
    '--foundation',
    default='mp',
    choices=['mp', 'off', 'anicc', 'omol'],
    help='Foundation family to download when --torch-model is not provided.',
  )
  parser.add_argument(
    '--model-name',
    help='Specific foundation variant (e.g., "medium-mpa-0"). See foundations_models for options.',
  )
  parser.add_argument(
    '--output',
    help=(
      "Output file for serialized JAX parameters. Defaults to '<checkpoint>-jax.msgpack' "
      "(or '<source>-<model>-jax.msgpack' for foundation downloads)."
    ),
  )
  parser.add_argument(
    '--n-template',
    type=int,
    help='Optional atom capacity for the fixed-shape JAX-MD template.',
  )
  parser.add_argument(
    '--e-template',
    type=int,
    help='Optional edge capacity for the fixed-shape JAX-MD template.',
  )
  args = parser.parse_args()

  if args.torch_model:
    bundle = torch.load(args.torch_model, map_location='cpu')
    torch_model = (
      bundle['model']
      if isinstance(bundle, dict) and 'model' in bundle
      else bundle
    )
    default_output = Path(args.torch_model).with_name(
      Path(args.torch_model).stem + '-jax.msgpack'
    )
  else:
    torch_model = load_torch_model_from_foundations(
      args.foundation, args.model_name
    )
  torch_model.eval()

  if args.output is None:
    if args.torch_model:
      output_path = default_output
    else:
      model_tag = args.model_name or args.foundation
      output_path = Path(f'{args.foundation}-{model_tag}-jax.msgpack')
  else:
    output_path = Path(args.output)

  config = extract_config_mace_model(torch_model)
  if 'error' in config:
    raise RuntimeError(config['error'])
  config['torch_model_class'] = torch_model.__class__.__name__

  graphdef, state, template_data, config = convert_model(
    torch_model,
    config,
    n_template=args.n_template,
    e_template=args.e_template,
  )
  del graphdef
  config['template_n_node'] = int(template_data['positions'].shape[0])
  config['template_n_edge'] = int(template_data['edge_index'].shape[1])

  variables = state_to_pure_dict(state)
  params_bytes = serialization.to_bytes(variables)
  output_path.write_bytes(params_bytes)
  print(f'Serialized JAX parameters written to {output_path}')
  # Persist config alongside parameters.
  config_path = output_path.with_suffix('.json')
  config_path.write_text(json.dumps(serialize_for_json(config), indent=2))
  print(f'Config written to {config_path}')


if __name__ == '__main__':
  main()
