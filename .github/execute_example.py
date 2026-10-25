"""Execute an example script as a notebook, keeping cell errors in its outputs.

Usage: python execute_example.py <example> <output dir>, where <example> is a
path under examples/ without the .py suffix, e.g. units/nve_si_sw. The notebook
is written to <output dir>/<example>.ipynb even if the kernel dies.
"""

import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

example, out = sys.argv[1], Path(sys.argv[2])
script = Path('examples') / f'{example}.py'
notebook = jupytext.read(script)
try:
  NotebookClient(
    notebook,
    timeout=None,
    allow_errors=True,
    resources={'metadata': {'path': str(script.parent)}},
  ).execute()
finally:
  path = out / f'{example}.ipynb'
  path.parent.mkdir(parents=True, exist_ok=True)
  nbformat.write(notebook, path)

errors = [
  f'{output.ename}: {output.evalue}'
  for cell in notebook.cells
  if cell.cell_type == 'code'
  for output in cell.get('outputs', [])
  if output.output_type == 'error'
]
for error in errors:
  print(f'::error title={example}::{error}')
sys.exit(1 if errors else 0)
