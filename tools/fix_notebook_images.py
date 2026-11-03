#!/usr/bin/env python3
"""
fix_notebook_images.py
Finds HTML <img src="..."/> occurrences in markdown cells of a notebook
and replaces them with standard Markdown image syntax: ![](path).

Usage:
    python3 tools/fix_notebook_images.py train_model/tech_challenge_tuberculose.ipynb

This is safe for simple cases where the image tag appears as a single string
in the markdown cell source. It writes the notebook back in-place.
"""
import json
import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python3 tools/fix_notebook_images.py PATH_TO_NOTEBOOK.ipynb")
    sys.exit(1)

notebook_path = Path(sys.argv[1])
if not notebook_path.exists():
    print(f"Notebook not found: {notebook_path}")
    sys.exit(2)

img_pattern = re.compile(r'<img\s+src="([^"]+)"\s*(?:alt="[^"]*")?\s*/?>')
changed = 0

nb = json.loads(notebook_path.read_text(encoding='utf-8'))
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'markdown':
        continue
    # source can be a list of lines or a single string
    src = cell.get('source')
    if isinstance(src, list):
        text = ''.join(src)
    else:
        text = src
    new_text, nsubs = img_pattern.subn(lambda m: f'![]({m.group(1)})', text)
    if nsubs > 0:
        changed += nsubs
        # preserve original structure (list of lines) if it was a list
        if isinstance(src, list):
            # split to lines keeping newline chars
            cell['source'] = [line for line in new_text.splitlines(keepends=True)]
        else:
            cell['source'] = new_text

if changed > 0:
    notebook_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Replaced {changed} <img> tags in {notebook_path}")
else:
    print("No <img> tags found/changed.")
