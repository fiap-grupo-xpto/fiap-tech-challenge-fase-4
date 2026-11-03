import nbformat
from nbclient import NotebookClient
from pathlib import Path

p = Path('ag_analise_cancer.ipynb')
nb = nbformat.read(p.open('r', encoding='utf-8'), as_version=4)
# choose kernel from notebook metadata if present, else fall back to python3
kernel = nb.metadata.get('kernelspec', {}).get('name', 'python3')
client = NotebookClient(nb, timeout=1200, kernel_name=kernel)
try:
    client.execute()
    nbformat.write(nb, p.open('w', encoding='utf-8'))
    print('EXECUTION_OK')
except Exception as e:
    import traceback, sys
    traceback.print_exc()
    # save partial notebook with outputs/errors for inspection
    outp = p.with_name('ag_analise_cancer.executed.ipynb')
    nbformat.write(nb, outp.open('w', encoding='utf-8'))
    print('EXECUTION_ERROR', e)
    sys.exit(2)
