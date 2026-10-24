"""Execute all notebook cells with this interpreter, preserving actual outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
nb = nbformat.read(root/'handwritten_recognition.ipynb',as_version=4)
nbformat.validate(nb)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client = NotebookClient(nb,km=km,timeout=900,resources={'metadata':{'path':str(root)}})
client.on_cell_start = lambda cell,cell_index: print(f'Executing cell {cell_index+1}/{len(nb.cells)}',flush=True)
try:
    client.execute()
finally:
    nbformat.write(nb,root/'handwritten_recognition.ipynb')
print('Notebook executed and saved with real outputs.',flush=True)
