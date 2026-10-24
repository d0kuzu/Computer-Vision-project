"""Package deliverables without virtual environments, caches or source download."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
dataset=OUT/'mnist_subset_2000.zip'
with zipfile.ZipFile(dataset,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT/'data/mnist_subset').rglob('*')):
        if path.is_file(): archive.write(path,path.relative_to(ROOT/'data'))
files=[ROOT/name for name in ['README.md','START_HERE.md','DEFENSE_RU.md','REPORT.md','requirements.txt',
       'handwritten_recognition.ipynb','self_collected_template.ipynb','build_notebook.py','prepare_project.py',
       'prepare_data.py','run_notebook.py','make_report.py','predict.py','demo.py','verify_project.py','package_project.py']]
for folder in ['data/mnist_subset','results','output/pdf']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
target=OUT/'handwritten_digits_project.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):archive.write(path,Path('handwritten_digits_project')/path.relative_to(ROOT))
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    names=archive.namelist()
    assert sum('/data/mnist_subset/' in n and n.endswith('.png') for n in names)==2000
    assert not any('.venv/' in n for n in names)
print(f'{target}: {target.stat().st_size/1024/1024:.2f} MB; {len(files)} files')
print(f'{dataset}: {dataset.stat().st_size/1024/1024:.2f} MB')
