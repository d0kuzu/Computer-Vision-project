"""Build the executable benchmark notebook while retaining the self-collected template."""
import json
from pathlib import Path
import shutil
import build_notebook

shutil.copyfile('handwritten_recognition.ipynb', 'self_collected_template.ipynb')
nb = json.loads(Path('handwritten_recognition.ipynb').read_text(encoding='utf-8'))
def put(i, source): nb['cells'][i]['source'] = source.strip().splitlines(True)
def get(i): return ''.join(nb['cells'][i]['source'])
put(0, '''# Handwritten Digit Recognition: A Reproducible Mini-Project

**Scope:** classification of isolated digits 0-9 using logistic regression and a three-block CNN.

**Data provenance:** this executed experiment uses 2,000 real handwritten images sampled from public MNIST, not handwriting collected by the student. It demonstrates the complete technical pipeline, but does **not** fulfill the assignment's self-collection requirement. `self_collected_template.ipynb` supports a replacement dataset with writer-disjoint splits.

MNIST attribution: Yann LeCun and Corinna Cortes; derived from NIST data. Source and license: https://keras.io/api/datasets/mnist/ (CC BY-SA 3.0). This selected, inverted PNG subset is distributed under the same dataset license. Download location documented by TensorFlow: https://www.tensorflow.org/tutorials/load_data/numpy .

Run from the project folder. Install `requirements.txt` before running. The exported PNG dataset and manifest are included; downloading MNIST again is needed only to rebuild that dataset.''')
put(1, '''# Dependencies: python -m pip install -r requirements.txt
# In Colab, uncomment the following line:
# %pip install numpy pandas pillow matplotlib scikit-learn torch torchvision joblib''')
put(2, '''## 1. Setup
Seed 42 is fixed for image selection, augmentation, and model initialization. CPU execution uses up to four threads. Each class contains 140 training images, 30 validation images, and 30 test images. Training and validation are sampled from the official training partition; the test subset comes from the official test partition. No writer IDs are supplied in this archive, so writer-disjoint validation cannot be asserted.''')
put(3, get(3).replace("ROOT = Path('data/raw')", "ROOT = Path('data/mnist_subset')").replace("VAL_WRITERS = {'w15', 'w16', 'w17'}\nTEST_WRITERS = {'w18', 'w19', 'w20'}\nassert not VAL_WRITERS & TEST_WRITERS", "import os, time\ntorch.set_num_threads(min(4, os.cpu_count() or 1))\nexperiment_started = time.perf_counter()"))
put(4, '''## 2. Load and audit the frozen dataset
`prepare_data.py` deterministically selects and exports the 2,000-image subset. The manifest records the original MNIST partition and index, split, class, and pixel hash. Exact duplicate images are excluded during selection. This check detects changed files, overlap, label mismatches, and unexpected split sizes. Near-duplicate or stylistically similar handwriting is not eliminated by a pixel hash.''')
put(5, '''manifest = ROOT/'manifest.csv'
assert manifest.exists(), 'Run python prepare_data.py first, or restore the included dataset.'
df = pd.read_csv(manifest, dtype={'class_name':str})
classes = [str(i) for i in range(10)]
df['path'] = df['relative_path'].map(lambda p: str(ROOT/p))
assert len(df) == 2000 and not df['relative_path'].duplicated().any()
assert not df.duplicated(['source_partition','source_index']).any()
assert set(df['split']) == {'train','val','test'}
hashes = []
for row in df.itertuples():
    assert row.class_name == classes[row.label]
    assert row.source_partition == ('test' if row.split == 'test' else 'train')
    with Image.open(row.path) as im:
        im = im.convert('L')
        assert im.size == (28,28)
        digest = hashlib.sha256(im.tobytes()).hexdigest()
    assert digest == row.sha256, f'Changed image: {row.path}'
    hashes.append(digest)
assert len(set(hashes)) == len(hashes), 'Exact duplicates found.'
counts = pd.crosstab(df.class_name,df.split).reindex(index=classes,columns=['train','val','test'])
assert (counts.to_numpy() == np.array([140,30,30])).all()
display(counts)
df.to_csv(OUT/'dataset_manifest.csv',index=False)
counts.to_csv(OUT/'dataset_counts.csv')
(OUT/'classes.json').write_text(json.dumps(classes),encoding='utf-8')
parts = {s:df[df.split==s].reset_index(drop=True) for s in ['train','val','test']}
print('Audit passed: 2,000 unique images; 1,400 train / 300 validation / 300 test.')''')
put(7, get(7).replace("fig, axes = plt.subplots(2,8,figsize=(12,3))\nfor ax,i in zip(axes.flat, range(min(16,len(parts['train'])))):", "fig, axes = plt.subplots(2,10,figsize=(12,3))\nexample_ids = [int(parts['train'].index[parts['train'].label==c][j]) for j in range(2) for c in range(10)]\nfor ax,i in zip(axes.flat, example_ids):") + '''
fig,axes = plt.subplots(2,10,figsize=(12,3))
for c in range(10):
    i = int(parts['train'].index[parts['train'].label==c][0])
    for r,offset in enumerate([0,len(parts['train'])]):
        axes[r,c].imshow(Xtrain[i+offset,0],cmap='gray')
        axes[r,c].axis('off'); axes[r,c].set_title(f'{c}: '+('original' if r==0 else 'augmented'),fontsize=8)
plt.tight_layout(); plt.savefig(OUT/'augmentation_examples.png',dpi=180); plt.show()
''')
put(9, 'baseline_started = time.perf_counter()\n' + get(9) + '''
baseline_seconds = time.perf_counter()-baseline_started
print('Selected C:', baseline.C, '| seconds:', round(baseline_seconds,2))
''')
put(11, 'cnn_started = time.perf_counter()\n' + get(11).replace("history.append({'epoch':epoch,'train_loss':total_loss/len(Xtrain),'val_macro_f1':f1})", "with torch.no_grad():\n        val_loss = loss_fn(cnn(Xval.to(DEVICE)),yval.to(DEVICE)).item()\n    history.append({'epoch':epoch,'train_loss':total_loss/len(Xtrain),'val_loss':val_loss,'val_macro_f1':f1})").replace("hist.plot(x='epoch',y='train_loss',ax=axes[0]);", "hist.plot(x='epoch',y=['train_loss','val_loss'],ax=axes[0]);") + '''
cnn_seconds = time.perf_counter()-cnn_started
print('CNN parameters:', sum(p.numel() for p in cnn.parameters()), '| seconds:', round(cnn_seconds,2))
''')
put(13, get(13) + '''
# Paired bootstrap: same resampled test images for both models, seed fixed.
# Describes this small image-level test sample, not writer-level uncertainty.
rng = np.random.default_rng(SEED)
truth = ytest.numpy()
boot = rng.integers(0,len(truth),size=(5000,len(truth)))
correct = {name:(p==truth).astype(float) for name,p in predictions.items()}
intervals = {name:np.quantile(v[boot].mean(axis=1),[0.025,0.975]).tolist() for name,v in correct.items()}
delta = correct['CNN']-correct['Logistic regression']
intervals['CNN_minus_baseline'] = np.quantile(delta[boot].mean(axis=1),[0.025,0.975]).tolist()
(OUT/'accuracy_bootstrap.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
print('Image-level bootstrap accuracy intervals:',intervals)
''')
put(14, '''## 7. Measured conclusion and limitations
The next cell generates a conclusion directly from the saved measurements. See `output/pdf/assignment_report.pdf` for the four-page report and `REPORT.md` for editable text.

This experiment does not test mathematical symbols, full equations, phone photographs, or unseen-writer validation. It uses public MNIST images, not a self-collected dataset. Evaluation uses one seed and a small balanced test subset. Both models use exactly the same original and augmented training examples. Repeated trials and fresh handwriting are appropriate follow-up work.''')
put(15,get(15)+'''
run = dict(seed=SEED,dataset='MNIST subset (not self-collected)',train_original=1400,train_with_augmentation=2800,validation=300,test=300,
           baseline_C=baseline.C,baseline_seconds=baseline_seconds,cnn_seconds=cnn_seconds,
           cnn_parameters=sum(p.numel() for p in cnn.parameters()),epochs_run=len(hist),
           best_epoch=int(hist.loc[hist.val_macro_f1.idxmax(),'epoch']),
           best_validation_macro_f1=float(hist.val_macro_f1.max()),
           total_seconds=time.perf_counter()-experiment_started)
(OUT/'run_summary.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
a = summary.set_index('model')
gap = 100*(a.loc['CNN','accuracy']-a.loc['Logistic regression','accuracy'])
print(f"Test accuracy: logistic regression {a.loc['Logistic regression','accuracy']:.2%}; CNN {a.loc['CNN','accuracy']:.2%}.")
print(f'CNN accuracy difference: {gap:+.2f} percentage points. Test size: 300 images.')
print('Remaining assignment requirement: collect original handwriting and repeat the experiment.')
''')
Path('handwritten_recognition.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
for c in nb['cells']:
    if c['cell_type']=='code': compile(''.join(c['source']),'notebook','exec')
print('Benchmark notebook built and syntax checked.')
