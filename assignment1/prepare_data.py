"""Download MNIST once, export a deterministic balanced 2,000-image subset."""
from pathlib import Path
import hashlib, json, urllib.request
import numpy as np
import pandas as pd
from PIL import Image

URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
ROOT = Path(__file__).resolve().parent
cache = ROOT/'data/source/mnist.npz'
target = ROOT/'data/mnist_subset'

def main():
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():
        print('Downloading public MNIST...',flush=True)
        urllib.request.urlretrieve(URL,cache)
    rng = np.random.default_rng(42)
    rows,seen = [],set()
    with np.load(cache,allow_pickle=False) as source:
        assert source['x_train'].shape == (60000,28,28)
        assert source['x_test'].shape == (10000,28,28)
        for label in range(10):
            for partition, wanted in [('train',170),('test',30)]:
                x,y = source['x_'+partition],source['y_'+partition]
                accepted = 0
                for index in rng.permutation(np.flatnonzero(y==label)):
                    image = 255-x[index]  # exported as dark ink on white
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in seen: continue
                    seen.add(digest)
                    split = 'test' if partition=='test' else 'train' if accepted<140 else 'val'
                    relative = f'{label}/mnist_{partition}_{index:05d}.png'
                    path = target/relative; path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(image).save(path)
                    rows.append(dict(relative_path=relative,label=label,class_name=str(label),split=split,
                                     source_partition=partition,source_index=int(index),sha256=digest))
                    accepted += 1
                    if accepted==wanted: break
                assert accepted==wanted
    pd.DataFrame(rows).to_csv(target/'manifest.csv',index=False)
    metadata = dict(source='MNIST',download_url=URL,documentation='https://keras.io/api/datasets/mnist/',
                    source_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),seed=42,
                    self_collected=False,writer_ids_available=False,images=2000,
                    modifications='Balanced subset, dark-on-white inversion, PNG export',
                    attribution='Yann LeCun and Corinna Cortes; derived from NIST datasets',
                    license='CC BY-SA 3.0',license_url='https://creativecommons.org/licenses/by-sa/3.0/')
    (target/'PROVENANCE.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (target/'README.txt').write_text('Public MNIST subset; NOT self-collected.\nCopyright: Yann LeCun and Corinna Cortes. Derived from NIST.\nLicense: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/\nModifications: subset selection, grayscale inversion, PNG conversion.\nSource: https://keras.io/api/datasets/mnist/\n140 train + 30 validation + 30 test images per digit. See manifest.csv.\n',encoding='utf-8')
    print(f'Exported {len(rows)} images to {target}',flush=True)

if __name__=='__main__': main()
