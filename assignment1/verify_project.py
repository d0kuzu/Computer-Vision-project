"""Check delivered results and inference against the executed notebook."""
from pathlib import Path
from io import BytesIO
import base64,json,threading,urllib.request,urllib.error
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import nbformat
from pypdf import PdfReader
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from predict import load_model,preprocess
from demo import Handler,HTTPServer

root=Path(__file__).resolve().parent
nb=nbformat.read(root/'handwritten_recognition.ipynb',as_version=4); nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type=='code':
        assert cell.execution_count is not None
        assert not any(o.output_type=='error' for o in cell.outputs)
model,classes=load_model()
def load_inputs(predictions):
    images=[]
    for path in predictions.path:
        with Image.open(root/Path(path.replace('\\','/'))) as im:images.append(preprocess(im))
    return torch.cat(images)

pred=pd.read_csv(root/'results/cnn_predictions.csv')
with torch.inference_mode(): actual=torch.cat([model(x).argmax(1) for x in load_inputs(pred).split(64)]).numpy()
assert np.array_equal(actual,pred.predicted_class.to_numpy()),'Saved model/CLI preprocessing differs from notebook.'
baseline=joblib.load(root/'results/logistic_regression.joblib')
baseline_pred=pd.read_csv(root/'results/logistic_regression_predictions.csv')
baseline_actual=baseline.predict(load_inputs(baseline_pred).flatten(1).numpy())
assert np.array_equal(baseline_actual,baseline_pred.predicted_class.to_numpy()),'Saved baseline differs from notebook.'
metrics=pd.read_csv(root/'results/model_comparison.csv').set_index('model')
for name,slug in [('CNN','cnn'),('Logistic regression','logistic_regression')]:
    p=pd.read_csv(root/f'results/{slug}_predictions.csv')
    assert len(p)==300
    precision,recall,f1,_=precision_recall_fscore_support(p.label,p.predicted_class,average='macro',zero_division=0)
    assert np.allclose(metrics.loc[name].to_numpy(),[accuracy_score(p.label,p.predicted_class),precision,recall,f1])
assert len(PdfReader(root/'output/pdf/assignment_report.pdf').pages)==5
server=HTTPServer(('127.0.0.1',0),Handler);server.model,server.classes=model,classes
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
url=f'http://127.0.0.1:{server.server_port}'
try:
    with urllib.request.urlopen(url) as response:assert b'<canvas' in response.read()
    def post(im):
        buffer=BytesIO();im.save(buffer,format='PNG')
        body=json.dumps({'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}).encode()
        return urllib.request.urlopen(urllib.request.Request(url+'/predict',data=body,headers={'Content-Type':'application/json'}))
    with Image.open(root/Path(pred.iloc[0].path.replace('\\','/'))) as sample:
        with post(sample.resize((280,280))) as response:
            result=json.load(response);assert result['digit'] in classes and len(result['scores'])==10
            assert abs(sum(x['score'] for x in result['scores'])-1)<1e-5
    try:post(Image.new('L',(280,280),255))
    except urllib.error.HTTPError as error:assert error.code==400
    else:raise AssertionError('Blank drawing should be rejected.')
finally:
    server.shutdown();server.server_close();thread.join()
summary={'notebook':'all code cells executed, no errors','cnn_reload':'300/300 predictions match notebook',
         'baseline_reload':'300/300 predictions match notebook',
         'metrics':'recomputed from saved predictions','report_pages':5,'demo':'HTTP page, prediction and blank-input checks passed'}
(root/'results/verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
