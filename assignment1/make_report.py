"""Create a five-page report from actual experiment outputs, plus editable Markdown."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
R = ROOT/'results'
metrics = pd.read_csv(R/'model_comparison.csv').set_index('model')
run = json.loads((R/'run_summary.json').read_text())
ci = json.loads((R/'accuracy_bootstrap.json').read_text())
history = pd.read_csv(R/'training_history.csv')
pred = pd.read_csv(R/'cnn_predictions.csv')
baseline_pred = pd.read_csv(R/'logistic_regression_predictions.csv')
wrong = pred[pred.label != pred.predicted_class]
confusions = wrong.groupby(['label','predicted_class']).size().sort_values(ascending=False)
top = '; '.join(f'{int(a)} predicted as {int(b)} ({n} images)' for (a,b),n in confusions.head(4).items()) or 'none'
baseline_errors = int((baseline_pred.label != baseline_pred.predicted_class).sum())
gap = (metrics.loc['CNN','accuracy']-metrics.loc['Logistic regression','accuracy'])*100
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=25,leading=29,textColor=colors.HexColor('#152a43'),spaceAfter=18))
styles.add(ParagraphStyle(name='Subtitle',fontSize=11,leading=16,textColor=colors.HexColor('#476079'),spaceAfter=15))
styles['BodyText'].fontSize=10; styles['BodyText'].leading=14; styles['BodyText'].spaceAfter=9
styles['Heading1'].fontSize=18; styles['Heading1'].leading=23; styles['Heading1'].textColor=colors.HexColor('#152a43')
styles['Heading2'].fontSize=12; styles['Heading2'].leading=16; styles['Heading2'].spaceBefore=10
styles.add(ParagraphStyle(name='CaptionCustom',fontSize=8,leading=11,textColor=colors.HexColor('#526174'),spaceAfter=12))
story=[]; md=[]
def p(text, style='BodyText'):
    story.append(Paragraph(escape(text),styles[style])); md.append(text+'\n')
def h(text): p(text,'Heading1')
def pic(name,width,height,caption):
    story.append(Image(str(R/name),width=width,height=height)); p(caption,'CaptionCustom')
def table(rows,widths):
    t=Table([[Paragraph(escape(str(v)),styles['CaptionCustom']) for v in row] for row in rows],colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3ecf6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8da4bf')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')])]))
    story.append(t); story.append(Spacer(1,10))
    md.extend([' | '.join(str(v) for v in row)+'\n' for row in rows])
def page(): story.append(PageBreak())

p('Handwritten Digit\nRecognition'.replace('\n',' '),'TitleCustom')
p('Computer Vision - Assignment 1 | Reproducible mini-project\nLogistic regression and a convolutional neural network','Subtitle')
h('1. Objective and dataset')
p('The objective is to classify an isolated handwritten digit into one of ten classes (0-9), and compare a linear baseline with a CNN under the same data budget. Mathematical symbols and multi-character expressions are outside this experiment. All reported results were obtained by executing the accompanying notebook.')
p('Dataset provenance: this project uses public MNIST images, not a self-collected dataset. Therefore, the technical experiment is complete, but the assignment requirement to collect original handwriting remains unmet. A separate notebook template is included for a future writer-separated collection.')
p('MNIST provides 28 x 28 grayscale images of handwritten digits [1]. We selected 200 unique images per class (2,000 total), using NumPy random seed 42. For each class, 170 images came from the official training partition and 30 from its test partition. The 170 training-partition images were divided into 140 training and 30 validation examples.')
table([['Partition','Per class','Total','Purpose'],['Training','140','1,400','Fit model parameters'],['Validation','30','300','Select C and CNN checkpoint'],['Test','30','300','Final held-out evaluation']],[105,75,70,235])
pic('training_examples.png',480,120,'Figure 1. Two training samples per digit after preprocessing. These are public handwritten images, not synthetic fonts.')
p('Each exported PNG is stored in a class folder. A manifest preserves the original partition and sample index, selected split, and pixel SHA-256. Exact pixel duplicates are excluded and audited; near-duplicates are not ruled out. The source archive does not include writer IDs, so writer independence of validation cannot be verified.')
page()
h('2. Preprocessing and models')
p('Exported PNG files use dark ink on a light background. The shared preprocessing converts to grayscale, inverts intensities, fits the image into a 24 x 24 area without changing aspect ratio, and centers it on a black 32 x 32 canvas. Pixel values are divided by 255 to obtain [0,1] inputs. No statistics are estimated from validation or test images.')
p('Training augmentation adds exactly one transformed copy per original image: random rotation between -10 and +10 degrees and translation up to 8% in each direction, with black fill. The resulting 2,800 training inputs are identical for both models. Validation and test samples are not augmented. Flips are excluded because they can alter the meaning of a digit.')
pic('augmentation_examples.png',480,120,'Figure 2. Original inputs (top) and their fixed augmented training copies (bottom).')
p('Baseline model','Heading2')
p(f'Logistic regression uses 1,024 flattened pixel features, L2 regularization, and a maximum of 2,000 optimization iterations. C was selected from {{0.1, 1, 10}} using validation macro F1; the selected value was {run["baseline_C"]:g}. Both models are evaluated on the same untouched test images.')
p('Convolutional neural network','Heading2')
table([['Block','Operation','Output'],['1','Conv 3x3, 32 channels; ReLU; max pool 2x2','32 x 16 x 16'],['2','Conv 3x3, 64 channels; ReLU; max pool 2x2','64 x 8 x 8'],['3','Conv 3x3, 128 channels; ReLU; max pool 2x2','128 x 4 x 4'],['Classifier','Flatten; dense 128; ReLU; dropout 0.4; dense 10','10 logits']],[65,315,105])
p(f'The CNN has {run["cnn_parameters"]:,} trainable parameters. It uses cross-entropy loss and Adam (learning rate 0.001, weight decay 0.0001), batch size 64, and at most 30 epochs. Early stopping allows six epochs without validation macro-F1 improvement. The run ended after {run["epochs_run"]} epochs; checkpoint {run["best_epoch"]} was selected. No test measurements informed checkpoint selection.')
page()
h('3. Held-out evaluation')
p('Accuracy is the fraction of correctly classified images. Precision, recall, and F1 are calculated per class and then averaged equally across the ten classes (macro averaging). All classes have 30 test images. Zero-denominator scores are set to zero. Detailed per-class reports and individual predictions are exported as CSV files.')
rows=[['Model','Accuracy','Macro precision','Macro recall','Macro F1']]
for name,row in metrics.iterrows(): rows.append([name]+[f'{row[k]*100:.2f}%' for k in ['accuracy','precision_macro','recall_macro','f1_macro']])
table(rows,[120,80,100,95,90])
p(f'Logistic regression made {baseline_errors} errors out of 300 test images; the CNN made {len(wrong)}. The CNN accuracy difference was {gap:+.2f} percentage points. This is a measured comparison for the selected subset and training run, not a guarantee that the same gap will hold on new handwriting.')
ims=Table([[Image(str(R/'logistic_regression_confusion_matrix.png'),width=242,height=215),Image(str(R/'cnn_confusion_matrix.png'),width=242,height=215)]],colWidths=[242,242])
story.append(ims)
p('Figure 3. Test confusion matrices. Rows are true labels; columns are predicted labels. Diagonal cells are correct classifications. Counts in each row sum to 30.','CaptionCustom')
p('Sampling uncertainty','Heading2')
for name in ['Logistic regression','CNN']:
    low,high=ci[name]
    p(f'{name}: the image-level percentile bootstrap 95% accuracy interval is {low*100:.1f}% to {high*100:.1f}%.')
lo,hi=ci['CNN_minus_baseline']
p(f'Using 5,000 paired bootstrap resamples, the accuracy-difference interval (CNN minus baseline) is {lo*100:+.1f} to {hi*100:+.1f} percentage points. The pairing preserves the fact that both models classify the same images. These intervals do not capture training-seed variability or writer clustering; they should be treated as descriptive uncertainty for this test subset.')
page()
h('4. Learning behavior and errors')
pic('learning_curves.png',480,144,'Figure 4. Training/validation cross-entropy and validation macro F1. The best validation-F1 checkpoint is restored before testing.')
best=history.loc[history.val_macro_f1.idxmax()]
p(f'Validation macro F1 reached {best.val_macro_f1*100:.2f}% at epoch {int(best.epoch)}. At that epoch, training loss was {best.train_loss:.4f} and validation loss was {best.val_loss:.4f}. Training uses dropout and augmented images, while validation uses evaluation mode and original images, so the two loss curves are not directly equivalent estimates.')
p(f'The most frequent CNN confusion pairs were: {top}. These counts come directly from held-out predictions. The error montage provides examples for qualitative inspection; appearance-based explanations remain hypotheses rather than established causes.')
if len(wrong): pic('cnn_errors.png',400,280,'Figure 5. Up to twelve CNN test mistakes, in manifest order. Labels show the ground truth and prediction; empty panels mean fewer than twelve errors.')
p('Errors on isolated digits can arise from ambiguous strokes, atypical shapes, or preprocessing effects. A future analysis should examine these factors on a larger, separately collected test set. This run does not include an augmentation ablation, so any improvement cannot be attributed specifically to augmentation.')
page()
h('5. Conclusion and reproducibility')
p(f'On the balanced 300-image test subset, logistic regression achieved {metrics.loc["Logistic regression","accuracy"]*100:.2f}% accuracy and the CNN achieved {metrics.loc["CNN","accuracy"]*100:.2f}%. The experiment demonstrates a complete pipeline from dataset auditing to model comparison and saved inference models. Its main limitation for this assignment is that the images were taken from MNIST rather than collected by the student.')
p('Limitations and next steps','Heading2')
p('The conclusions are limited to ten digit classes, one random seed, 1,400 original training images, and a small test set. The validation split has no writer identifiers. MNIST inputs are clean and normalized, so performance on photographs, varied backgrounds, pen colors, or mathematical symbols is unmeasured. Model softmax scores are not calibrated confidence estimates.')
p('To meet the self-collection requirement, collect 200 original images per class, for example from 20 writers producing ten examples of each digit. Keep writers w01-w14 in training, w15-w17 in validation, and w18-w20 in testing. Use the included self-collected template to rerun both models, and replace these benchmark results with the new measurements. Adding mathematical symbols requires new labeled classes and retraining; the current model predicts only digits.')
p('Reproduction and delivered artifacts','Heading2')
p(f'The experiment used seed 42 and {json.loads((R/"environment.json").read_text())["device"]} execution. Baseline model selection took {run["baseline_seconds"]:.1f} seconds; CNN training and its learning-curve output took {run["cnn_seconds"]:.1f} seconds on this machine. Timing is hardware-dependent. Library versions are saved in results/environment.json and requirements.txt.')
p('Install dependencies with python -m pip install -r requirements.txt. The included dataset is ready to use; python prepare_data.py rebuilds it from the source archive if necessary. Run python run_notebook.py to regenerate all experiment outputs. Run python predict.py PATH_TO_IMAGE for inference with the saved CNN, or python demo.py for a local drawing interface. Run python make_report.py after training to regenerate this report.')
p('The package includes the executed notebook, 2,000 class-organized PNGs, provenance and split manifests, both saved models, evaluation CSVs, confusion matrices, learning curves, error examples, editable report text, and reproduction scripts.')
p('References and data attribution','Heading2')
p('[1] Keras documentation. MNIST digits classification dataset. https://keras.io/api/datasets/mnist/ . MNIST copyright: Yann LeCun and Corinna Cortes; derived from NIST datasets. The selected, inverted PNG subset is distributed under CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/ .')
p('[2] TensorFlow documentation. Load NumPy data. https://www.tensorflow.org/tutorials/load_data/numpy . Documents the MNIST NPZ download used by the reproduction script.')
p('[3] Computer Vision Course Assignment - Dataset & Project. Instructor-provided Assignment1.pdf. Requirements: self-collected data, preprocessing and augmentation, baseline/improved models, evaluation, and a 3-5 page report.')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d4dce5')); canvas.line(54,43,541,43)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#526174'))
    canvas.drawString(54,30,'HANDWRITTEN DIGIT RECOGNITION | MNIST benchmark')
    canvas.drawRightString(541,30,str(doc.page))
doc=SimpleDocTemplate(str(OUT/'assignment_report.pdf'),pagesize=(595,842),leftMargin=54,rightMargin=54,topMargin=44,bottomMargin=55,title='Handwritten Digit Recognition - Assignment 1',author='Computer Vision Project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'REPORT.md').write_text('# Handwritten Digit Recognition\n\n'+'\n'.join(md),encoding='utf-8')
print('Created report PDF and editable Markdown.')
