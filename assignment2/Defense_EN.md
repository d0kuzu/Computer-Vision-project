# Assignment 2 defense

## Speaker 1

We expanded our digit dataset to 3,000 images. It contains 2,000 original MNIST digits and 1,000 synthetic versions. We used TA-TiTok and MaskGen-L to vary the strokes of existing training digits. In the demo, I can draw a digit, see one prediction and open a heatmap showing which strokes influenced the answer.

## Speaker 2

We used the same 300 test digits for every model. Our original A1 model recognized 295. After fine-tuning with 1,000 synthetic examples, the new model recognized 296. Accuracy was 98.67% and macro-F1 was 98.67%. Repeating the original handwriting also gave 296 correct answers, so this test showed no accuracy advantage from the synthetic data. The demo uses that control model because it performed better on the separate validation set.
