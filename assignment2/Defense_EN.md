# Assignment 2 defense

## Speaker 1

In our first assignment, we trained a model to recognize handwritten digits. This time, we tested whether extra training images would help. We started with real MNIST digits and used TA-TiTok and MaskGen-L to make 120 variations with thinner, thicker or tilted strokes. In the demo, I can draw a seven and see both the prediction and a heatmap showing which strokes influenced it.

## Speaker 2

We tested all three models on the same 300 digits. Our original model got 295 right. After training with the extra images, it got 296 right, and macro-F1 rose from 98.35% to 98.68%. We also trained the model again using only the original data. That version got 297 right. So we saw a small improvement over our first model, but we cannot give the synthetic images all the credit.
