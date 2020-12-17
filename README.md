# CS386-ICME2019GC
SJTU CS386 project on ASD saliency map prediction. Currently model is a ResNet-based autoencoder. The model can be replaced by any network matching input and output.

## Note
Data is split into train/val/test as 240:30:30. e.g. The path to training data may be "/ROOT_PATH/data/Images/train/x.png".

## Train
```
python train.py [--cuda] [--root_path <ROOT_PATH>]  [--max_epochs <MAX_EPOCHS>] [--batch_size <BATCH_SIZE>]
```
Detailed arguments are in train.py, such as coefficients "alpha" of different losses.

## Evaluate
```
python eval.py [--cuda] [--root_path <ROOT_PATH>] <--model_path MODEL_PATH>
```

## Visualize
The image to visualize is set in visualize.py instead of using arguments.
```python visualize.py```

## TODO
[ ] Add pre-training.
[ ] Use fancy models.
[ ] Complete evaluation.

