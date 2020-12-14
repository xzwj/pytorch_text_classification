# PyTorch Implementations for Text Classification

## Dataset
- IMDB

## Model
- Vanilla RNN
- LSTM

## Experiment Settings and Results
The word embedding vectors are taken from GloVe with dimension 300 and pretrained from the Wikipedia 2014 and Gigaword 5 corpus. Dropout is applied to word embedding layer. Cross entropy loss is used for every models. The first seven columns of Table 1 are their hyperparameter settings.


Table 1: Hyperparameter settings and experiment results. state_dim—state dimension, lr—initially learning rate, step_size—period of learning rate decay, bs—batch size, optim—optimizer, num_epoch—total number of epochs. The last column is the experiment result.

|     Model          |     state_dim    |     lr      |     step_size    |     bs    |     optim    |     dropout    |     Acc (%)    |
|--------------------|------------------|-------------|------------------|-----------|--------------|----------------|----------------|
|     Vanilla RNN    |     20           |     2e-2    |     20           |     32    |     SGD      |     0.2        |     83.8       |
|                    |     50           |     2e-2    |     20           |     32    |     SGD      |     0.2        |     85.5       |
|                    |     100          |     2e-2    |     20           |     32    |     SGD      |     0.2        |     **86.3**       |
|                    |     200          |     2e-2    |     20           |     32    |     SGD      |     0.2        |     85.8       |
|                    |     500          |     2e-2    |     20           |     32    |     SGD      |     0.2        |     85.2       |
|     LSTM           |     20           |     1e-3    |     -            |     32    |     Adam     |     0.4        |     87.9       |
|                    |     50           |     1e-3    |     -            |     32    |     Adam     |     0.4        |     88.3       |
|                    |     100          |     1e-3    |     -            |     32    |     Adam     |     0.4        |     88.6       |
|                    |     200          |     1e-3    |     -            |     32    |     Adam     |     0.4        |     **88.8**       |
|                    |     500          |     1e-3    |     -            |     32    |     Adam     |     0.4        |     88.6       |





## Code Layout
```
.
├── evaluate.py
├── experiments/
├── data/
├── models/
│   ├── data_loaders.py
│   ├── dynamic_rnn.py
│   └── nets.py
├── requirements.txt
├── search_hyperparams.py
├── train.py
└── utils.py
```

-	train.py: contains main training loop
-	utils.py: utility functions
-	evaluate.py: contains main evaluation loop
-	data/: store preprocessed datasets
-	models/data_loaders.py: data loaders for each dataset
-	models/dynamic_rnn: RNN layer which can hold variable length sequence
-	models/nets.py: Vanilla RNN, LSTM and evaluation metrics
-	experiments/: store hyperparameters, model weight parameters, checkpoint, training log and evaluation log for each experiment

## Requirements
Create a conda environment and install requirements using pip:
```
>>> conda create -n ass3 python=3.7
>>> source activate ass3
>>> pip install -r requirements.txt
```

## How to Run
Train a model with the specified hyperparameters:
```
>>> python train.py --model {model name} --model_dir {hyperparameter directory}
```
Evaluate a trained model:
```
>>> python evaluate.py --model {model name} --model_dir {checkpoint directory} --restore_file {checkpoint name}
```

For example, train a LSTM model with the hyperparameters in experiments/lstm_500/params.json:
```
>>> python train.py --model lstm --model_dir experiments/lstm_500
```
It will automatically download the dataset and pretrained word embedding if they are not downloaded. After preprocessing, the preprocessed data will be saved in “data” directory. During the training loop, best model weight parameters, last model weight parameters, checkpoint, and training log will be saved in experiments/lstm_500.

Evaluate a LSTM model with the checkpoint file “experiments/lstm_500/best.pth.tar”:
```
>>> python evaluate.py --model lstm --model_dir experiments/lstm_500 --restore_file best
```
The evaluation results will be saved in experiments/lstm_500.
