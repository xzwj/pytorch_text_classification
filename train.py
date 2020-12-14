"""Train the model"""

import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from torch.autograd import Variable

import models.nets as nets
import models.data_loaders as data_loaders
from evaluate import evaluate
import utils

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='mnist',
#                     help="Choose dataset (cifar10 or mnist)")
parser.add_argument('--model', default='lstm',
                    help="Choose model vrnn or lstm)")
parser.add_argument('--model_dir', default='experiments/lstm_200',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, data_iter, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iter: (BucketIterator) a torch.data.BucketIterator object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    Note:
        data_iter has 2 useful varibles: data_iter.text and data_iter.label.
        data_iter.text is a tuple (because when it is initialized, we set batch_first=True).
        data_iter.text[0] is a tensor, contains a padded minibatch, size (batch_size, max_length).
        data_iter.text[1] is a tensor, contains the lengths of each examples, size (batch_size).
        data_iter.label is a tensor, size (batch_size).
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # with tqdm(total=len(data_iter), ncols=80, disable=True) as t:
    with tqdm(disable=False) as t:
        # for i, (train_batch, labels_batch) in enumerate(data_iter):
        for i, train_batch in enumerate(data_iter):
            # move to GPU if available
            # if params.cuda:
            #     train_batch, labels_batch = train_batch.cuda(
            #         non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            # train_batch, labels_batch = Variable(
            #     train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch.text) # size (batch_size, 1)
            output_batch = output_batch.squeeze(1) # size (batch_size)
            loss = loss_fn(output_batch, train_batch.label) # size (batch_size)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = train_batch.label.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)



def train_and_evaluate(model, dl, optimizer, scheduler, loss_fn, metrics, params, 
                        model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, dl.train_iterator, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, dl.val_iterator, metrics, params)

        # update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


def main():
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if params.cuda else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dl = data_loaders.IMDB_Dataloader(params.batch_size, device, './data')
    train_iter = dl.train_iterator
    val_iter = dl.val_iterator
    test_iter = dl.test_iterator
    embedding_matrix = dl.TEXT.vocab.vectors

    logging.info("- done.")

    # Define model
    if args.model == 'lstm':
        model = nets.LSTM_(params, embedding_matrix).cuda() if params.cuda \
                                                            else nets.LSTM_(params, embedding_matrix)
    else:
        model = nets.Vanilla_RNN(params, embedding_matrix).cuda() if params.cuda \
                                                            else nets.Vanilla_RNN(params, embedding_matrix)
    
    # Define optimizer
    if params.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, 
                                weight_decay=(params.wd if params.dict.get('wd') is not None else 0.0))
    elif params.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params.lr, 
                                weight_decay=(params.wd if params.dict.get('wd') is not None else 0.0))
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, 
                                weight_decay=(params.wd if params.dict.get('wd') is not None else 0.0))
    
    if params.dict.get('lr_adjust') is not None:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=params.lr_adjust, gamma=0.1)
    else:
        scheduler = None

    # fetch loss function and metrics
    loss_fn = nn.BCELoss()
    metrics = nets.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, dl, optimizer, scheduler, loss_fn, metrics, params, 
                        args.model_dir, args.restore_file)




if __name__ == '__main__':
    main()
