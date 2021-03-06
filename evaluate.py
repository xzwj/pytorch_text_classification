"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
# from torch.autograd import Variable

import utils
import models.nets as nets
import models.data_loaders as data_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lstm',
                    help="Choose model vrnn or lstm)")
parser.add_argument('--model_dir', default='experiments/lstm_200',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch in dataloader:
        # # move to GPU if available
        # if params.cuda:
        #     data_batch, labels_batch = data_batch.cuda(
        #         non_blocking=True), labels_batch.cuda(non_blocking=True)
        # # fetch the next evaluation batch
        # data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch.text) # size (batch_size, 1)
        output_batch = output_batch.squeeze(1) # size (batch_size)
        loss = loss_fn(output_batch, data_batch.label) # size (batch_size)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = data_batch.label.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available
    device = torch.device("cuda:0" if params.cuda else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(138)
    if params.cuda:
        torch.cuda.manual_seed(138)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # fetch dataloaders
    dl = data_loaders.IMDB_Dataloader(params.batch_size, device, './data')
    test_iter = dl.test_iterator
    embedding_matrix = dl.TEXT.vocab.vectors

    logging.info("- done.")

    # Define the model
    if args.model == 'lstm':
        model = nets.LSTM_(params, embedding_matrix).cuda() if params.cuda \
                                                            else nets.LSTM_(params, embedding_matrix)
    else:
        model = nets.Vanilla_RNN(params, embedding_matrix).cuda() if params.cuda \
                                                            else nets.Vanilla_RNN(params, embedding_matrix)

    loss_fn = nn.BCELoss()
    metrics = nets.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, dl.test_iterator, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)