"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/cifar10_vgg16_lr',
                    help='Directory containing params.json')
parser.add_argument('--dataset', default='cifar10',
                    help="Choose dataset (cifar10 or mnist)")
parser.add_argument('--model', default='vgg16',
                    help="Choose model (lenet5, vgg[11, 13, 16, 19], mlp, or softmax-reg)")


def launch_training_job(parent_dir, model, dataset, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        model_dir: (string) directory containing config, weights and log
        dataset: (string) dataset name
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model={model} --model_dir={model_dir} --dataset={dataset}".format(
            python=PYTHON, 
            model=model,
            model_dir=model_dir,
            dataset=dataset,
        )
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    learning_rates = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

    for learning_rate in learning_rates:
        # Modify the relevant parameter in params
        params.learning_rate = learning_rate

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.model, args.dataset, job_name, params)

