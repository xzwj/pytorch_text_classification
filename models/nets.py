import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.dynamic_rnn import DynamicRNN


class LSTM_(nn.Module):
    def __init__(self, params, embedding_matrix):
        super(LSTM_, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix) # word embedding layer
        self.lstm = DynamicRNN(params.emb_dim, 
                            params.hidden_dim, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False, 
                            rnn_type='LSTM',
                            )
        self.logistic_regression = nn.Linear(params.hidden_dim, 1)

        self.dropout = nn.Dropout(params.dropout)if params.dict.get('dropout') is not None else None


    def forward(self, inputs):
        inputs_text, inputs_text_len = inputs # inputs_text: (batch_size, max_length), inputs_text_len: (batch_size)
        input_emb = self.emb(inputs_text) # embed words to vectors, size (batch_size, max_length, emb_dim)
        if self.dropout is not None:
            input_emb = self.dropout(input_emb)
        out, _ = self.lstm(input_emb, inputs_text_len) # size (batch_size, max_length, hidden_dim)
        feature = torch.mean(out, 1) # mean pooling of all outputs of the recurrent unit, size (batch_size, hidden_dim)
        output = torch.sigmoid(self.logistic_regression(feature)) # size (batch_size, 1)
        return output




class Vanilla_RNN(nn.Module):
    def __init__(self, params, embedding_matrix):
        super(Vanilla_RNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix) # word embedding layer
        self.rnn = DynamicRNN(params.emb_dim, 
                            params.hidden_dim, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False, 
                            rnn_type='RNN',
                            )
        self.logistic_regression = nn.Linear(params.hidden_dim, 1)

        self.dropout = nn.Dropout(params.dropout)if params.dict.get('dropout') is not None else None


    def forward(self, inputs):
        inputs_text, inputs_text_len = inputs # inputs_text: (batch_size, max_length), inputs_text_len: (batch_size)
        input_emb = self.emb(inputs_text) # embed words to vectors, size (batch_size, max_length, emb_dim)
        if self.dropout is not None:
            input_emb = self.dropout(input_emb)
        out, _ = self.rnn(input_emb, inputs_text_len) # size (batch_size, max_length, hidden_dim)
        feature = torch.mean(out, 1) # mean pooling of all outputs of the recurrent unit, size (batch_size, hidden_dim)
        output = torch.sigmoid(self.logistic_regression(feature)) # size (batch_size, 1)
        return output

        


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) (batch_size) - sigmoid output of the model
        labels: (np.ndarray) dimension (batch_size), where each element is a value in [0, 1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = (outputs > 0.5)
    return np.sum(outputs==labels) / float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}




# if __name__ == '__main__':
#     # Test for class `LeNet5`
#     import torch
#     import sys 
#     sys.path.append(".") 
#     import utils

#     params = utils.Params('./experiments/cifar10_lenet5/params.json')
#     model = LeNet5(params)
#     print(model)
#     x = torch.randn(2,3,32,32)
#     print(x)
#     y = model(x)
#     print(y)
#     print(y.size())
    


