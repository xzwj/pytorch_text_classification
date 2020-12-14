import torch
import torch.nn as nn

class DynamicRNN(nn.Module):
    """RNN which can hold variable length sequence, use like TensorFlow's RNN (input, length...). 

    :param input_size: The number of expected features in the input x
    :param hidden_size: The number of features in the hidden state h
    :param num_layers: Number of recurrent layers.
    :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
    :param bidirectional: If True, becomes a bidirectional RNN. Default: False
    :param rnn_type: {LSTM, GRU, RNN}
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    

    def forward(self, x, x_len):
        """sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        
        If batch_first==True, then `batch_size` is the 1st dimension at following parameters.

        :param x: pre-padded input sequence, size (seq_len, batch_size, feature_dim)
        :param x_len: indicating corresponding actual sequence length, size (batch_size)

        :return out: tensor containing the output features (h_t) from the last layer of the LSTM, for each t. 
                    size (seq_len, batch_size, num_directions * hidden_size)
        :return ht: tensor containing the hidden state for t = seq_len. 
                    size (num_layers * num_directions, batch_size, hidden_size)
        :return ct: tensor containing the cell state for t = seq_len. 
                    size (num_layers * num_directions, batch_size, hidden_size)
        """
        total_length = x.size(1) if self.batch_first else x.size(0) # the max length of sequences

        # ========= sort =========
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx] if self.batch_first else x[:, x_sort_idx]

        # ========= pack ========= 
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # ======== process ======== 
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        
        # ======== unsort ======== 
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first, total_length=total_length) # total_length is useful in DataParallel
            out = out[x_unsort_idx] if self.batch_first else out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)
