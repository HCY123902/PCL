import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 

import random
import numpy as np 

class UtteranceEncoder(nn.Module):
    def __init__(self, word_dict, word_emb=None, bidirectional=False, n_layers=1, input_dropout=0, \
                        dropout=0, rnn_cell='lstm', mode='contrast', args={}):
        super(UtteranceEncoder, self).__init__()
        # Added
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.mode = args.mode

        self.word_emb = word_emb
        self.word_emb_matrix = nn.Embedding(len(word_dict), self.embedding_size)
        self.init_embedding()
        # self.bidirectional = bidirectional
        # self.n_layers = n_layers
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.bidirectional = bidirectional
        
        # bi = 2 if self.bidirectional else 1
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(self.embedding_size, self.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError("The architecture needs to be either lstm or gru")

        if self.bidirectional:
            self.bidirectional_projection = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size), 
                nn.ReLU()
            )

        self.mode = mode

    def init_embedding(self):
        if self.word_emb is None:
            self.word_emb_matrix.weight.data.uniform_(-0.1, 0.1)
        else:
            self.word_emb_matrix.weight.data.copy_(torch.from_numpy(self.word_emb))
    
    def forward(self, input_var, input_lens):
        if self.mode == "disentanglement":
            shape = input_var.size() # batch_size, max_conversation_length, max_utterance_length
            input_var = input_var.view(-1, shape[2])
            input_lens = input_lens.reshape(-1)
            embeded_input = self.word_emb_matrix(input_var)
            word_output, _ = self.encoder(embeded_input)
            # word_output: [batch_size * max_conversation_length, max_utterance_length, hidden_size]
            if self.bidirectional:
                word_output = self.bidirectional_projection(word_output)
            return word_output, shape
        else:
            shape = input_var.size() # batch_size, max_utterance_length
            input_var = input_var.view(-1, shape[1])
            input_lens = input_lens.reshape(-1)
            embeded_input = self.word_emb_matrix(input_var)
            word_output, _ = self.encoder(embeded_input)
            # word_output: [batch_size, max_utterance_length, hidden_size]
            if self.bidirectional:
                word_output = self.bidirectional_projection(word_output)
            return word_output, shape

class ConversationEncoder(nn.Module):
    def __init__(self, bidirectional=False, n_layers=1, dropout=0, rnn_cell='lstm'):
        super(ConversationEncoder, self).__init__()
        self.bidirectional = bidirectional
        if rnn_cell == 'lstm':
            self.encoder = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        elif rnn_cell == 'gru':
            self.encoder = nn.GRU(self.hidden_size, self.hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        if self.bidirectional:
            self.bidirectional_projection = nn.Sequential(
                nn.Linear(self.hidden_size*2, self.hidden_size), 
                nn.ReLU()
            )
    
    def forward(self, input_var):
        # input_var: [batch_size, max_conversation_length, hidden_size]
        conv_output, _ = self.encoder(input_var)
        # conv_output: [batch_size, max_conversation_length, hidden_size]
        if self.bidirectional:
            conv_output = self.bidirectional_projection(conv_output)
        return conv_output


