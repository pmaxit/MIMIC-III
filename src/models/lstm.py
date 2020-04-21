import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig

class BiLSTM(nn.Module):
    def __init__(self, V, E, C, h=50, bidirectional=True, dropout=0.5):
        """
            V | Vocab Size
            E | Embedding Size
            C | Number of classes
            H | hidden dimension

            -----------------------------
            B | Batch dimension
            L | Max Seqnelence length
            H | 2*h or h depending on bidrectional
        
        """
        super().__init__()
        self.embed = nn.Embedding(V, E)
        self.encoder = nn.LSTM(
            input_size = E, 
            hidden_size = h,
            bidirectional=bidirectional,
            batch_first = True
        )

        self.H = 2*h if bidirectional else h
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features = self.H , out_features = C )


    def forward(self, input):
        x = input[0]
        attention_mask = input[1]
        mask = x != 0                           # B X L
        x = self.embed(x)                       # B X L X E

        lengths =  attention_mask.sum(axis=1) #

        sent_packed = nn.utils.rnn.pack_padded_sequence(x, lengths,batch_first=True,enforce_sorted=False)
        sent_encoder, _ = self.encoder(sent_packed)                  # B X L X H
        output, _ = nn.utils.rnn.pad_packed_sequence(sent_encoder, batch_first=True)   # B X max_seq X H

        output = output.mean(dim=1)         # B X H
        # output layer
        t = self.dropout(output)
        logits = self.linear(t)

        return logits



class BiLSTMWithBertEmbedding(nn.Module):
    def __init__(self, config, V, E, C, h=50, bidirectional=True, dropout=0.5):
        """
            V | Vocab Size
            E | Embedding Size
            C | Number of classes
            H | hidden dimension

            -----------------------------
            B | Batch dimension
            L | Max Seqnelence length
            H | 2*h or h depending on bidrectional
        
        """
        super().__init__()
        self.embed = nn.Embedding(V, E)
        self.bert = BertModel(config)
        self.encoder = nn.LSTM(
            input_size = E, 
            hidden_size = h,
            bidirectional=bidirectional,
            batch_first = True
        )

        self.H = 2*h if bidirectional else h
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features = self.H , out_features = C )


    def forward(self, input):
        input_ids = input[0]
        attention_mask = input[1]
        mask = input_ids != 0                           # B X L

        # just replacing the embedding layers with bert layer
        import pdb
        pdb.set_trace()
        #x = self.embed(x)                       # B X L X E
        x = self.bert(input_ids, attention_mask )[0]


        lengths =  attention_mask.sum(axis=1) #

        sent_packed = nn.utils.rnn.pack_padded_sequence(x, lengths,batch_first=True,enforce_sorted=False)
        sent_encoder, _ = self.encoder(sent_packed)                  # B X L X H
        output, _ = nn.utils.rnn.pad_packed_sequence(sent_encoder, batch_first=True)   # B X max_seq X H

        output = output.mean(dim=1)         # B X H
        # output layer
        t = self.dropout(output)
        logits = self.linear(t)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True