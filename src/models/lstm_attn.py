import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import create_emb_layer


MASK_VALUE  = -1e-18



class SelfAttentionLSTM(nn.Module):
    def __init__(self, V, E, C, h=50, bidirectional=True, dropout=0.5, weights_matrix= None):
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
        if type(weights_matrix) != type(None):
            print("Embedding matrix loaded ")
            self.embed,_,_ = create_emb_layer(weights_matrix)
        else:
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
        x, _ = self.encoder(x)                  # B X L X H

        # self-attention
        attn_scores = x.bmm(x.transpose(-1,-2))
        mask = mask.unsqueeze(-1).expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill(~mask, MASK_VALUE)
        attn = torch.softmax(attn_scores, dim=-1)

        x = x + attn.bmm(x)                     #  B X L X H
        x = x.mean(dim=1)                       # B X H

        # output layer
        x = self.dropout(x)
        logits = self.linear(x)

        return logits
