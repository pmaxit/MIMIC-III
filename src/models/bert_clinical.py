import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig


MASK_VALUE  = -1e-18

class BertClinical(nn.Module):
    def __init__(self, config, num_labels=19):
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_output_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input):
        input_ids = input[0]
        attention_mask = input[1]
        _, pooled_output = self.bert(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class BertPoolClinical(nn.Module):
    def __init__(self, config, num_labels=19):
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_output_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input):
        input_ids = input[0]
        attention_mask = input[1]
        outputs = self.bert(input_ids, attention_mask, output_all_encoded_layers=False )

        # Sequence output  = outputs[0]
        # pooled output = outputs[1]
        hidden_states = outputs[2]       # hidden states , 12 layers

        h12 = hidden_states[-1][:,0].reshape((-1,1,768))
        h11 = hidden_states[-2][:,0].reshape((-1,1,768))
        h10 = hidden_states[-3][:,0].reshape((-1,1,768))
        h9  = hidden_states[-4][:,0].reshape((-1,1,768))

        all_h = torch.cat([h9, h10, h11, h12],1)    # also don't forget to add the last CLS seq_op / pooled_op
        mean_pool = torch.mean(all_h, 1)

        pooled_output = self.dropout(mean_pool)
        logits = self.classifier(pooled_output)
        outputs = (logits, ) + outputs[2:]          # add hidden states and attention if they are here

        return outputs
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True