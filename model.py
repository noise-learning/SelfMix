import torch
import torch.nn as nn
from transformers import BertModel


class Bert4Classify(nn.Module):
    def __init__(self, args):
        super(Bert4Classify, self).__init__()

        self.encoder = BertModel.from_pretrained(args.bert_type)

        self.classifier1 = torch.nn.Linear(768, 768)
        self.classifier2 = torch.nn.Linear(768, args.num_class)
        self.dropout = torch.nn.Dropout(args.dropout_rate)

    def forward(self, input_ids, att_mask, is_training=True):

        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:,:max_len]
        att_mask = att_mask[:,:max_len]

        all_hidden = self.encoder(input_ids=input_ids,attention_mask=att_mask)

        emb = all_hidden[0][:,0,:]

        logits = self.classifier1(emb)
        logits = torch.nn.Tanh()(logits)

        if is_training:
            logits = self.dropout(logits)

        logits = self.classifier2(logits)

        return logits
    
    def emb(self, input_ids, att_mask, is_training=True):

        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:,:max_len]
        att_mask = att_mask[:,:max_len]

        bert = self.encoder(input_ids,att_mask)
        emb= bert[0][:,0,:]

        return emb

    def classify(self, x, is_training=True):

        logits = self.classifier1(x)
        logits = torch.nn.Tanh()(logits)

        if is_training:
            logits = self.dropout(logits)

        logits = self.classifier2(logits)

        return logits
