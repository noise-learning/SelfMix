import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertLayer, BertPreTrainedModel
from transformers import AutoModel

class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, input_ids2=None, attention_mask=None, attention_mask2=None, l=None, mix_layer=1000,  token_type_ids=None, position_ids=None, head_mask=None):

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:

            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=torch.float32)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=torch.float32)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)


        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]


        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class Bert4Classify(nn.Module):
    def __init__(self, args):
        super(Bert4Classify, self).__init__()

        if args.mix_option==0:
            self.encoder = AutoModel.from_pretrained(args.pretrain_model_dir + args.bert_type)
        elif args.mix_option==1:
            self.encoder = BertModel4Mix.from_pretrained('bert-base-uncased')

        self.classifier1 = torch.nn.Linear(768, 768)
        self.classifier2 = torch.nn.Linear(768, args.num_class)
        self.dropout = torch.nn.Dropout(args.dropout_rate)

    def forward(self, input_ids, att_mask, input_ids2=None, att_mask2=None, l=None, mix_layer=1000, is_training=True):

        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:,:max_len]
        att_mask = att_mask[:,:max_len]

        if input_ids2 is not None:
            max_len2 = att_mask2.sum(1).max()
            input_ids2 = input_ids2[:,:max_len2]
            att_mask2 = att_mask2[:,:max_len2]

            all_hidden, pooler = self.encoder(input_ids=input_ids, input_ids2=input_ids2, attention_mask=att_mask, attention_mask2=att_mask2, l=l, mix_layer=mix_layer)
            # pooled_output = torch.mean(all_hidden, 1)
            emb = all_hidden[:,0,:]
        else:
            all_hidden = self.encoder(input_ids=input_ids,attention_mask=att_mask)

            emb = all_hidden[0][:,0,:]

        
        logits = self.classifier1(emb)
        logits = torch.nn.Tanh()(logits)

        if is_training:
            logits = self.dropout(logits)

        logits = self.classifier2(logits)

        return logits
    
    def emb(self, input_ids, att_mask, is_training=True, trace_grad=False):

        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:,:max_len]
        att_mask = att_mask[:,:max_len]

        bert = self.encoder(input_ids,att_mask)
        if trace_grad:
            word_emb = bert[0]
            word_emb = word_emb.detach().requires_grad_(True)
            emb = word_emb[:,0,:]
            return word_emb, emb
        else:
            emb = bert[0][:,0,:]
            return emb

    def classify(self, x, is_training=True):

        logits = self.classifier1(x)
        logits = torch.nn.Tanh()(logits)

        if is_training:
            logits = self.dropout(logits)

        logits = self.classifier2(logits)

        return logits
