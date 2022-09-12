'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from inspect import trace
import torch
import torch.nn.functional as F

def get_saliency(model, optimizer, input, target):
    batch_size = target.shape[0]

    # Saliency
    for key in input.keys():
        if input[key] is not None and len(input[key].size()) < 2:
            input[key] = input[key].unsqueeze(0)
    model.train()

    word_emb, emb = model.module.emb(input['input_ids'], input['attention_mask'], trace_grad=True)
    logit = model.module.classify(emb, is_training=True)

    optimizer.zero_grad()
    loss = F.cross_entropy(logit, target)
    loss.backward()

    unary = torch.sqrt(torch.mean(word_emb.grad ** 2, dim=2))
    unary = unary / unary.view(batch_size, -1).max(1)[0].view(batch_size, 1)
    return unary, word_emb, target
