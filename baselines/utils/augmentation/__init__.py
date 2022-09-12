'''
SSMix
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
__all__ = ['ssmix', 'unk']

from ast import arg
from . import *
import torch


class SSmixAugment:
    def __init__(self, args, model, optimizer):
        self.max_len = args.sentence_len
        assert self.max_len % 2 == 0, 'Invalid max_len: must be multiple of 2'
        self.augment = ssmix.SSMix(args, model, optimizer)

    def __call__(self, input1, target1, length1, input2, target2, length2):
        mix_input1, mix_ratio1 = self.augment(input1, input2,
                                                target1, target2,
                                                length1, length2, self.max_len)

        mix_input2, mix_ratio2 = self.augment(input2, input1,
                                                target2, target1,
                                                length2, length1, self.max_len)
        mix_input = {'input_ids': torch.cat((mix_input1['input_ids'], mix_input2['input_ids']), dim=0),
                     'attention_mask': torch.cat((mix_input1['attention_mask'], mix_input2['attention_mask']), dim=0)}
        mix_ratio = torch.cat((mix_ratio1, mix_ratio2), dim=0)
        return mix_input, mix_ratio

    def split_pairs(self, inputs, length):
        key_set = ['input_ids', 'attention_mask', 'token_type_ids']
        left, right = {key: inputs[key].clone().detach() for key in key_set}, \
                      {key: inputs[key].clone().detach() for key in key_set}

        left_length, right_length = list(), list()
        batch_size = len(inputs['input_ids'])
        for batch_idx in range(batch_size):
            mask_pair = torch.where(inputs['token_type_ids'][batch_idx] == 1)[0]
            pair_length = len(mask_pair)

            for key in key_set:
                left[key][batch_idx][mask_pair] = 0  # remove second pair info for left
                right[key][batch_idx][1:pair_length + 1] = \
                    right[key][batch_idx][mask_pair]  # move right text to first idx, with SOS token.
                right[key][batch_idx][pair_length + 1:] = 0  # remove unnecessary tokens

            right['token_type_ids'][batch_idx][0] = 1  # edge case for SOS token token_type_ids
            left_length.append(length[batch_idx].item() - pair_length)
            right_length.append(pair_length + 1)

        left_length, right_length = torch.tensor(left_length), torch.tensor(right_length)
        return left, right, left_length, right_length

    def merge_pairs(self, input1, input2):
        batch_size = input1['input_ids'].shape[0]

        key_set = ['input_ids', 'attention_mask', 'token_type_ids']

        # merged = input1 + input2, then return input2
        merged = {key: torch.zeros((batch_size, self.max_len),
                                   device=self.args.device, dtype=torch.long) for key in key_set}

        length_left, length_right = list(), list()
        for batch_idx in range(len(input1['input_ids'])):
            length1 = input1['attention_mask'][batch_idx].sum()
            length2 = input2['attention_mask'][batch_idx].sum()

            # Leave out the SOS token.
            merged['input_ids'][batch_idx][:length1 + 1] = input1['input_ids'][batch_idx][:length1 + 1]
            assert (length1 + length2) <= self.max_len

            merged['input_ids'][batch_idx][length1:(length1 + length2)] = input2['input_ids'][batch_idx][1:length2 + 1]
            merged['attention_mask'][batch_idx][:length1 + length2 - 1] = 1
            merged['token_type_ids'][batch_idx][length1:length1 + length2 - 1] = 1

            length1 -= 2
            length2 -= 2 # remove EOS, SOS
            length_left.append(length1)
            length_right.append(length2)

        return merged, torch.tensor(length_left), torch.tensor(length_right)
