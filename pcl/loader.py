import os
import sys
import numpy as np 
import torch 
import random 

from utils import build_batch

import constant

# PAD_ID = 0
# UNK_ID = 1

from PIL import ImageFilter
import random
import torchvision.datasets as datasets


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index

class ContrastTrainDataSet(object):
    def __init__(self, all_utterances, word_dict, labels, name='train', add_noise=False):
        self.all_utterances = []

        self.augmented_utterances = []

        for i, dialogue in enumerate(all_utterances):
            self.all_utterances.extend(dialogue)
            
            for j, _ in enumerate(dialogue):
                session_positions = [k for k, label in enumerate(labels[i]) if label == labels[i, j]]
                if len(dialogue) <= 1:
                    # Select the first utterance in the next dialogue
                    self.augmented_utterances.append(all_utterances[(i + 1) % len(all_utterances)][0])
                elif len(session_positions) <= 1:
                    # Select the closest utterance regardless of the session
                    self.augmented_utterances.append(dialogue[(j - 1) if j > 0 else 1])
                elif j == session_positions[0]:
                    # Select the next utterance in the same session
                    self.augmented_utterances.append(dialogue[session_positions[1]])
                else:
                    # Select the previous utterance in the same session
                    self.augmented_utterances.append(dialogue[session_positions[j - 1]])


        # self.labels_batch = [labels[i:i+batch_size] \
        #                     for i in range(0, len(labels), batch_size)]
        self.word_dict = word_dict
        self.add_noise = add_noise
        # assert len(self.all_utterances_batch) == len(self.labels_batch)
        # self.batch_num = len(self.all_utterances_batch)
        # print("{} batches created in {} set.".format(self.batch_num, name))

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, index):
        # if not isinstance(key, int):
        #     raise TypeError
        # if key < 0 or key >= self.batch_num:
        #     raise IndexError
        if index < 0 or index >= len(self.all_utterances):
            raise ValueError("The index is not valid")

        utterance = self.all_utterances[index]
        augmented_utterance = self.augmented_utterances[index]
        # labels = self.labels_batch[key]

        # new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, session_transpose_matrix, \
        #         state_transition_matrix, session_sequence_length, max_conversation_length, loss_mask \
        #                     = build_batch(utterances, labels, self.word_dict, add_noise=self.add_noise)

        utterance = self.convert_to_id(utterance, self.word_dict)
        augmented_utterance = self.convert_to_id(augmented_utterance, self.word_dict)

        converted_index = torch.Tensor(index)

        sample = {"utterance": utterance, "augmented_utterance": augmented_utterance, "index": converted_index}

        # if self.add_noise:
        #     _, label_for_loss, _, _, _, _, _, _, _ = build_batch(utterances, labels, self.word_dict)

        # batch_size, max_length_1, max_length_2 = new_utterance_num_numpy.shape
        # new_utterance_num_numpy = self.convert_to_tensors_1(new_utterance_num_numpy, batch_size, \
        #                                                     max_length_1, max_length_2)
        # batch_size, max_length_1 = loss_mask.shape
        # loss_mask = self.convert_to_tensors_2(loss_mask, batch_size, max_length_1)
        # batch_size, max_length_1 = new_utterance_sequence_length.shape
        # new_utterance_sequence_length = self.convert_to_tensors_2(new_utterance_sequence_length, batch_size, max_length_1)
        # return new_utterance_num_numpy, label_for_loss, new_labels, new_utterance_sequence_length, \
        #             session_transpose_matrix, state_transition_matrix, session_sequence_length, \
        #                 max_conversation_length, loss_mask
        
        # return utterance, converted_index
        return sample

    def convert_to_id(self, utterance, word_dict):
        sequence = [word_dict.get(token, constant.UNK_ID) for token in utterance]
        return torch.Tensor(sequence)

    def convert_to_tensors_1(self, utterances, batch_size, max_length, h_size):
        # batch_size, max_conversation_length, max_utterance_length
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length, h_size).fill_(constant.PAD_ID)
        for i in range(len(utterances)):
            for j in range(len(utterances[i])):
                new_batch[i, j] = torch.LongTensor(utterances[i][j])
        return new_batch

    def convert_to_tensors_2(self, batch, batch_size, max_length):
        if not torch.cuda.is_available():
            new_batch = torch.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        else:
            new_batch = torch.cuda.LongTensor(batch_size, max_length).fill_(constant.PAD_ID)
        for i in range(len(batch)):
            new_batch[i] = torch.LongTensor(batch[i])
        return new_batch



def collate_fn_nlg_turn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    
    # augment negative samples
    if "neg_resp_idx_arr" in item_info.keys():
        neg_resp_idx_arr = []
        for arr in item_info['neg_resp_idx_arr']:
            neg_resp_idx_arr += arr
        
        # remove neg samples that are the same as one of the gold responses
        #print('item_info["response"]', item_info["response"])
        #print('neg_resp_idx_arr', neg_resp_idx_arr)
        
        for bi, arr in enumerate(item_info['neg_resp_arr']):
            for ri, neg_resp in enumerate(arr):
                if neg_resp not in item_info["response_plain"]:
                    item_info["response"] += [item_info['neg_resp_idx_arr'][bi][ri]]

    # merge sequences    
    # context, context_lengths = merge(item_info['context'])
    # context_delex, context_delex_lengths = merge(item_info['context_delex'])
    # response, response_lengths = merge(item_info["response"])
    # response_delex, response_delex_lengths = merge(item_info["response_delex"])
    # utterance, utterance_lengths = merge(item_info["utterance"])
    # utterance_delex, utterance_delex_lengths = merge(item_info["utterance_delex"])
    utterances = merge(item_info["utterance"])
    augmented_utterances = merge(item_info["augmented_utterances"])
    
    #print("context", context.size())
    #print("response", response.size())
    
    # item_info["context"] = to_cuda(context)
    # item_info["context_lengths"] = context_lengths
    # item_info["response"] = to_cuda(response)
    # item_info["response_lengths"] = response_lengths
    # item_info["utterance"] = to_cuda(utterance)
    # item_info["utterance_lengths"] = response_lengths

    utterances = to_cuda(utterances)
    augmented_utterances = to_cuda(augmented_utterances)
    indexes = to_cuda(item_info["index"])
    
    return [utterances, augmented_utterances], indexes

def merge(sequences, ignore_idx=None):
    '''
    merge from batch * sent_len to batch * max_len 
    '''
    pad_token = constant.PAD_ID if type(ignore_idx)==type(None) else ignore_idx
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths)==0 else max(lengths)
    padded_seqs = torch.ones(len(sequences), max_len).long() * pad_token 
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
    return padded_seqs, lengths

def to_cuda(x):
    if torch.cuda.is_available(): x = x.cuda()
    return x
