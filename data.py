#encoding=utf-8

import os
import sys

import struct
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SOS = 0
EOS = 1
UNK = 2

def read_label(label_file):
    vocab = Vocab()
    with open(label_file, 'r', encoding='gbk') as f:
        for line in f.readlines():
            sen = line.strip().split(' ', 1)[1]
            vocab.addSentence(sen)
    return vocab

def read_feat(path_pos):
    path, pos = path_pos.strip().split(':')

    ark_read_buffer = open(path, 'rb')
    ark_read_buffer.seek(int(pos),0)
    header = struct.unpack('<xcccc', ark_read_buffer.read(5))

    #if header[0] != b"B":
    #    print("Input .ark file is not binary")
    #    sys.exit(1)

    rows = 0; cols= 0
    m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
    n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

    tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
    utt_mat = np.reshape(tmp_mat, (rows, cols))

    ark_read_buffer.close()
    return utt_mat

class Vocab(object):
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class SpeechDataset(Dataset):
    def __init__(self, data_dir, dataset="train", vocab=None, lab_len=40):
        self.dataset = dataset
        self.vocab = vocab
        self.lab_len = lab_len
        
        scp_path = os.path.join(data_dir, dataset+".scp")
        label_file = os.path.join(data_dir, dataset+".text")
        
        self.process_feature_label(scp_path, label_file)

    def process_feature_label(self, scp_path, label_file):
        #read the path
        path_dict = {}
        with open(scp_path) as fin:
            for line in fin.readlines():
                utt, path = line.strip().split(' ')
                path_dict[utt] = path
        
        #read the label
        label_dict = dict()
        with open(label_file, encoding='gbk') as fin:
            for line in fin.readlines():
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else UNK for c in label.split()]
                label_dict[utt].append(EOS)
                if self.dataset != "train":
                    label_dict[utt].extend([0] * (self.lab_len-len(label_dict[utt])))

        assert len(label_dict) == len(path_dict)
        
        self.item = []
        for utt in label_dict:
            self.item.append((path_dict[utt], label_dict[utt]))
    
    def __getitem__(self, idx):
        path, label = self.item[idx]
        feat = read_feat(path)
        return (torch.from_numpy(feat), torch.LongTensor(label))

    def __len__(self):
        return len(self.item)

def EncoderInput(batch):
    inputs_max_length = max(x[0].size(0) for x in batch)
    feat_size = batch[0][0].size(1)
    
    targets_max_length = max(x[1].size(0) for x in batch)
    batch_size = len(batch)

    inputs = torch.zeros(batch_size, inputs_max_length, feat_size)
    targets = torch.zeros(batch_size, targets_max_length)

    for x in range(batch_size):
        feature, label = batch[x]
        feature_length = feature.size(0)
        label_length = label.size(0)
        
        inputs[x].narrow(0, 0, feature_length).copy_(feature)
        targets[x].narrow(0, 0, label_length).copy_(label)
    targets = targets.long()
    return inputs, targets

class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = EncoderInput

if __name__ == "__main__":
    train_label = '../speech_100h_data/train.text'
    vocab = read_label(train_label)
    train_dataset = SpeechDataset('../speech_100h_data', dataset="train", vocab=vocab)
    #print(train_dataset[0][1])
    
    train_loader = SpeechDataLoader(train_dataset, batch_size=10, shuffle=False)
    
    i = 0
    for data in train_loader:
        inputs, targets = data
        #print(inputs.size(), targets.size())
        print(type(inputs))
        if i == 5:
            break
        i += 1

