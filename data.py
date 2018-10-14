import torch
import numpy as np
import os
from torch.utils.data import Dataset
from collections import Counter
import random
from config import config

unk_token = '<UNK>'
pad_token = '<pad>'
# sos_token = '<s>'  # start of sentence
eos_token = '</s>'  # end of sentence


class Vocabulary():

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.__vocab_size = 0
        self.add_word(pad_token)
        self.add_word(unk_token)
        # self.add_word(sos_token)
        self.add_word(eos_token)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.__vocab_size
            self.__vocab_size += 1

    def __len__(self):
        return self.__vocab_size

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[unk_token]

    def get_word(self, idx):
        return self.idx2word[idx]


class CorpusDataset(Dataset):
    def __init__(self, corpus_ids):  # corpus is an instance of Corpus
        self.corpus_ids = corpus_ids

    def __len__(self):
        return len(self.corpus_ids)

    def __getitem__(self, idx):
        return self.corpus_ids[idx]


class Corpus():
    def __init__(self, data_dir):
        self.corpus = []
        self.corpus_ids = []
        for file in os.listdir(data_dir):
            print('Loading ', file)
            with open(os.path.join(data_dir, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    tokens = line.split()
                    tokens = tokens + [eos_token]  # add sos and eos token
                    self.corpus.append(tokens)

    def tokenize(self, vocab):  # convert corpus into idx
        self.corpus_ids = []
        for tokens in self.corpus:
            sent_ids = []
            for token in tokens:
                sent_ids.append(vocab.get_index(token))
            self.corpus_ids.append(sent_ids)
        return self.corpus_ids

    @property
    def all_tokens(self):
        return [token for tokens in self.corpus for token in tokens]

    def sort_corpus_ids(self): # deprecated
        self.corpus_ids = sorted(self.corpus_ids, key=lambda i: len(i), reverse=True)

    def shuffle_corpus_ids(self):  # deprecated
        self.corpus_ids = random.shuffle(self.corpus_ids)

    def __len__(self):
        return len(self.corpus_ids)


def build_vocab(all_tokens, vocab_size):  # corpus is a list of list of tokens
    vocab = Vocabulary()
    cnt_tokens = Counter(all_tokens).most_common(vocab_size - len(vocab))
    for i in cnt_tokens:
        vocab.add_word(i[0])
    return vocab


def collate_fn(batch):  # split the data into X and Y
    lengths = np.array([len(sent) for sent in batch])
    sorted_idx = np.argsort(-lengths)
    lengths = lengths[sorted_idx]  # descend order
    max_len = min(lengths[0],config.MAX_SENT)  # truncate
    batch_size = len(batch)
    X, Y = torch.zeros((batch_size, max_len - 1), dtype=torch.long), torch.zeros((batch_size, max_len - 1),
                                                                                 dtype=torch.long)

    for i, idx in enumerate(sorted_idx):
        length=min(lengths[i],max_len)
        X[i][:length - 1] = torch.LongTensor(batch[idx][:length - 1])
        Y[i][:length - 1] = torch.LongTensor(batch[idx][1:length])

    if config.use_cuda:
        X = X.cuda()
        Y = Y.cuda()

    return X, Y


if __name__ == '__main__':
    pass
