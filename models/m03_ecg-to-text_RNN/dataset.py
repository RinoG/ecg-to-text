import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import unicodedata
import wfdb

class Lang:
    def __init__(self):
        self.word2index = {"<sos>": 0, "<eos>": 1}
        self.word2count = {}
        self.index2word = {0: "<sos>", 1: "<eos>"}
        self.n_words = 2  # Count SOS and EOS
        self.max_len = 0

    def addSentence(self, sentence):
        sentences = sentence.split(' ')
        for word in sentences:
            self.addWord(word)

        n = len(sentences) + 2
        if n >= self.max_len:
            self.max_len = n + 2

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(reports, reverse=False):
    sentences = [normalizeString(s)for s in reports]

    output_lang = Lang()
    return output_lang, sentences

def prepareData(reports):
    output_lang, sentences = readLangs(reports)
    for s in sentences:
        output_lang.addSentence(s)
    return output_lang, sentences

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.word2index['<eos>'])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_signals(file_path, data):
    ecg_signal_path = data["filename_hr"]
    return [wfdb.rdrecord(file_path + '/' + path).p_signal for path in ecg_signal_path]

def get_dataloader(file_path, mode, batch_size, device):
    data = pd.read_csv(file_path+f'/{mode}.csv', sep=',')
    # signal_files = [f"{file_path}/signal_{i + 1}.csv" for i in range(10)]
    # signals = np.array([pd.read_csv(f).values for f in signal_files])
    signals = get_signals(file_path, data)

    output_lang, sentences = prepareData(data['preprocessed_report'])

    n = len(sentences)
    target_ids = np.zeros((n, output_lang.max_len), dtype=np.int32)

    for idx, tgt in enumerate(sentences):
        tgt_ids = indexesFromSentence(output_lang, tgt)
        tgt_ids.append(output_lang.word2index['<eos>'])
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.FloatTensor(signals).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return output_lang, train_dataloader



if __name__ == '__main__':
    import os
    os.chdir('../../')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang, dataloader = get_dataloader('data_ptb-xl', 'train', 2, device)

    for signal, report in dataloader:
        print(report)
