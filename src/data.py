import random
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import pandas as pd
import torch

class Ply_Dataset(Dataset):
    def __init__(self, split_path, dataset_type, tokenzier, context_length, split, title_tokenizer, title_vocab, song_vocab, shuffle):
        self.split_path = split_path
        self.dataset_type = dataset_type
        self.tokenzier = tokenzier
        self.context_length = context_length
        self.split = split
        self.title_tokenizer = title_tokenizer
        self.title_vocab = title_vocab.token_to_idx
        self.song_vocab = song_vocab.token_to_idx
        self.shuffle = shuffle
        self.get_fl()
    def get_fl(self):
        if self.split == "TRAIN":
            if self.shuffle:
                origin = torch.load(os.path.join(self.split_path, self.dataset_type, "train.pt"))
                shuffle_aug = []
                for instance in origin:
                    shuffle_instance = instance.copy()
                    song_list = list(shuffle_instance['songs'])
                    random.shuffle(song_list)
                    shuffle_instance['songs'] = song_list
                    shuffle_aug.append(shuffle_instance)
                aug_data = origin + shuffle_aug
                random.shuffle(aug_data)
                self.fl = aug_data
            else:
                self.fl = torch.load(os.path.join(self.split_path, self.dataset_type, "train.pt"))
        elif self.split == "VALID":
            self.fl = torch.load(os.path.join(self.split_path, self.dataset_type, "val.pt"))
        elif self.split == "TEST":
            self.fl = torch.load(os.path.join(self.split_path, self.dataset_type, "test.pt"))
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")

    def __getitem__(self, index):
        instance = self.fl[index]
        pid = instance['pid']
        plylst_title = instance['nrm_plylst_title']
        playlist_song = instance['songs']
        song_seq = self._song_tokenize(playlist_song, self.dataset_type,  context_length=self.context_length)
        token_seq = self._title_tokenize(plylst_title, context_length=self.context_length, tokenzier=self.tokenzier)
        return song_seq, token_seq

    def _title_tokenize(self, text, context_length, tokenzier):
        if tokenzier == "bpe":
            token = self.title_tokenizer.encode(text) 
            all_tokens = [1] + token + [2]
        else:
            token = ["<sos>"] + text.split() + ["<eos>"]
            all_tokens = [self.title_vocab[i] for i in token]
        text = torch.zeros(context_length, dtype=torch.long)
        if len(all_tokens) < context_length:
            text[:len(all_tokens)] = torch.tensor(all_tokens)
        else:
            text[:context_length-1] = torch.tensor(all_tokens[:context_length-1])
            text[-1] = all_tokens[-1]
        return text
    
    def _song_tokenize(self, song, dataset_type, context_length):
        song_token = ["<sos>"] + song + ["<eos>"]
        all_tokens = [self.song_vocab[i] for i in song_token]
        song_seq = torch.zeros(context_length, dtype=torch.long)
        if len(all_tokens) < context_length:
            song_seq[:len(all_tokens)] = torch.tensor(all_tokens)
        else:
            song_seq[:context_length-1] = torch.tensor(all_tokens[:context_length-1])
            song_seq[-1] = all_tokens[-1]
        return song_seq
    
    def __len__(self):
        return len(self.fl)