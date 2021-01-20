import random
import numpy as np
import hparams as hp
import torch
import torch.utils.data
import torch.nn.functional as F
import os
import pickle as pkl
from text import text_to_sequence


def load_filepaths_and_text(metadata, split="|"):
    with open(metadata, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


class TextMelSet(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hp):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.data_type=hp.data_type
        self.seq_list=[]
        self.mel_list=[]
        for f in self.audiopaths_and_text:
            file_name = f[0][:10]
            seq_path = os.path.join(hp.data_path, self.data_type)
            mel_path = os.path.join(hp.data_path, 'melspectrogram')
            self.seq_list.append(torch.from_numpy(np.load(f'{seq_path}/{file_name}_sequence.npy')))
            self.mel_list.append(torch.from_numpy(np.load(f'{mel_path}/{file_name}_melspectrogram.npy')))

    def __getitem__(self, index):
        return (self.seq_list[index], self.mel_list[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths=torch.LongTensor([len(x[0]) for x in batch])
        max_input_len = input_lengths.max().item()

        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(batch)):
            text = batch[i][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad melspectrogram
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len%hp.downsample_ratio != 0:
            max_target_len = max_target_len - max_target_len%hp.downsample_ratio

        mel_padded = torch.zeros(len(batch), num_mels, max_target_len)
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(batch)):
            mel = batch[i][1]
            if mel.size(1)%hp.downsample_ratio!=0:
                mel=mel[:,:-(mel.size(1)%hp.downsample_ratio)]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        # Normalize
        mel_padded = (torch.clamp(mel_padded, hp.min_db, hp.max_db)-hp.min_db) / (hp.max_db-hp.min_db)
        
        return text_padded, input_lengths, mel_padded, output_lengths
