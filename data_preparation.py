import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.X = self.dataset.input
        self.y = self.dataset.label

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def read_data(pkl_path):
    pickle_in = open(pkl_path, "rb")
    movie_plots = pickle.load(pickle_in)
    return movie_plots


def clean_data(data_):
    movie_plots_lst = []
    for i in data_:
        if re.search(r'<.+}}', i):
            movie_plots_lst.append(i.replace(re.search(r'<.+}}', i).group(0), ''))
        else:
            movie_plots_lst.append(i)
    movie_plots = [re.sub("[^a-z' ]", "", i) for i in movie_plots_lst]
    return movie_plots


def create_seq(text, seq_len=8):
    sequences = []
    if len(text.split()) > seq_len:
        for i in range(seq_len, len(text.split())):
            seq = text.split()[i - seq_len:i + 1]
            sequences.append(" ".join(seq))
        return sequences
    else:
        return [text]


def process_and_save_data():
    data = read_data(pkl_path='plots_text.pickle')
    cleaned_data = clean_data(data)
    seqs = [create_seq(i) for i in cleaned_data]
    seqs = sum(seqs, [])
    inputs = []
    targets = []
    for s in seqs:
        inputs.append(" ".join(s.split()[:int(len(s.split())/2)]))
        targets.append(" ".join(s.split()[int(len(s.split())/2):]))

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
    train_dataset = pd.DataFrame(columns=['input', 'label'])
    test_dataset = train_dataset.copy()

    train_dataset['input'] = X_train
    test_dataset['input'] = X_test
    train_dataset['label'] = y_train
    test_dataset['label'] = y_test

    train_dataset.to_csv('data/train.csv', index=False)
    test_dataset.to_csv('data/test.csv', index=False)


def prepare_data_for_training():
    train_data = MovieDataset('data/train.csv')
    test_data = MovieDataset('data/test.csv')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    return train_loader, test_loader


if __name__ == '__main__':
    process_and_save_data()
