import os
import torch
from torch.utils.data import Dataset


class Vocab(object):
    def __init__(self, tags):
        self.word2id = {'<pad>': 0, '<unk>': 1}
        self.tags = tags
        self.tag2id = {tag: idx for idx, tag in enumerate(tags)}
        self.id2tag = {idx: tag for idx, tag in enumerate(tags)}
        self.data = []

    def load_file(self, data_dir):
        all_data = open(data_dir, 'r').read().strip().split('\n\n')
        for data in all_data:
            words = [line.split()[0] for line in data.splitlines()]
            tags = [line.split()[-1] for line in data.splitlines()]
            assert len(words) == len(tags)
            self.data.append((words, tags))

    def build_vocab(self):
        for words, tags in self.data:
            for word in words:
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
            # for tag in tags:
            #     if tag not in self.tag2id:
            #         self.tag2id[tag] = len(self.tag2id)


class InputExample(object):
    def __init__(self, words, tags=None):
        self.words = words
        self.tags = tags


class DataProcessor:

    def get_train_examples(self, train_dir):
        return self._process_data(train_dir)

    def get_dev_examples(self, dev_dir):
        return self._process_data(dev_dir)

    def get_test_examples(self, test_dir):
        return self._process_data(test_dir)

    def _process_data(self, data_dir):
        examples = []
        all_data = open(data_dir, 'r').read().strip().split('\n\n')
        for data in all_data:
            words = [line.split()[0] for line in data.splitlines()]
            tags = [line.split()[-1] for line in data.splitlines()]
            assert len(words) == len(tags)
            examples.append(InputExample(words=words, tags=tags))

        return examples


class NerDataset(Dataset):
    def __init__(self, examples, word2id, tag2id, max_len=512):
        self.examples = examples
        self.word2id = word2id
        self.tags2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        words = self.examples[idx].words
        tags = self.examples[idx].tags
        word_vector = []
        for word in words:
            if word in self.word2id:
                word_vector.append(self.word2id[word])
            else:
                word_vector.append(self.word2id['<unk>'])
        tag_vector = [self.tags2id[tag] for tag in tags]

        # word_vector = word_vector[:self.max_len] + [self.word2id['<pad>']] * (self.max_len - len(word_vector))
        # tag_vector = tag_vector[:self.max_len] + [self.tags2id['O']] * (self.max_len -len(tag_vector))
        # print(len(tag_vector))
        # data_sample = {'feature': torch.tensor(word_vector), 'tag': torch.tensor(tag_vector), 'length': len(word_vector)}
        # return data_sample
        return torch.tensor(word_vector), torch.tensor(tag_vector), len(word_vector)
