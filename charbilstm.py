import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, batch_size, dropout):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tag_size = len(self.tag2id)

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embeddings(vocab_size, embedding_dim)))

        self.char_dropout = nn.Dropout(dropout)
        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            batch_first=True,
                            num_layers=1,
                            bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def random_embeddings(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def get_last_hiddens(self, sentence, seq_length):
        """
        :param sentence: (batch_size, max_seq_length)
        :param seq_length: (batch_size, 1)
        :return: (batch_size, char_hidden_dim)
        """
        self.hidden = self.init_hidden()  # （2，32，128）
        # print('sentence size >>', sentence)
        embeds = self.char_embeddings(sentence)  # （32， 128， 100）
        embeds_drop = self.char_dropout(embeds)

        pack_input = pack_padded_sequence(embeds_drop, seq_length, batch_first=True, enforce_sorted=False)

        lstm_out, self.hidden = self.char_lstm(pack_input, self.hidden)  # （32， 128， 256）
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)  # （32， 128， 17）
        return lstm_feats

    def _get_lstm_feature(self, sentence, seq_length):
        '''
        :param sentence: (batch_size, max_seq_length)
        :param seq_length: (batch_size, 1)
        :return: (batch_size, max_seq_length, char_hidden_dim
        '''
        self.hidden = self.init_hidden()  # （2，32，128）
        # print('sentence size >>', sentence)
        embeds = self.char_embeddings(sentence)  # （32， 128， 100）
        embeds_drop = self.char_dropout(embeds)

        pack_input = pack_padded_sequence(embeds_drop, seq_length, batch_first=True, enforce_sorted=False)

        lstm_out, self.hidden = self.char_lstm(pack_input, self.hidden)  # （32， 128， 256）
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)   # （32， 128， 17）
        return self.hidden[0].transpose(1,0).contiguous().view(self.batch_size, -1)

    def forward(self, sentence, seq_length, tags=None):
        lstm_feats = self._get_lstm_feature(sentence, seq_length)
        # print('lstm_feats size >>', lstm_feats)
        logits = self.logsoftmax(lstm_feats)  # （32， 128， 17）
        return logits


