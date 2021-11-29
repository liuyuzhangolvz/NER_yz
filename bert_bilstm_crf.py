import torch
import torch.nn as nn
import torch.nn.functional as F
from charbilstm import BiLSTM
from crf import CRF


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, batch_size, dropout, device, bert=None, config=None):
        super(Bert_BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tag_size = len(self.tag2id)
        self.device = device

        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.lstm = BiLSTM(vocab_size, tag2id, embedding_dim, hidden_dim, batch_size, dropout, device, config)
        self.crf = CRF(self.tag_size, tag2id, device)

    def neg_log_likelihood_loss(self, sentence, seq_length, mask, tags=None):
        # print('sentence size>>', sentence.shape)  # (batch_size, seq_len)
        last_encoder_layer = self.bert(sentence).last_hidden_state
        # print('last_encoder_layer size>>', last_encoder_layer.shape)
        last_encoder_layer = self.dropout(last_encoder_layer)  # (batch_size, seq_len, config.hidden)
        # print('last_encoder_layer size>>', last_encoder_layer.shape)

        lstm_outs = self.lstm.get_output_score(last_encoder_layer, seq_length)  # (batch_size, seq_len, tag_size)
        # print('lstm_outs size >>', lstm_outs.shape)
        total_loss = self.crf.neg_log_likelihood_loss(lstm_outs, mask, tags)
        # print('total_loss>>', total_loss)
        _, tag_seq = self.crf._viterbi_decode(lstm_outs, seq_length, mask)
        # print('tag_seq>>', tag_seq)
        return total_loss, tag_seq

    def forward(self, sentence, seq_length, mask):
        last_encoder_layer = self.bert(sentence).last_hidden_state
        last_encoder_layer = self.dropout(last_encoder_layer)

        lstm_out = self.lstm.get_output_score(last_encoder_layer, seq_length)
        scores, tag_seq = self.crf._viterbi_decode(lstm_out, seq_length, mask)
        return tag_seq
        