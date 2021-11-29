import torch
import torch.nn as nn
from charbilstm import BiLSTM
from crf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2id, embedding_dim, hidden_dim, batch_size, dropout, device, config=None, bert=None):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tag_size = len(self.tag2id)

        self.lstm = BiLSTM(vocab_size, tag2id, embedding_dim, hidden_dim, batch_size, dropout, device, config)
        self.crf = CRF(self.tag_size, tag2id, device)

    def neg_log_likelihood_loss(self, sentence, seq_length, mask, tags=None):
        embeds = self.lstm.char_embeddings(sentence)
        embeds_drop = self.lstm.char_dropout(embeds)
        lstm_outs = self.lstm.get_output_score(embeds_drop, seq_length)
        total_loss = self.crf.neg_log_likelihood_loss(lstm_outs, mask, tags)
        # print('total_loss>>', total_loss)
        _, tag_seq = self.crf._viterbi_decode(lstm_outs, seq_length, mask)
        # print('tag_seq>>', tag_seq)
        return total_loss, tag_seq

    def forward(self, sentence, seq_length, mask):
        embeds = self.lstm.char_embeddings(sentence)
        embeds_drop = self.lstm.char_dropout(embeds)
        lstm_out = self.lstm.get_output_score(embeds_drop, seq_length)
        scores, tag_seq = self.crf._viterbi_decode(lstm_out, seq_length, mask)
        return tag_seq
        