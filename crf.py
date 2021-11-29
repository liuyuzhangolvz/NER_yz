import torch
import torch.autograd as autograd
import torch.nn as nn


def log_sum_exp(vec, tag_size):
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, tag_size)).view(-1, 1, tag_size)
    return max_score.view(-1, tag_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, tag_size)
    # max_score = vec[0, torch.argmax(vec).item()]
    # max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class CRF(nn.Module):
    def __init__(self, tag_size, tag2id, device):
        super(CRF, self).__init__()
        self.tag_size = tag_size
        self.tag2id = tag2id
        self.START_TAG = '<START>'
        self.STOP_TAG = '<STOP>'
        self.device = device
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        init_transitions = torch.zeros(tag_size, tag_size).to(device)

        self.transitions = nn.Parameter(init_transitions)
        # self.transitions.data[tag2id[self.START_TAG], :] = -10000
        # self.transitions.data[:, tag2id[self.STOP_TAG]] = - 10000

    def _viterbi_decode(self, feats, seq_length, mask):
        """
            input:
                feats: (batch_size, seq_len, self.tag_size + 2)
                mask: (batch, seq_len)
            output:
                path_score: (batch, 1) corresponding score for each sentence
                decode_idx: (batch, seq_len) decoded sequence
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        # calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()  # (batch_size, 1)
        mask = mask.transpose(1, 0).contiguous()  # (seq_len, batch_size)

        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # print('scores size>>', scores.shape)

        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()

        mask = (1-mask.long()).bool()
        _, inivalues = next(seq_iter)  # (batch_size, tag_size, tag_size)
        partition = inivalues[:, self.tag2id[self.START_TAG], :].clone().view(batch_size, tag_size)  # (batch_size, tag_size)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            #print('partition size>>', partition.shape)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        # print('partition_history>>', partition_history.shape)
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zeros = autograd.Variable(torch.zeros(batch_size, tag_size)).long().to(self.device)

        back_points.append(pad_zeros)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        # print('back_points size>>', back_points.shape)

        # select end ids in STOP_TAG
        pointer = last_bp[:, self.tag2id[self.STOP_TAG]]
        # print('pointer>>', pointer)
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()  # (batch_size, seq_len, tag_size)
        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()  # (seq_len, batch_size, tag_size)
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size)).to(self.device)

        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(batch_size).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        # print('path score>>', path_score)
        # print('decode_idx>>', decode_idx)
        return path_score, decode_idx

        # backpointers = []
        #
        # # Initialize the viterbi variable in log space
        # init_vvars = torch.full((1, self.tag_size), -10000)
        # init_vvars[0][self.tag2id[self.START_TAG]] = 0
        #
        # # forward_var at step i holds the viterbi variable for step i - 1
        # forward_var = init_vvars
        # for feat in feats:
        #     bptrs_t = []  # holds the backpointers for this step
        #     viterbivars_t = []  # holds the veterbi variables for the step
        #     for next_tag in range(self.tag_size):
        #         next_tag_var = forward_var + self.transitions[next_tag]
        #         best_tag_id = torch.argmax(next_tag_var).item()
        #         bptrs_t.append(best_tag_id)
        #         viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
        #     forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
        #     backpointers.append(bptrs_t)
        # # Transition to STOP_TAG
        # terminal_var = forward_var + self.transitions[self.tag2id[self.STOP_TAG]]
        # best_tag_id = torch.argmax(terminal_var).item()
        # path_score = terminal_var[0][best_tag_id]
        #
        # # follow the back pointers to decode the best path
        # best_path = [best_tag_id]
        # for bptrs_t in reversed(backpointers):
        #     best_tag_id = bptrs_t[best_tag_id]
        #     best_path.append(best_tag_id)
        # # Pop off the start tag
        # start = best_path.pop()
        # assert start == self.tag2id[self.START_TAG]
        # best_path.reverse()
        # return path_score, best_path

    def _calculate_PZ(self, feats, mask):
        """
        input:
            feats: (batch_size, seq_len, tag_size + 2)
            mask: (batch_size, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        mask = mask.transpose(1, 0).contiguous()
        # print('mask>>', mask.shape)
        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        partition = inivalues[:, self.tag2id[self.START_TAG], :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size).bool()
            masked_cur_partition = cur_partition.masked_select(mask_idx)

            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + \
                     partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.tag2id[self.STOP_TAG]]
        return final_partition.sum(), scores

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        for feat in feats:
            alpha_t = []
            for next_tag in range(self.tag_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alpha_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alpha_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2id[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, scores, mask, tags):
        """
        input:
            scores: (seq_len, batch_size, tag_size, tag_size)
            mask: (batch_size, seq_len)
            tags: (batch_size, seq_len)
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len)).to(self.device)

        for idx in range(seq_len):
            if idx == 0:
                # print('tags>>', tags.shape)
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]
        end_transition = self.transitions[:, self.tag2id[self.STOP_TAG]].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)

        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0).bool())

        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

        # score = torch.zeros(1)
        # tags = torch.cat([torch.tensor([self.tag2id[self.START_TAG]], dtype=torch.long), tags])
        # for i, feat in enumerate(feats):
        #     score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # score += score + self.transitions[self.tag2id[self.STOP_TAG], tags[-1]]
        # return score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score

    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path
