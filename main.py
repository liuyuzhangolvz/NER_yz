import os
import random
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader import Vocab, DataProcessor, NerDataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from charbilstm import BiLSTM
from crf import CRF
from bilstm_crf import BiLSTM_CRF
from bert_bilstm_crf import Bert_BiLSTM_CRF
from pytorchtools import EarlyStopping

from transformers import BertConfig, BertTokenizer, BertModel

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score

from torch.nn.utils.rnn import pad_sequence

MODE_CLASS = {
    'bilstm': BiLSTM,
    'crf': CRF,
    'bilstm_crf': BiLSTM_CRF,
    'bert_bilstm_crf': Bert_BiLSTM_CRF
}

def collate_func(batch):
    batch_len = len(batch)
    max_seq_len = max([dic['length'] for dic in batch])
    mask_batch = torch.zeros((batch_len, max_seq_len))
    fea_batch = []
    tag_batch = []
    for i in range(batch_len):
        dic = batch[i]
        fea_batch.append(dic['feature'])
        tag_batch.append(dic['tag'])
        mask_batch[i: dic['length']] = 1
    res = {'feature': pad_sequence(fea_batch, batch_first=True),
           'tag': pad_sequence(tag_batch, batch_first=True),
           'mask': mask_batch}
    return res


def sort_batch(seqs, tags, lengths, mask):
    """
    Sort a mini-batch by the length of the sequence with the longest sequence first.
    :param seqs:
    :param tags:
    :param lengths:
    :return:
    """
    seq_length, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = seqs[perm_idx]
    tag_tensor = tags[perm_idx]
    mask_tensor = mask[perm_idx]
    return seq_tensor, tag_tensor, seq_length, mask_tensor


def collate_pad_and_sort(batch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    :param batch:
    :return:
    """
    batch_size = len(batch)
    batch_split = list(zip(*batch))

    seqs, tags, lengths = batch_split[0], batch_split[1], batch_split[2]
    max_seq_len = max(lengths)
    padded_seqs = np.zeros((batch_size, max_seq_len))
    padded_tags = np.zeros((batch_size, max_seq_len))
    mask = np.zeros((batch_size, max_seq_len))
    for i, length in enumerate(lengths):
        padded_seqs[i, 0: length] = seqs[i][0: length]
        padded_tags[i, 0: length] = tags[i][0: length]
        mask[i, 0: length] = 1
    return sort_batch(torch.tensor(padded_seqs, dtype=torch.int64),
                      torch.tensor(padded_tags, dtype=torch.int64),
                      torch.tensor(lengths, dtype=torch.int64),
                      torch.tensor(mask, dtype=torch.int64))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Bi-LSTM-CRF')
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--train_dataset', type=str, default='../../dataset/demo_data/demo.train.char')
    parser.add_argument('--dev_dataset', type=str, default='../../dataset/demo_data/demo.dev.char')
    parser.add_argument('--test_dataset', type=str, default='../../dataset/demo_data/demo.test.char')
    # parser.add_argument('--train_dataset', type=str, default='../../dataset/ResumeNER/train.char.bmes')
    # parser.add_argument('--dev_dataset', type=str, default='../../dataset/ResumeNER/dev.char.bmes')
    # parser.add_argument('--test_dataset', type=str, default='../../dataset/ResumeNER/test.char.bmes')
    parser.add_argument('--model_class', type=str, default='bilstm_crf')
    parser.add_argument('--pretrained_model', type=str, default='prajjwal1/bert-small',
                        help='Pre-trained model would be download from Hugging Face.')
    parser.add_argument('--save_model_dir', type=str, default='./save_model')
    parser.add_argument('--max_len', type=int, default=128, help='The maximum length of input.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true. ")

    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if args.model_class not in ['bilstm_crf', 'bert_bilstm_crf']:
        raise ValueError("The model not exists.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ngpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if ngpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tags = ['B-PER', 'M-PER', 'E-PER', 'S-PER',
            'B-ORG', 'M-ORG', 'E-ORG',  'S-ORG',
            'B-GPE', 'M-GPE', 'E-GPE', 'S-GPE',
            'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC',
            'O']
    START_TAG = '<START>'
    STOP_TAG = '<STOP>'
    vocab = Vocab(tags + [START_TAG, STOP_TAG])

    vocab.load_file(args.train_dataset)
    vocab.build_vocab()
    print('word2id>>', len(vocab.word2id))
    print('tag2id>>', vocab.tag2id)

    data_processor = DataProcessor()


    config = None
    bert = None
    if args.model_class.startswith('bert'):
        config = BertConfig.from_pretrained(args.pretrained_model)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        bert = BertModel.from_pretrained(args.pretrained_model, config=config)

    model_name = MODE_CLASS[args.model_class]
    model = model_name(vocab_size=len(vocab.word2id),
                       tag2id=vocab.tag2id,
                       embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim,
                       batch_size=args.batch_size,
                       dropout=args.dropout,
                       device=device,
                       config=config,
                       bert=bert)

    print('model>>', model)
    model.to(device)

    loss_fn = nn.NLLLoss()

    global_step = 0

    if args.do_train:
        print('Start to get train examples')
        train_examples = data_processor.get_train_examples(args.train_dataset)

        train_dataset = NerDataset(train_examples,
                                   word2id=vocab.word2id,
                                   tag2id=vocab.tag2id,
                                   max_len=args.max_len)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=collate_pad_and_sort)
        print('Start to get dev examples')
        dev_examples = data_processor.get_dev_examples(args.dev_dataset)
        dev_dataset = NerDataset(dev_examples,
                                 word2id=vocab.word2id,
                                 tag2id=vocab.tag2id,
                                 max_len=args.max_len)
        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=collate_pad_and_sort)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=3, verbose=True)

        for epoch in trange(args.num_train_epochs, desc="Epoch"):
            model.train()

            train_loss = 0.0
            train_steps = 0
            best_acc = 0.0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                optimizer.zero_grad()
                batch = tuple(p.to(device) for p in batch)

                inputs = {
                    'sentence': batch[0],
                    'seq_length': batch[2],
                    'mask': batch[3],
                    'tags': batch[1]
                }
                tags = batch[1]

                #outputs = model(**inputs)
                loss, tag_seq = model.neg_log_likelihood_loss(**inputs)
                # loss = loss_fn(outputs.view(-1, self.tag_size), tags.view(1, -1).squeeze(dim=0))
                # loss = loss_fn(outputs.transpose(1, 2), tags)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                train_loss += loss.detach().item()
                train_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    loss.backward()
                    optimizer.step()

            print('Start to eval...')
            model.eval()
            eval_loss = 0
            eval_steps = 0
            preds = None
            for batch in tqdm(dev_dataloader, desc="Evaluating"):
                batch = tuple(p.to(device) for p in batch)

                with torch.no_grad():
                    inputs = {
                        'sentence': batch[0],
                        'seq_length': batch[2],
                        'mask': batch[3],
                        'tags': batch[1]
                    }
                    tags = batch[1]
                    #outputs = model(**inputs)
                    loss, tag_seq = model.neg_log_likelihood_loss(**inputs)
                    #tmp_eval_loss = loss_fn(outputs.transpose(1, 2), tags)

                    eval_loss += loss.detach().mean().item()

                eval_steps += 1
                # print('outputs size>>', outputs.shape)
                # print('tags size>>', tags.shape)
                if preds is None:
                    preds = tag_seq.detach().cpu().numpy()
                    gold_tags = tags.detach().cpu().numpy()
                else:
                    # print('outputs>>', outputs.shape)
                    preds = np.append(preds, tag_seq.detach().cpu().numpy(), axis=1)
                    gold_tags = np.append(gold_tags, tags.detach().cpu().numpy(), axis=1)
            # print('preds size>>', preds)
            # print('gold_tags size>>', gold_tags)
            loss = train_loss / train_steps if args.do_train else None
            eval_loss = eval_loss / eval_steps


            #pred_tags = np.argmax(preds, axis=-1).flatten()
            # print('pred_tags size>>', pred_tags.shape)
            pred_tags = preds.flatten()
            gold_tags = gold_tags.flatten()
            pred_tags_list = [[vocab.id2tag[idx] for i, idx in enumerate(pred_tags) if idx in vocab.id2tag]]
            gold_tags_list = [[vocab.id2tag[idx] for i, idx in enumerate(gold_tags) if idx in vocab.id2tag]]
            # print('pred_tags size>>', pred_tags_list)
            # print('gold_tags size>>', gold_tags_list)
            result = {}
            result['epoch'] = epoch
            result['accuracy'] = accuracy_score(gold_tags_list, pred_tags_list)
            result['precision'] = precision_score(gold_tags_list, pred_tags_list)
            result['recall'] = recall_score(gold_tags_list, pred_tags_list)
            result['F1'] = f1_score(gold_tags_list, pred_tags_list)

            result['loss'] = loss
            result['eval_loss'] = eval_loss
            result['global_step'] = global_step

            output_eval_file = os.path.join(args.save_model_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                # logger.info("***** Eval result *****")
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))
                    # logger.info("%s = %s", key, str(result[key]))
                    writer.write("%s = %s \n" % (key, str(result[key])))

            output_model_file = os.path.join(args.save_model_dir,
                                             str(epoch) + "_" \
                                             + str(result['F1']) + '.pkl')
            if (epoch + 1) % 50 == 0:
                torch.save(model.state_dict(), output_model_file)

            early_stopping(eval_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Congratulation~ training done.')
