import torch
from torch.utils.data import DataLoader
from model import GatedCNN
from data import *
from train import training
torch.manual_seed(1)
from utils import start_time,get_time_dif

if __name__ == '__main__':
    # ---- prepare data ---- #
    train_corpus = Corpus(data_dir='./data')
    print('training corpus has',len(train_corpus.corpus),'lines')
    vocab = build_vocab(train_corpus.all_tokens, vocab_size=config.vocab_size)
    print('vocab_size is :',len(vocab))
    train_corpus_ids = train_corpus.tokenize(vocab)  # contains all tokenized corpus
    train_set = CorpusDataset(train_corpus_ids)
    train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                          collate_fn=collate_fn)

    # todo load test dataset

    # ---- model&optim&etc... ---- #
    model=GatedCNN(vocab_size=len(vocab),embed_dim=config.embed_dim,
                   kernel_width=config.kernel_width,out_channel=config.out_channel,
                   n_layers=config.n_layers,res_block_cnt=config.res_block_cnt,
                   dropout=config.dropout)
    if config.use_cuda:
        model=model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # very important to set ignore_index to 0

    # train
    print('start time is',get_time_dif(start_time))
    loss=training(model,optimizer,criterion,train_dl)
    print('loss for this epoch is',loss)
    print('time used for this epoch',get_time_dif(start_time))


