import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class GatedCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 kernel_width, out_channel, n_layers,
                 res_block_cnt, dropout=0.1):
        super(GatedCNN, self).__init__()
        self.res_block_cnt = res_block_cnt
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)

        self.conv_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_channel,
                                kernel_size=kernel_width)
        self.b_0 = nn.Parameter(torch.zeros(out_channel, 1))  # same as paper
        self.conv_gate_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_channel,
                                     kernel_size=kernel_width)
        self.c_0 = nn.Parameter(torch.zeros(out_channel, 1))

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                                              kernel_size=kernel_width)
                                    for _ in range(n_layers)])

        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(out_channel, 1))  # collections of b
                                    for _ in range(n_layers)])

        self.conv_gates = nn.ModuleList([nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                                                   kernel_size=kernel_width)
                                         for _ in range(n_layers)])

        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(out_channel, 1))
                                    for _ in range(n_layers)])

        self.fc = nn.Linear(out_channel, vocab_size)
        self.dropout = nn.Dropout(p=dropout)  # todo use dropout

    # conv1d Input: (N, Cin, Lin)
    # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

    def forward(self, seq):
        # seq:(batch,seq_len)
        batch_size = seq.size(0)
        seq_len = seq.size(1)
        x = self.embedding(seq)  # x: (batch,seq_len,embed_dim)
        x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel

        x = self.padding_left(x)  # x:(batch,embed_dim,seq_len+kernel-1)  #padding left with 0
        A = self.conv_0(x)  # A: (batch,out_channel,seq_len)   seq_len because of padding (kernel-1)
        A += self.b_0  # b_0 broadcast
        B = self.conv_gate_0(x)  # B: (batch,out_channel,seq_len)
        B += self.c_0

        h = A * F.sigmoid(B)  # h: (batch,out_channel,seq_len)
        # todo: add resnet
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.convs, self.conv_gates)):
            h = self.padding_left(h)  # h: (batch,out_channel,seq_len+kernel-1)
            A = conv(h) + self.bs[i]
            B = conv_gate(h) + self.cs[i]
            h = A * F.sigmoid(B)  # h: (batch,out_channel,seq_len+kernel-1)
            if i % self.res_block_cnt == 0:  # todo Is this correct?
                h += res_input
                res_input = h

        h.transpose_(1, 2)  # h:(batch,seq_len,out_channel)

        logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        logic.transpose_(1,2)  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return logic


if __name__ == '__main__':
    model = GatedCNN(vocab_size=config.vocab_size, embed_dim=config.embed_dim,
                     kernel_width=config.kernel_width, out_channel=config.out_channel,
                     n_layers=config.n_layers, res_block_cnt=config.res_block_cnt)

    input = torch.LongTensor([[1, 2, 3],
                              [4, 5, 6]])

    print(model(input))
