import torch

class Config():
    # data config
    MAX_SENT=100
    embed_dim=64
    vocab_size=100000


    #GCNN config
    n_layers=12
    kernel_width=6
    out_channel=64
    res_block_cnt=5  # todo
    dropout=0.1

    #training config

    epoch=20
    batch_size=10
    use_cuda=torch.cuda.is_available()
    CUDA_VISIBLE_DEVICES = 5  # 可用显卡编号
    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)

config=Config()