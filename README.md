# Gated-Convolutional-Networks

##Description
A PyTorch implementation of [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083). This paper applies a convolutional approach to language modeling with a novel Gated-CNN model.
As all other repos don't support varied sentence lengths, This implementation remove the fixed sized sentences restriction. One of the highlights of this implementation is the lots of comments added. And I'm still working on reproducing the result mentioned in the original paper. All discussions are welcome.

##Architecture
<img src="http://pcaqhp90s.bkt.clouddn.com/2018-10-14-15394927176407.jpg" width="50%" height="50%">

##Requirements
* PyTorch 0.4

##TODO
* [ ] Add argparse
* [ ] Fine-tune to reproduce the result in the paper
* [ ] Add module to extract hidden states of GCNN as the dynamic embedding of input.