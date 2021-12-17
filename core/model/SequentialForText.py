import torch
from torch import nn

class SequentialFortText(nn.Module):

    def __init__(self,layer_num=1,GRU=False,embedding=True,embedding_dim=200):
        self.layer_num = layer_num
        self.GRU = GRU

        if embedding:
            self.embedding = nn.Embedding()
        else:
            self.embedding = nn.Linear()




    def forward(self):
        pass


