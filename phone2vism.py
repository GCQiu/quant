import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
#from wav2vec import Wav2Vec2Model
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Basic_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases



class TCN(nn.Module):

    def __init__(self, ):
        super(TCN, self).__init__()
        self.embedding = nn.Embedding(106, 128)
        #self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(128, 64, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(64, 128, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(128, 256, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(256, 256, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(256, 128, stride=1, kernel_size=3, padding=1)
        #self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        #x = self.avgpool(x)
        #print(x.shape)
        #x = x.view(x.size(0), -1)
        x = x.transpose(2, 1)
        x = self.fc(x)
        return x

class TCN_GRU_res_kor(nn.Module):

    def __init__(self, ):
        super(TCN_GRU_res_kor, self).__init__()
        self.embedding = nn.Embedding(60, 128)
        #self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(128, 256, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(256, 256, stride=1, kernel_size=3, padding=1)


        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(256, 256,1, batch_first=True, bidirectional=True) #batch_size, sequence_length, input_size

        #self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(256*2, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        out = self.layer1(x)
        x = out + x
        out = self.layer2(x)
        x = out + x
        out = self.layer3(x)
        x = out + x
        out = self.layer4(x)
        x = out + x
        x = self.layer5(x)
        out = self.layer6(x)
        x = out + x
        x = x.transpose(2, 1)

        if hidden is None:
            hidden = x.data.new(2, x.shape[0], 256).fill_(0).float()
        else:
            hidden = hidden

        x, hidden = self.gru(x, hidden)

        x = self.fc(x)
        return x

class TCN_GRU(nn.Module):

    def __init__(self, ):
        super(TCN_GRU, self).__init__()
        self.embedding = nn.Embedding(106, 128)
        #self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(128, 256, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(256, 256, stride=1, kernel_size=3, padding=1)


        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(256, 256,1, batch_first=True, bidirectional=True) #batch_size, sequence_length, input_size

        #self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(256*2, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        #print(x.shape)
        x = x.transpose(2, 1)

        if hidden is None:
            hidden = x.data.new(2, x.shape[0], 256).fill_(0).float()
        else:
            hidden = hidden
        x, hidden = self.gru(x, hidden)

        x = self.fc(x)
        return x

class TCN_GRU_res_jap(nn.Module):

    def __init__(self, ):
        super(TCN_GRU_res_jap, self).__init__()
        self.embedding = nn.Embedding(79, 128)
        #self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(128, 256, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(256, 256, stride=1, kernel_size=3, padding=1)


        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(256, 256,1, batch_first=True, bidirectional=True) #batch_size, sequence_length, input_size

        #self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(256*2, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        out = self.layer1(x)
        x = out + x
        out = self.layer2(x)
        x = out + x
        out = self.layer3(x)
        x = out + x
        out = self.layer4(x)
        x = out + x
        x = self.layer5(x)
        out = self.layer6(x)
        x = out + x
        x = x.transpose(2, 1)

        if hidden is None:
            hidden = x.data.new(2, x.shape[0], 256).fill_(0).float()
        else:
            hidden = hidden

        x, hidden = self.gru(x, hidden)

        x = self.fc(x)
        return x

class TCN_GRU_res_ours(nn.Module):

    def __init__(self, ):
        super(TCN_GRU_res_ours, self).__init__()
        self.embedding = nn.Embedding(72, 128)
        #self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(128, 128, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(128, 256, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(256, 256, stride=1, kernel_size=3, padding=1)


        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(256, 256,1, batch_first=True, bidirectional=True) #batch_size, sequence_length, input_size

        #self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(256*2, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        out = self.layer1(x)
        x = out + x
        out = self.layer2(x)
        x = out + x
        out = self.layer3(x)
        x = out + x
        out = self.layer4(x)
        x = out + x
        x = self.layer5(x)
        out = self.layer6(x)
        x = out + x
        x = x.transpose(2, 1)

        if hidden is None:
            hidden = x.data.new(2, x.shape[0], 256).fill_(0).float()
        else:
            hidden = hidden
        x, hidden = self.gru(x, hidden)

        x = self.fc(x)
        return x

class TCN_GRU_res_multilingual(nn.Module):

    def __init__(self, ):
        super(TCN_GRU_res_multilingual, self).__init__()
        self.embedding = nn.Embedding(392, 512)
        # self.feature = nn.Conv1d(256, 64, stride=1, kernel_size=1, padding=0)
        self.layer1 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer2 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer3 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer4 = Basic_layer(512, 256, stride=1, kernel_size=3, padding=1)
        self.layer5 = Basic_layer(256, 512, stride=1, kernel_size=3, padding=1)
        self.layer6 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer7 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer8 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)
        self.layer9 = Basic_layer(512, 512, stride=1, kernel_size=3, padding=1)

        # 这里指定了BATCH FIRST,所以输入时BATCH应该在第一维度
        self.gru = nn.GRU(512, 512, 1, batch_first=True, bidirectional=True)  # batch_size, sequence_length, input_size

        # self.avgpool = nn.AvgPool1d(2, stride=1)
        self.fc = nn.Linear(512 * 2, 10)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = x.transpose(2, 1)
        out = self.layer1(x)
        x = out + x
        out = self.layer2(x)
        x = out + x
        out = self.layer3(x)
        x = out + x
        out = self.layer4(x)
        x = out #+ x
        x = self.layer5(x)
        out = self.layer6(x)
        x = out + x
        out = self.layer7(x)
        x = out + x
        out = self.layer8(x)
        x = out + x
        x = x.transpose(2, 1)

        if hidden is None:
            hidden = x.data.new(2, x.shape[0], 512).fill_(0).float()
        else:
            hidden = hidden
        x, hidden = self.gru(x, hidden)

        x = self.fc(x)
        return x


if __name__ =='__main__':
    model = TCN()
    print(model)
    torch.save(model.state_dict(), 'model_TCN_.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.ones(1,21).long()
    out = model(dummy_input)
    #print(out)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=10,
                      verbose=False, input_names=["input"], output_names=["output"] \
                      , dynamic_axes={'input': {0: 'batch_size',1:'length'},  # variable length axes
                                      'output': {0: 'batch_size'}})


