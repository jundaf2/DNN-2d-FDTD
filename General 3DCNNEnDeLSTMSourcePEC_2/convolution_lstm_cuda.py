
import torch
import torch.nn as nn
from torch.autograd import Variable
import parameters as param
import numpy as np
param = param.parameter()

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = True*bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              stride=(1, 1),
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        #c_next = f * c_cur + i * g
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, shape):
        height, width = shape
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        """
            ARCHITECTURE:
            Encoder (ConvLSTM)
            Encoder Vector (final hidden state of encoder)
            Decoder (ConvLSTM) - takes Encoder Vector as input
            Decoder (3D CNN) - produces regression predictions for our model
        """
        super(EncoderDecoderConvLSTM, self).__init__()

        kernal_size = 2
        features =  np.logspace(0, 5, 6, base=2, dtype=int)*in_chan

        self.encoder_CNN1 = nn.Conv2d(in_channels=features[0], out_channels=features[1],
                                     kernel_size=kernal_size*2, padding=0, stride=2)
        nn.init.xavier_uniform_(self.encoder_CNN1.weight)
        self.fn1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN1 = nn.BatchNorm2d(features[1])
        self.encoder_CNN2 = nn.Conv2d(in_channels=features[1], out_channels=features[2],
                                      kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.encoder_CNN2.weight)
        self.fn2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN2 = nn.BatchNorm2d(features[2])
        self.encoder_CNN3 = nn.Conv2d(in_channels=features[2], out_channels=features[3],
                                      kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.encoder_CNN3.weight)
        self.fn3 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN3 = nn.BatchNorm2d(features[3])
        self.encoder_CNN4 = nn.Conv2d(in_channels=features[3], out_channels=features[4],
                                      kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.encoder_CNN4.weight)
        self.fn4 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN4 = nn.BatchNorm2d(features[4])
        self.encoder_CNN5 = nn.Conv2d(in_channels=features[4], out_channels=features[5],
                                      kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.encoder_CNN5.weight)
        self.fn5 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN5 = nn.BatchNorm2d(features[5])

        # RNN的核心思想即是将数据按时间轴展开，每一时刻数据均对应相同的神经单元，且上一时刻的结果能传递至下一时刻
        self.lstm1 = LSTMCell(input_dim=features[5], hidden_dim=nf, kernel_size=(1, 1), bias=True)

        self.lstm2 = LSTMCell(input_dim=nf, hidden_dim=features[5], kernel_size=(1, 1), bias=True)
        #self.conv1 = nn.Conv2d(in_channels=features[5], out_channels=features[5],
        #                              kernel_size=kernal_size, padding=0, stride=2)

        self.decoder_CNN5 = nn.ConvTranspose2d(in_channels=features[5], out_channels=features[4],
                                      kernel_size=(3, 3), padding=0, stride=2)
        nn.init.xavier_uniform_(self.decoder_CNN5.weight)
        self.fn6 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN6 = nn.BatchNorm2d(features[4])
        self.decoder_CNN4 = nn.ConvTranspose2d(in_channels=features[4], out_channels=features[3],
                                               kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.decoder_CNN4.weight)
        self.fn7 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN5 = nn.BatchNorm2d(features[3])
        self.decoder_CNN3 = nn.ConvTranspose2d(in_channels=features[3], out_channels=features[2],
                                               kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.decoder_CNN3.weight)
        self.fn8 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN8 = nn.BatchNorm2d(features[2])
        self.decoder_CNN2 = nn.ConvTranspose2d(in_channels=features[2], out_channels=features[1],
                                               kernel_size=kernal_size, padding=0, stride=2)
        nn.init.xavier_uniform_(self.decoder_CNN2.weight)
        self.fn9 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN9 = nn.BatchNorm2d(features[1])
        self.decoder_CNN1 = nn.ConvTranspose2d(in_channels=features[1], out_channels=features[0],
                                               kernel_size=kernal_size*2, padding=0, stride=2)
        nn.init.xavier_uniform_(self.decoder_CNN1.weight)
        self.fn10 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.BN10 = nn.BatchNorm2d(features[0])


    def forward(self, x, h1, c1, h2, c2):
        """
            Parameters
            ----------
            input_tensor:
                5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        future_seq = param.opt.n_future_seq
        b, seq_len, ch, h, w = x.size()

        # encoder
        x = self.encoder_CNN1(x)
        x = self.fn1(x)
        x = self.BN1(x)
        print(x.size())
        x = self.encoder_CNN2(x)
        x = self.fn2(x)
        x = self.BN2(x)
        print(x.size())
        x = self.encoder_CNN3(x)
        x = self.fn3(x)
        x = self.BN3(x)
        print(x.size())
        x = self.encoder_CNN4(x)
        x = self.fn4(x)
        x = self.BN4(x)
        print(x.size())
        x = self.encoder_CNN5(x)
        x = self.fn5(x)
        x = self.BN5(x)
        print(x.size())

        # latent space
        for i in range(seq_len+1):
            h1, c1 = self.lstm1(input_tensor=x[:, i, :, :, :],
                                                   cur_state=[h1, c1])
        encoder_vector = [h1]
        o = param.opt.n_future_seq
        encoder_vector = torch.stack(encoder_vector, 1)
        print(encoder_vector.size())
        encoder_vector = torch.repeat_interleave(encoder_vector, o, dim=0)
        print(encoder_vector.size())
        outputs = []
        for i in range(o):
            h2, c2 = self.lstm1(input_tensor=encoder_vector[:, i, :, :, :],
                                                   cur_state=[h2, c2])
            outputs += [h2]
        outputs = torch.stack(outputs, 1)
        print(outputs.size())
        # decoder
        outputs = self.encoder_CNN5(outputs)
        outputs = self.fn6(outputs)
        outputs = self.BN6(outputs)
        print(outputs.size())
        outputs = self.encoder_CNN4(outputs)
        outputs = self.fn7(outputs)
        outputs = self.BN7(outputs)
        print(outputs.size())
        outputs = self.encoder_CNN3(outputs)
        outputs = self.fn8(outputs)
        outputs = self.BN8(outputs)
        print(outputs.size())
        outputs = self.encoder_CNN2(outputs)
        outputs = self.fn9(outputs)
        outputs = self.BN9(outputs)
        print(outputs.size())
        outputs = self.encoder_CNN1(outputs)
        outputs = self.fn10(outputs)
        outputs = self.BN10(outputs)
        print(outputs.size())

        return outputs, h1, c1, h2, c2



