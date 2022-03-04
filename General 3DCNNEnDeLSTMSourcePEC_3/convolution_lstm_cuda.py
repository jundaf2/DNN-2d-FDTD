
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



        # RNN的核心思想即是将数据按时间轴展开，每一时刻数据均对应相同的神经单元，且上一时刻的结果能传递至下一时刻
        self.lstm1 = LSTMCell(input_dim=in_chan, hidden_dim=nf, kernel_size=(5, 5), bias=True)

        self.lstm2 = LSTMCell(input_dim=nf, hidden_dim=param.opt.n_output_channels, kernel_size=(5, 5), bias=True)



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
        #print(x.size())
        # latent space
        encoder_vector = []
        for i in range(seq_len):
            h1, c1 = self.lstm1(input_tensor=x[:, i, :, :, :],
                                                   cur_state=[h1, c1])
            encoder_vector += [h1]
        o = param.opt.n_future_seq
        encoder_vector = torch.stack(encoder_vector, 1)

        outputs = []
        for i in range(o):
            h2, c2 = self.lstm2(input_tensor=encoder_vector[:, i, :, :, :],
                                                   cur_state=[h2, c2])
            outputs += [h2]
        outputs = torch.stack(outputs, 1)
        #print(outputs.size())

        return outputs, h1, c1, h2, c2



