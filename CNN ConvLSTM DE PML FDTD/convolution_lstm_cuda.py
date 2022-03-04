import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import parameters as param
param = param.parameter()

# CNN可以提取空间特性,LSTM可以提取时间特性,ConvLSTM可以同时利用时空特性
class ConvLSTMCell(nn.Module):

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

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = False*bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              stride= 1,
                              padding=self.padding,
                              bias=self.bias)



    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print(input_tensor.size(), h_cur.size())
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

        # Convolution 1
        self.kernel_size = (3, 3)
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=self.kernel_size,
                               padding=self.padding)
        nn.init.constant_(self.conv1.weight, 1)
        self.swish1 = nn.ReLU()

        # Max Pool 1
        # 通常认为如果选取区域均值(mean pooling)，往往能保留整体数据的特征，较好的突出背景信息；
        # 如果选取区域最大值(max pooling)，则能更好保留纹理特征
        max_pool_kernal_size = param.opt.max_pool_kernal_size
        self.maxpool1 = nn.MaxPool2d(kernel_size=max_pool_kernal_size, return_indices=True)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=self.kernel_size,
                               padding=self.padding)
        nn.init.constant_(self.conv2.weight, 1)
        self.swish2 = nn.ReLU()

        # Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=max_pool_kernal_size, return_indices=True)

        # Convolution 3
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=self.kernel_size,
                               padding=self.padding)
        nn.init.constant_(self.conv3.weight, 1)
        self.swish3 = nn.ReLU()

        # Max Pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=max_pool_kernal_size, return_indices=True)

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(1, 1),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(1, 1),
                                               bias=True)

        # De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3,padding=self.padding)
        nn.init.constant_(self.deconv2.weight, 1)
        self.swish6 = nn.ReLU()
        # Max UnPool 2
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=max_pool_kernal_size)


        # DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3,padding=self.padding)
        nn.init.constant_(self.deconv3.weight, 1)
        self.swish7 = nn.ReLU()
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=max_pool_kernal_size)


        self.deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3,padding=self.padding)
        nn.init.constant_(self.deconv4.weight, 1)
        self.swish8 = nn.ReLU()
        self.maxunpool3 = nn.MaxUnpool2d(kernel_size=max_pool_kernal_size)



    def autoencoder(self, x, h_t, c_t, h_t2, c_t2):
        outputs = []
        # encoder
        #for t in range(seq_len):
        input_tensor = x.permute(1, 0, 2, 3)
        seq_len = list(input_tensor.size())[0]
        encoder_vector = []
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=input_tensor[np.newaxis, t, :, :, :], cur_state=[h_t, c_t])
            encoder_vector += [h_t]
        encoder_vector = torch.stack(encoder_vector, 0)
        #print(encoder_vector.size())
        # decoder
        for t in range(seq_len):
            h_t2, c_t2 = self.decoder_1_convlstm(input_tensor=encoder_vector[t, :, :, :, :], cur_state=[h_t2, c_t2])
            outputs += [h_t2]

        outputs = torch.stack(outputs, 0)
        outputs = outputs.permute(1, 2, 0, 3, 4)
        #print(outputs.size())
        #outputs = self.decoder_CNN(outputs)
        #outputs = torch.nn.Sigmoid()(outputs)

        return outputs[:,0,:,:,:], h_t, c_t, h_t2, c_t2

    def forward(self, x, h_t, c_t, h_t2, c_t2):
        """
            Parameters
            ----------
            input_tensor:
                5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        # future_seq = param.opt.n_future_seq

        out = self.conv1(x)
        out = self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        #print(out.size())

        out = self.conv2(out)
        out = self.swish2(out)
        size2 = out.size()
        out,indices2 = self.maxpool2(out)
        #print(out.size())

        out = self.conv3(out)
        out = self.swish3(out)
        size3 = out.size()
        out, indices3 = self.maxpool3(out)
        #print(out.size())


        # save for use in decoder CNN
        out_others = out[1:, :, :, :]
        # autoencoder forward, 网络结构越浅越好 ？
        out, h_t, c_t, h_t2, c_t2 = self.autoencoder(out, h_t, c_t, h_t2, c_t2)
        #print(out.size())

        out = torch.cat([out, out_others], dim=0)
        #print(out.size())

        out = self.maxunpool1(out, indices3, size3)
        out = self.deconv2(out)
        out = self.swish6(out)
        #print(out.size())

        #print(out.size())
        out = self.maxunpool2(out, indices2, size2)
        out = self.deconv3(out)
        out = self.swish7(out)

        #print(out.size())
        out = self.maxunpool3(out, indices1, size1)
        out = self.deconv4(out)
        out = self.swish8(out)


        return out, h_t, c_t, h_t2, c_t2



