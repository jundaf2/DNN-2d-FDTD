import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param
import torch
import fdtd
import numpy as np

param = param.parameter()

if __name__ == '__main__':
    conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=param.opt.n_input_channels).cuda()
    data_src = fdtd.Grid(param.opt.n_domain_dim)
    data_src.update(param.opt.batch_size)
    input_data = data_src.train_data
    print(np.array(input_data).shape)
    input = model.Variable(
        torch.tensor(np.array(input_data)[:, np.newaxis, :, :], dtype=torch.float)).cuda()
    print(input.size())
    b, ch, h, w = input.size()
    b = 1
    lstm_h = h // param.opt.max_pool_kernal_size**3
    lstm_w = w // param.opt.max_pool_kernal_size**3
    h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))
    h_t2, c_t2 = conv_lstm_model.decoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))

    print(h_t.size(), c_t.size())
    conv_lstm_model.load_state_dict(torch.load('mymodule_params.pt'))
    conv_lstm_model.eval()
    conv_lstm_model.cuda()
    output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)
    output = output.cpu().detach().data
    fig1 = plt.figure("FDTD vs ConvRNN")
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    for epoch in range(1, 130):
        # list Incept-Net里大量用到的1x1卷积操作。基本上完全就是在通道与通道之间进行交互
        data_src.update(epoch % 130)
        future_data = [data_src.train_data[0]]  # list
        input_data = data_src.train_data
        input_data_others = np.array(input_data)[1:, :, :]
        # autoencoder forward, 网络结构越浅越好 ？
        # print(out.size())

        output = torch.cat([output[0,:,:,:], torch.tensor(input_data_others)], dim=0)
        # print(np.array(self.input_data).shape, np.array(self.future_data).shape)
        input = model.Variable(
            torch.tensor(np.array(output)[:, np.newaxis, :, :], dtype=torch.float)).cuda()

        output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)

        h_t = h_t.detach().data
        c_t = c_t.detach().data
        h_t2 = h_t.detach().data
        c_t2 = c_t.detach().data

        output = output.cpu().detach().data

        ax1.imshow(future_data[0],  interpolation='none')
        ax2.imshow(output[0, 0, :,:], interpolation='none')
        plt.pause(0.01)