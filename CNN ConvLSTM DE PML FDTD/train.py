
import torch
from torchviz import make_dot
import numpy as np
import fdtd
import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param
import matplotlib
matplotlib.use('TKAgg')

param = param.parameter()

class Train(object):

    def __init__(self, domain_dim, input_channels, output_channels):
        print("Traning Preparation:")
        self.fig1 = plt.figure("FDTD vs ConvRNN")
        self.ax1 = plt.subplot(231)
        self.ax2 = plt.subplot(232)
        self.ax3 = plt.subplot(234)
        self.ax4 = plt.subplot(235)
        self.ax5 = plt.subplot(236)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_data = []
        self.future_data = []
        self.data_src = fdtd.Grid(domain_dim)


    def show(self, object, source, in_field, targ, pred, epoch):
        # print("Animation:")
        self.ax1.imshow(object, interpolation='none')
        self.ax2.imshow(source, interpolation='none')
        self.ax3.imshow(in_field, interpolation='none')
        self.ax4.imshow(targ, interpolation='none')
        self.ax5.imshow(pred, interpolation='none')
        plt.pause(0.01)
        plt.savefig('C:/Work/High_Perfomance_Multiphysics_Simulation_CEM/ML-CEM/DNN/Conv_EnDe_FDTD_LSTM_PML_Object/result_1005/epoch {}.png'.format(epoch), dpi=50, bbox_inches='tight')


        '''
        for i in range(self.input_channels+self.output_channels):
            self.ax1.imshow(self.input_data[-self.input_channels - self.output_channels + i], interpolation='none')
            self.ax2.imshow(self.output_data[ -self.input_channels + i], interpolation='none')
            plt.pause(0.001)
        '''


    def gpu_info(self):
        print("GPU Information:")
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))


    def train(self):

        conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=self.input_channels).cuda()

        self.data_src.update(param.opt.batch_size)
        self.input_data = self.data_src.train_data
        print(np.array(self.input_data).shape)
        input = model.Variable(
                torch.tensor(np.array(self.input_data)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)).cuda()
        print(input.size())
        b, t, ch, h, w = input.size()

        lstm_h = h // param.opt.max_pool_kernal_size**3
        lstm_w = w // param.opt.max_pool_kernal_size**3
        h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))
        h_t2, c_t2 = conv_lstm_model.decoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))

        print(h_t.size(), c_t.size())
        # conv_lstm_model.cuda()
        optimizer = torch.optim.Adam(conv_lstm_model.parameters(), lr=param.opt.lr, betas=(param.opt.beta_1, param.opt.beta_2))
        criterion = torch.nn.MSELoss()
        count_loss = []
        duration = 130
        for epoch in range(1, int(3e4)):
            if epoch%130 ==0:
                self.data_src = fdtd.Grid(param.opt.n_domain_dim)
                self.data_src.update(epoch % duration)
            # list Incept-Net里大量用到的1x1卷积操作。基本上完全就是在通道与通道之间进行交互
            self.input_data = self.data_src.train_data

            self.data_src.update(epoch % duration)
            self.future_data = [self.data_src.train_data[0]] # list

            # print(np.array(self.input_data).shape, np.array(self.future_data).shape)
            input = model.Variable(
                torch.tensor(np.array(self.input_data)[:, np.newaxis, :, :], dtype=torch.float)).cuda()
            target = model.Variable(
                torch.tensor(np.array(self.future_data)[:, np.newaxis, :, :], dtype=torch.float)).cuda()

            output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)

            h_t = h_t.detach().data
            c_t = c_t.detach().data
            h_t2 = h_t.detach().data
            c_t2 = c_t.detach().data

            loss = criterion(output[0, 0, :,:], target[0, 0, :,:]) / target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lass = loss.data.cpu().detach().numpy()
            if epoch % 100 == 0:
                count_loss.append(lass)
            #print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, lass))
            output = output.cpu().detach().data
            # target = target.cpu().detach().data
            if epoch % duration == 90:
            #if epoch % 20 == 0:
                self.show(self.input_data[1], self.input_data[2], self.input_data[0], self.future_data[0], output[0, 0, :,:], epoch)
        torch.save(conv_lstm_model.state_dict(), 'mymodule_params.pt') # params
        torch.save(conv_lstm_model, 'mymodule_model.pt') # all
        plt.plot(np.array(count_loss))
        plt.show()


if __name__ == '__main__':
    plt.ion()
    a = Train(param.opt.n_domain_dim, param.opt.n_input_channels, param.opt.n_output_channels)
    a.gpu_info()
    a.train()
