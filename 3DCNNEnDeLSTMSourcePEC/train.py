
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


    def show(self, object_epsilon, object_sigma, in_field, targ, pred, epoch):
        # print("Animation:")
        self.ax1.imshow(object_epsilon, interpolation='none')
        self.ax2.imshow(object_sigma, interpolation='none')
        self.ax3.imshow(in_field, interpolation='none')
        self.ax4.imshow(targ, interpolation='none')
        self.ax5.imshow(pred, interpolation='none')
        self.ax1.set_title("Modified Permeability")
        self.ax2.set_title("Modified Conductivity")
        self.ax3.set_title("$E_z^{n-1}$")
        self.ax4.set_title("$E_z^{n}$")
        self.ax5.set_title("$E_z^{n+1}$")
        plt.pause(0.01)
        plt.savefig('C:/Work/High_Perfomance_Multiphysics_Simulation_CEM/ML-CEM/DNN/3DCNNEnDeLSTMSourcePEC/result/epoch {}.png'.format(epoch), dpi=50, bbox_inches='tight')


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

        ez_list = [0, 0]
        input_list = []
        for prepare_index in range(50):
            self.data_src.update(prepare_index)
            self.input_data = self.data_src.train_data
            ez_list.append(self.input_data[0])
            ez_list = ez_list[-2:]  # update to the recent 2
            input_list = ez_list.copy()
            input_list.extend(self.data_src.train_data[-2:])

        start_index = prepare_index + 1
        print(np.array(input_list).shape)

        input = model.Variable(
                torch.tensor(np.array(input_list)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)).cuda()
        print(input.size())
        b, t, ch, h, w = input.size()

        lstm_h = h
        lstm_w = w
        h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))
        h_t2, c_t2 = conv_lstm_model.decoder_4_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))

        print(h_t.size(), c_t.size())
        # conv_lstm_model.cuda()
        optimizer = torch.optim.Adam(conv_lstm_model.parameters(), lr=param.opt.lr, betas=(param.opt.beta_1, param.opt.beta_2))
        criterion = torch.nn.MSELoss()
        count_loss = []
        duration = 130-start_index
        index = 0
        for epoch in range(0, int(param.opt.epochs)):
            if epoch%(duration) ==0:
                index = start_index
                self.data_src = fdtd.Grid(param.opt.n_domain_dim)
                for prepare_index in range(50):
                    self.data_src.update(prepare_index)
                    ez_list.append(self.data_src.train_data[0])
                    ez_list = ez_list[-2:]  # update to the recent 2
                input_list = ez_list.copy()
                input_list.extend(self.data_src.train_data[-2:])
                h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b, shape=(lstm_h, lstm_w))
                h_t2, c_t2 = conv_lstm_model.decoder_4_convlstm.init_hidden(batch_size=b, shape=(lstm_h, lstm_w))

            self.data_src.update(index)
            self.future_data = [self.data_src.train_data[0]] # list

            # print(np.array(self.input_data).shape, np.array(self.future_data).shape)
            input = model.Variable(
                torch.tensor(np.array(input_list)[np.newaxis, np.newaxis, :,  :, :], dtype=torch.float)).cuda()
            ez_list.append(self.data_src.train_data[0])
            ez_list = ez_list[-2:]  # update to the recent 2
            input_list = ez_list.copy()
            input_list.extend(self.data_src.train_data[-2:])
            target = model.Variable(
                torch.tensor(np.array(self.future_data)[np.newaxis, np.newaxis, :,  :, :], dtype=torch.float)).cuda()

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
            #print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, lass))
            output = output.cpu().detach().data
            #if epoch % duration == 90:
            if epoch % 10 == 0:
                self.show(input_list[2], input_list[3], input_list[0], self.future_data[0], output[0, 0, 0, :,:], epoch)
            index += 1

        #torch.save(conv_lstm_model.state_dict(), 'mymodule_params.pt') # params
        #torch.save(conv_lstm_model, 'mymodule_model.pt') # all

if __name__ == '__main__':
    plt.ion()
    a = Train(param.opt.n_domain_dim, param.opt.n_input_channels, param.opt.n_output_channels)
    a.gpu_info()
    a.train()
