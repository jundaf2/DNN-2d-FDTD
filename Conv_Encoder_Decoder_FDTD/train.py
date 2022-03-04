
import torch
import numpy as np
import fdtd
import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param

param = param.parameter()

class Train(object):

    def __init__(self, domain_dim, input_channels, output_channels):
        print("Traning Preparation:")
        self.fig1 = plt.figure("FDTD vs ConvRNN")
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(122)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_data = []
        self.future_data = []
        self.data_src = fdtd.Simulation(domain_dim)

    def show(self, targ, pred):
        # print("Animation:")

        self.ax1.imshow(targ, interpolation='none')
        self.ax2.imshow(pred, interpolation='none')
        plt.pause(0.01)


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

        conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=self.input_channels)
        conv_lstm_model.cuda()
        optimizer = torch.optim.Adam(conv_lstm_model.parameters(), lr=param.opt.lr, betas=(param.opt.beta_1, param.opt.beta_2))
        criterion = torch.nn.MSELoss()

        for epoch in range(80):
            self.data_src.data_gen(param.opt.batch_size)
            self.input_data = self.data_src.dataset[-param.opt.batch_size:]  # list
            self.data_src.data_gen(self.output_channels)
            self.future_data = self.data_src.dataset[-self.output_channels:]  # list
            print('fdtd step {}'.format(self.data_src.time // self.data_src.time_step))

            input = model.Variable(
                torch.tensor(np.array(self.input_data)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)).cuda()
            target = model.Variable(
                torch.tensor(np.array(self.future_data)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)).cuda()

            output = conv_lstm_model(input)

            loss = criterion(output, target) / target.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data.cpu().detach().numpy()))
            output = output.cpu().detach().data
            # target = target.cpu().detach().data
            self.show(self.future_data[0], output[0,0,0,:,:])
        torch.save(conv_lstm_model.state_dict(), 'mymodule.pt')

        # self.output_data.append(output)

if __name__ == '__main__':
    plt.ion()
    a = Train(param.opt.n_domain_dim, param.opt.n_input_channels, param.opt.n_output_channels)
    a.gpu_info()
    a.train()
