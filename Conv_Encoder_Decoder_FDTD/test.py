import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param
import torch
import fdtd
import numpy as np

param = param.parameter()

if __name__ == '__main__':
    conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=param.opt.n_input_channels)
    conv_lstm_model.load_state_dict(torch.load('mymodule.pt'))
    conv_lstm_model.eval()
    conv_lstm_model.cuda()
    # start data
    data_src = fdtd.Simulation(param.opt.n_domain_dim)
    data_src.data_gen(param.opt.batch_size)
    input_data = data_src.dataset[-param.opt.batch_size:]
    input_data = input_data[0]

    for i in range(100):
        input = model.Variable(
            torch.tensor(np.array(input_data)[np.newaxis, np.newaxis, np.newaxis, :, :], dtype=torch.float)).cuda()

        output = conv_lstm_model(input)

        output = output.cpu().detach().data
        print('epoch: {}, source_jz: {}'.format(i + 1, data_src.grid.source_jz))
        plt.imshow(output[0, 0, 0, :, :])
        plt.pause(0.01)

        #data_src.data_gen(param.opt.batch_size)
        input_data = output[0, 0, 0, :, :]
        #input_data[int(data_src.grid.width/2), int(data_src.grid.height / 2)] += data_src.grid.source_jz
