import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param
import torch
import fdtd
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
#matplotlib.use('TKAgg')
from PIL import Image
from DatasetGen import CircularPEC
from torch.utils.data import DataLoader
import h5py

if __name__ == '__main__':
    param = param.parameter()
    gif_images = []
    outfilename = "FDTD_vs_DNN.gif"  # 转化的GIF图片名称

    conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=param.opt.n_input_channels)
    with h5py.File("CircularPEC_Dataset.hdf5", 'r') as f:  # read high dimension data
        print(f.keys())
        batch_sequencial_data = f.get("batch_sequencial_data")[:]
        batch_sequencial_target_data = f.get("batch_sequencial_target_data")[:]

    size= 90
    '''
    data_src = fdtd.Grid(param.opt.n_domain_dim, radius=15)
    ez_list = [0, 0]
    input_list = []
    for prepare_index in range(50):
        data_src.update(prepare_index)
        ez_list.append(data_src.train_data[0])
        ez_list = ez_list[-2:]  # update to the recent 2
        input_list = ez_list.copy()
        input_list.extend(data_src.train_data[-1:])

    ez_list_output = ez_list.copy()
    start_index = prepare_index + 1
    print(np.array(input_list).shape)
    '''

    input_list = batch_sequencial_data[size, 0, :, :, :]
    #ez_list_output = [batch_sequencial_data[size, 0, 0, :, :]]

    input = model.Variable(
        torch.tensor(input_list[np.newaxis, np.newaxis, :, :, :], dtype=torch.float))
    print(input.size())
    b, _, ch, h, w = input.size()
    lstm_h = h
    lstm_w = w
    h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))
    h_t2, c_t2 = conv_lstm_model.decoder_4_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))

    fig2 = plt.figure("imput data and output data")

    conv_lstm_model.load_state_dict(torch.load('mymodule_params_1.pt', map_location=torch.device('cpu'))['model'])
    conv_lstm_model.eval()

    output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)
    output = output.detach().data
    output = torch.squeeze(output)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    '''
    # prepare input data for the first loop
    data_src.update(start_index)
    future_data = [data_src.train_data[0]]  # list
    ez_list.append(data_src.train_data[0])
    ez_list = ez_list[-2:]
    '''

    for index in range(1, 100):
        ez_list_output = [output.numpy()]
        input_list = ez_list_output.copy()
        for i in range(1,5):
            input_list.append(batch_sequencial_data[size, index, i, :, :])  # the property of medium does not change with time
        if index < 1:
            input = torch.tensor(np.array(batch_sequencial_data[size,index,:,:,:])[np.newaxis,np.newaxis,:,:,:], dtype=torch.float)
        else:
            input = torch.tensor(np.array(input_list)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)

        input = model.Variable(input)

        output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)

        h_t = h_t.detach().data
        c_t = c_t.detach().data
        h_t2 = h_t2.detach().data
        c_t2 = c_t2.detach().data

        # ground truth for prediction and next loop
        future_data = batch_sequencial_target_data[size, index, 0, :,:]

        # network output
        output = output.detach().data
        output = torch.squeeze(output)
        ax1.clear()
        ax2.clear()
        ax1.imshow(future_data,  interpolation='none')
        ax1.set_title("FDTD")
        ax2.imshow(output, interpolation='none')
        ax2.set_title("DNN")

        fig2.canvas.draw()  # draw the canvas, cache the renderer
        image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(image_from_plot, 'RGB')
        gif_images.append(img)

        plt.pause(0.1)



    imageio.mimsave(outfilename, gif_images, fps=10)

