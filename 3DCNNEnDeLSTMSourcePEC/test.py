import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
import parameters as param
import torch
import fdtd
import numpy as np
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image




if __name__ == '__main__':
    param = param.parameter()
    fig1 = plt.figure("FDTD vs ConvRNN")
    gif_images = []
    outfilename = "FDTD_vs_DNN.gif"  # 转化的GIF图片名称

    conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=param.opt.n_input_channels).cuda()

    data_src = fdtd.Grid(param.opt.n_domain_dim)
    ez_list = [0, 0]
    input_list = []
    for prepare_index in range(50):
        data_src.update(prepare_index)
        ez_list.append(data_src.train_data[0])
        ez_list = ez_list[-2:]  # update to the recent 2
        input_list = ez_list.copy()
        input_list.extend(data_src.train_data[-2:])

    ez_list_output = ez_list.copy()
    start_index = prepare_index + 1
    print(np.array(input_list).shape)

    input = model.Variable(
        torch.tensor(np.array(input_list)[np.newaxis, np.newaxis, :,  :, :], dtype=torch.float)).cuda()
    print(input.size())
    b, _, ch, h, w = input.size()
    lstm_h = h
    lstm_w = w
    h_t, c_t = conv_lstm_model.encoder_1_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))
    h_t2, c_t2 = conv_lstm_model.decoder_4_convlstm.init_hidden(batch_size=b,  shape=(lstm_h, lstm_w))

    fig2 = plt.figure("imput data and output data")

    conv_lstm_model.load_state_dict(torch.load('mymodule_params.pt'))
    conv_lstm_model.eval()
    conv_lstm_model.cuda()

    output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)
    output = output.cpu().detach().data
    output = torch.squeeze(output)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # prepare input data for the first loop
    data_src.update(start_index)
    future_data = [data_src.train_data[0]]  # list
    ez_list.append(data_src.train_data[0])
    ez_list = ez_list[-2:]

    for index in range(start_index+1, 130):

        ez_list_output.append(output.numpy())
        ez_list_output = ez_list_output[-2:]  # update to the recent 2
        #print(np.array(ez_list_output).shape)
        input_list = ez_list_output.copy()
        input_list.extend(data_src.train_data[-2:])  # the property of medium does not change with time
        #print(np.array(input_list).shape)
        if index < 1:
            input = torch.cat([torch.tensor(np.array(ez_list)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float),
                            torch.tensor(np.array(data_src.train_data[-2:])[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)]
                            , dim=2)
        else:
            input = torch.tensor(np.array(input_list)[np.newaxis, np.newaxis, :, :, :], dtype=torch.float)

        #print(output.size())
        # print(np.array(self.input_data).shape, np.array(self.future_data).shape)
        input = model.Variable(input).cuda()
        #print(input.size())
        output, h_t, c_t, h_t2, c_t2 = conv_lstm_model(input, h_t, c_t, h_t2, c_t2)

        h_t = h_t.detach().data
        c_t = c_t.detach().data
        h_t2 = h_t.detach().data
        c_t2 = c_t.detach().data

        # ground truth for prediction and next loop
        data_src.update(index)
        future_data = [data_src.train_data[0]]
        ez_list.append(data_src.train_data[0])
        ez_list = ez_list[-2:]

        # network output
        output = output.cpu().detach().data
        output = torch.squeeze(output)
        ax1.imshow(future_data[0],  interpolation='none')
        ax1.set_title("FDTD")
        ax2.imshow(output, interpolation='none')
        ax2.set_title("DNN")
        plt.pause(0.1)


        fig1.canvas.draw()  # draw the canvas, cache the renderer
        image_from_plot = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(image_from_plot, 'RGB')
        gif_images.append(img)

    imageio.mimsave(outfilename, gif_images, fps=10)

