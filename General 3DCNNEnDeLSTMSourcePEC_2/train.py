
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import convolution_lstm_cuda as model
import matplotlib.pyplot as plt
fig1 = plt.figure("FDTD vs ConvRNN")
import parameters as param
param = param.parameter()
import matplotlib
#matplotlib.use('TKAgg')
import h5py

ax1 = plt.subplot(111)
#ax2 = plt.subplot(142)
#ax3 = plt.subplot(143)
#ax4 = plt.subplot(144)

conv_lstm_model = model.EncoderDecoderConvLSTM(nf=param.opt.n_hidden_dim, in_chan=param.opt.n_input_channels).cuda()
optimizer = torch.optim.Adam(conv_lstm_model.parameters(), lr=param.opt.lr,
                             betas=(param.opt.beta_1, param.opt.beta_2))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = torch.nn.L1Loss(reduction='sum')

class FdtdDataSet(torch.utils.data.Dataset):
    def __getitem__(self, index):
        data = self.input[index]
        labels = self.target[index]
        return data, labels

    def __len__(self):
        return self.size


class CircularPEC(FdtdDataSet):
    def __init__(self, data_root, data_label, data_size):
        self.input = data_root
        self.target = data_label
        self.size = data_size

def gpu_info():
    print("GPU Information:")
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

if __name__ == '__main__':
    gpu_info()
    log_dir = "mymodule_params.pt"
    trained = 0

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        conv_lstm_model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Start from epoch {}.'.format(start_epoch))
    else:
        start_epoch = 0
        print('Start from beginning.')

    with h5py.File("CircularPEC_Dataset.hdf5", 'r') as f:  # read high dimension data
        print(f.keys())
        batch_sequencial_data = f.get("batch_sequencial_data")[:]
        batch_sequencial_target_data = f.get("batch_sequencial_target_data")[:]

    total_batch_size = np.size(batch_sequencial_target_data, 0)
    print(batch_sequencial_data.shape, batch_sequencial_target_data.shape, total_batch_size)
    sequancial_length = np.size(batch_sequencial_target_data, 1)

    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = CircularPEC(batch_sequencial_data, batch_sequencial_target_data, total_batch_size)  #

    #datas = DataLoader(torch_data, batch_size=10, shuffle=True, drop_last=False, num_workers=1)  # 组合器
    count_loss = []

    for epoch in range(start_epoch, param.opt.epochs):
        total_loss = 0
        datas = DataLoader(torch_data, batch_size=param.opt.batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory = False)  # 组合器
        for i, (input, target) in enumerate(datas):
            b, s, ch, h, w = input.size()  # batch, seq, channel, height, width
            h1, c1 = conv_lstm_model.lstm1.init_hidden(batch_size=b, shape=(h, w))
            h2, c2 = conv_lstm_model.lstm2.init_hidden(batch_size=b, shape=(h, w))

            for j in range(sequancial_length-param.opt.n_future_seq):
                #print(j)
                seq_input = input.float()[:, j:j+param.opt.n_future_seq, :, :, :]
                seq_target = target.float()[:, j:j+param.opt.n_future_seq, :, :, :]
                seq_input = model.Variable(seq_input).cuda()
                seq_target = model.Variable(seq_target).cuda()

                output, h1, c1, h2, c2= conv_lstm_model(seq_input, h1, c1, h2, c2)
                h1 = h1.detach().data
                c1 = c1.detach().data
                h2 = h2.detach().data
                c2 = c2.detach().data

                loss = criterion(output[:, :, :, :, :], seq_target[:, :, :, :, :]) / seq_target.shape[0]
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(conv_lstm_model.parameters(), 10)
                optimizer.step()
                loss.item()
                total_loss += loss.item()

        scheduler.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, total_loss/(i+1)))
        count_loss.append(total_loss/(i+1))
        # 保存模型
        state = {'model': conv_lstm_model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)

    torch.save(conv_lstm_model, 'mymodule_model.pt')  # all
    plt.plot(count_loss)
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig('loss.png', bbox_inches='tight')
    plt.show()