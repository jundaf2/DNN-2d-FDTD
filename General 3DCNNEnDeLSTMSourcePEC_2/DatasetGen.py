import torch
import numpy as np
import fdtd
import parameters as param
import h5py
import matplotlib.pyplot as plt
param = param.parameter()

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

def get_batch(source, i = 0):
    seq_len = param.opt.simulation_length
    data = source[i:i+seq_len] # [ seq_len * batch_size * feature_size ]
    target = source[i+1:i+1+seq_len] # [ (seq_len x batch_size x feature_size) ]
    return data, target

from torch.utils.data import DataLoader
def DataGenerator(size):
    epsr = 1
    sigma = 1e4
    radius = np.linspace(1, 20, size)
    ez_list = [0, 0]
    i = 0
    batch_sequencial_data = []
    batch_sequencial_target_data = []
    for obj_type in ["circle"]: #, "square"]:
        for r in radius:

            sequencial_data = []
            sequencial_target_data = []
            data_src = fdtd.Grid(param.opt.n_domain_dim, epsr=epsr, sigma=sigma, radius=r, obj_type=obj_type)
            for prepare_index in range(50):
                data_src.update(prepare_index)
                #ez_list.append(data_src.train_data[0])
                #ez_list = ez_list[-1:]  # update to the recent 2
            #input_list = ez_list.copy()
            #input_list.extend(data_src.train_data[-1:])
            for index in range(prepare_index + 1, 150):
                sequencial_data.append(data_src.train_data)
                data_src.update(index)
                sequencial_target_data.append(data_src.train_data[0])


            batch_sequencial_data.append(sequencial_data)
            batch_sequencial_target_data.append(sequencial_target_data)
            i += 1
            print("{}% complete".format(100*i/(2*size)))

    batch_sequencial_data = np.array(batch_sequencial_data)
    batch_sequencial_target_data = np.array(batch_sequencial_target_data)[:,:, np.newaxis,:,:]

    with h5py.File("CircularPEC_Dataset.hdf5", 'w') as f:  # write high dimension data
        f.create_dataset("batch_sequencial_data", data=batch_sequencial_data, compression="gzip", compression_opts=5)
        f.create_dataset("batch_sequencial_target_data", data=batch_sequencial_target_data, compression="gzip",
                         compression_opts=5)



if __name__ == '__main__':

    DataGenerator(100)

    with h5py.File("CircularPEC_Dataset.hdf5", 'r') as f:  # read high dimension data
        print(f.keys())
        batch_sequencial_data = f.get("batch_sequencial_data")[:]
        batch_sequencial_target_data = f.get("batch_sequencial_target_data")[:]

    total_batch_size = np.size(batch_sequencial_target_data, 0)
    print(batch_sequencial_data.shape, batch_sequencial_target_data.shape, total_batch_size)

    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = CircularPEC(batch_sequencial_data, batch_sequencial_target_data, total_batch_size)  #

    datas = DataLoader(torch_data, batch_size=10, shuffle=True, drop_last=False, num_workers=2)  #组合器
    #fig = plt.figure()
    #ax = plt.subplot(111)
    for i, (input, target) in enumerate(datas):
        # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
        print("第 {} 个Batch \n{}{}".format(i, input.size(), target.size()))

        #b, s, ch, h, w = input.size()
        #for i in range(s):
            #ax.clear()
            #ax.imshow(input.data[0,i,0,:,:], interpolation='none')
            #plt.pause(0.1)

