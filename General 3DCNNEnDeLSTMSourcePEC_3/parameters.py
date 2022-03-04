import argparse


class parameter(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--max_pool_kernal_size', default=2, type=int, help='max_pool_kernal_size')
        parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
        parser.add_argument('--beta_2', type=float, default=0.999, help='decay rate 2')
        parser.add_argument('--batch_size', default=5, type=int, help='batch size')
        parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
        parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
        # hidden_size，也就是隐状态的维度，神经元的个数，而不是cell数目。至于这个大小设置，应该是来自计算力与表达能力的权衡。越多的hidden_size可以包含更多的细节，以及更丰富的表达能力，但是同时也会带来过拟合以及耗时间等缺点
        parser.add_argument('--n_hidden_dim', type=int, default=96, help='number of hidden dim for ConvLSTM layers')
        parser.add_argument('--n_input_channels', type=int, default=2, help='number of input channel for ConvLSTM layers')
        parser.add_argument('--n_output_channels', type=int, default=1, help='number of output channel for ConvLSTM layers')
        parser.add_argument('--n_domain_dim', type=int, default=80, help='number of domain dimensions for simulation')
        parser.add_argument('--n_future_seq', type=int, default=3, help='number of domain dimensions for simulation')
        parser.add_argument('--simulation_length', type=int, default=1, help='time of fdtd iteration to be simulated')
        self.opt = parser.parse_args()
