import argparse


class parameter(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
        parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
        parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
        parser.add_argument('--n_hidden_dim', type=int, default=32, help='number of hidden dim for ConvLSTM layers')
        parser.add_argument('--n_input_channels', type=int, default=1, help='number of input channel for ConvLSTM layers')
        parser.add_argument('--n_output_channels', type=int, default=1, help='number of output channel for ConvLSTM layers')
        parser.add_argument('--n_domain_dim', type=int, default=256, help='number of domain dimensions for simulation')
        parser.add_argument('--n_future_seq', type=int, default=1, help='number of domain dimensions for simulation')
        self.opt = parser.parse_args()
