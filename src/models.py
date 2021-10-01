import os
import json
import argparse
import time
from datetime import datetime

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

class VideoDCD(nn.Module):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, x_h=64, x_w=64, filter_size=128, j=16, nchannel=1, bn=False):
        super(VideoDCD, self).__init__()

        self.batch_size = batch_size
        self.filter_size = filter_size
        self.nchannel = nchannel
        self.ks = 6
        # j is the dimension of h and M
        self.j = j
        self.bn = bn
        self.x_height = x_h
        self.x_width = x_w
        self.output_dims = 0

        h_in = 8
        s = 2
        p = self.compute_padding(h_in, s, self.ks)
        k_size = [self.ks, self.ks]
        stride = [s, s]
        padding = [p, p]
        self.input_shape = [self.x_height, self.x_width]

        self.out_shape1 = self.compute_output_shape(self.input_shape, padding, k_size, stride)
        self.out_shape2 = self.compute_output_shape(self.out_shape1, padding, k_size, stride)
        self.out_shape3 = self.compute_output_shape(self.out_shape2, padding, k_size, stride)

        # conv layer 1
        conv_layers = [nn.Conv2d(self.nchannel, self.filter_size, kernel_size=k_size,
                                 stride=stride, padding=padding)]
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size))
        conv_layers.append(nn.LeakyReLU())

        # conv layer 2
        conv_layers.append(nn.Conv2d(self.filter_size, self.filter_size * 2, kernel_size=k_size,
                                     stride=stride, padding=padding))
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        conv_layers.append(nn.LeakyReLU())

        # conv layer 3
        conv_layers.append(nn.Conv2d(self.filter_size * 2, self.filter_size * 4, kernel_size=k_size,
                                     stride=stride, padding=padding))
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size * 4))
        conv_layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*conv_layers)

        # need to compute the shape of output from conv layers

        self.lstm1 = nn.LSTM(self.filter_size * 4 * self.out_shape3[0] * self.out_shape3[1],
                             self.filter_size * 4, batch_first=True)
        self.lstmbn = nn.BatchNorm1d(self.filter_size * 4)

        self.final1_lstm2 = nn.LSTM(self.filter_size * 4, self.j, batch_first=True)
        self.final2_lstm2 = nn.LSTM(self.filter_size * 4, self.j, batch_first=True)

    def compute_output_shape(self, h_in, padding, ksize, stride):
        out_h = torch.floor(torch.tensor((h_in[0] + 2.0 * padding[0] - ksize[0]) / stride[0])) + 1
        out_w = torch.floor(torch.tensor((h_in[1] + 2.0 * padding[1] - ksize[1]) / stride[1])) + 1
        return [int(out_h), int(out_w)]

    # padding computation when h_in = 2h_out
    def compute_padding(self, h_in, s, k_size):
        return max((h_in * (s - 2) - s + k_size) // 2, 0)

    def forward(self, inputs1, inputs2=None):
        time_steps = inputs1.shape[1]
        x = inputs1.reshape(self.batch_size * time_steps, self.nchannel, self.x_height, self.x_width)
        x = self.conv_net(x)
        x = x.reshape(self.batch_size, time_steps, -1)
        # first output dimension is the sequence of h_t.
        # second output is h_T and c_T(last cell state at t=T).
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn(x)
        x = x.permute(0, 2, 1)
        x1, _ = self.final1_lstm2(x)

        if inputs2 is not None:
            time_steps = inputs2.shape[1]
            x = inputs2.reshape(self.batch_size * time_steps, self.nchannel, self.x_height, self.x_width)
            x = self.conv_net(x)
            x = x.reshape(self.batch_size, time_steps, -1)
            # first output dimension is the sequence of h_t.
            # second output is h_T and c_T(last cell state at t=T).
            x, _ = self.lstm1(x)
            x = x.permute(0, 2, 1)
            if self.bn:
                x = self.lstmbn(x)
            x = x.permute(0, 2, 1)
            x2, _ = self.final2_lstm2(x)
            return x1, x2

        return x1


class VideoDCG(nn.Module):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size=8, time_steps=32, x_h=64, x_w=64, filter_size=32, state_size=32, nchannel=1, z_dim=25,
                 y_dim=20, bn=False, output_act='sigmoid'):
        super(VideoDCG, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.state_size = state_size
        self.nchannel = nchannel
        self.n_noise_t = z_dim
        self.n_noise_y = y_dim
        self.x_height = x_h
        self.x_width = x_w
        self.bn = bn
        self.output_activation = output_act

        self.lstm1 = nn.LSTM(self.n_noise_t + self.n_noise_y, self.state_size, batch_first=True)
        self.lstmbn1 = nn.BatchNorm1d(self.state_size)
        self.lstm2 = nn.LSTM(self.state_size, self.state_size * 2, batch_first=True)
        self.lstmbn2 = nn.BatchNorm1d(self.state_size * 2)

        # compute paddings
        if self.x_height == self.x_width:
            stride4 = [2, 2]
            k_size4 = [6, 6]
            h4_in = [self.x_height // 2, self.x_width // 2]
            h4_out = [self.x_height, self.x_width]
            padding4 = self.computing_padding(h4_out, h4_in, k_size4, stride4)

            stride3 = [2, 2]
            k_size3 = [6, 6]
            h3_in = [self.x_height // 4, self.x_width // 4]
            padding3 = self.computing_padding(h4_in, h3_in, k_size3, stride3)

            stride2 = [2, 2]
            k_size2 = [4, 4]
            h2_in = [self.x_height // 8, self.x_width // 8]
            padding2 = self.computing_padding(h3_in, h2_in, k_size2, stride2)

            stride1 = [2, 2]
            k_size1 = [2, 2]
            h_in = [8, 8]
            padding1 = self.computing_padding(h2_in, h_in, k_size1, stride1)
        elif self.x_height < self.x_width:
            stride4 = [2, 3]
            k_size4 = [8, 9]
            h4_in = [self.x_height // 2, self.x_width // 2]
            h4_out = [self.x_height, self.x_width]
            padding4 = self.computing_padding(h4_out, h4_in, k_size4, stride4)

            stride3 = [2, 3]
            k_size3 = [6, 7]
            h3_in = [self.x_height // 4, self.x_width // 4]
            padding3 = self.computing_padding(h4_in, h3_in, k_size3, stride3)

            stride2 = [2, 3]
            k_size2 = [6, 7]
            h2_in = [self.x_height // 8, self.x_width // 8]
            padding2 = self.computing_padding(h3_in, h2_in, k_size2, stride2)

            stride1 = [2, 3]
            k_size1 = [6, 7]
            h_in = [8, 8]
            padding1 = self.computing_padding(h2_in, h_in, k_size1, stride1)
        else:
            stride4 = [3, 2]
            k_size4 = [9, 8]
            h4_in = [self.x_height // 2, self.x_width // 2]
            h4_out = [self.x_height, self.x_width]
            padding4 = self.computing_padding(h4_out, h4_in, k_size4, stride4)

            stride3 = [3, 2]
            k_size3 = [7, 6]
            h3_in = [self.x_height // 4, self.x_width // 4]
            padding3 = self.computing_padding(h4_in, h3_in, k_size3, stride3)

            stride2 = [3, 2]
            k_size2 = [7, 6]
            h2_in = [self.x_height // 8, self.x_width // 8]
            padding2 = self.computing_padding(h3_in, h2_in, k_size2, stride2)

            stride1 = [3, 2]
            k_size1 = [7, 6]
            h_in = [8, 8]
            padding1 = self.computing_padding(h2_in, h_in, k_size1, stride1)

        dense_layers = [nn.Linear(self.state_size * 2, 8 * 8 * self.filter_size * 4)]
        if self.bn:
            dense_layers.append(nn.BatchNorm1d(8 * 8 * self.filter_size * 4))
        dense_layers.append(nn.LeakyReLU())

        self.dense_net = nn.Sequential(*dense_layers)

        # conv layer 1
        deconv_layers = [nn.ConvTranspose2d(self.filter_size * 4, self.filter_size * 4, kernel_size=k_size1,
                                            stride=stride1, padding=padding1)]

        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size * 4))
        deconv_layers.append(nn.LeakyReLU())

        # conv layer 2
        deconv_layers.append(nn.ConvTranspose2d(self.filter_size * 4, self.filter_size * 2,
                                                kernel_size=k_size2, stride=stride2, padding=padding2))

        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        deconv_layers.append(nn.LeakyReLU())

        # conv layer 3
        deconv_layers.append(nn.ConvTranspose2d(self.filter_size * 2, self.filter_size,
                                                kernel_size=k_size3, stride=stride3, padding=padding3))
        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size))
        deconv_layers.append(nn.LeakyReLU())

        # conv layer 4
        deconv_layers.append(nn.ConvTranspose2d(self.filter_size, self.nchannel, kernel_size=k_size4,
                                                stride=stride4, padding=padding4))

        if self.output_activation == 'sigmoid':
            deconv_layers.append(nn.Sigmoid())
        elif self.output_activation == 'tanh':
            deconv_layers.append(nn.Tanh())
        else:
            deconv_layers = deconv_layers

        self.deconv_net = nn.Sequential(*deconv_layers)

        # self.final1 = nn.Sequential(nn.ConvTranspose2d(self.filter_size, self.nchannel, kernel_size=k_size4,
        #                                         stride=stride4, padding=padding4))
        # self.final2 = nn.Sequential(nn.ConvTranspose2d(self.filter_size, self.nchannel, kernel_size=k_size4,
        #                                         stride=stride4, padding=padding4))

    def computing_padding(self, h_out, h_in, k_size, stride):
        p1 = torch.tensor(((h_in[0] - 1) * stride[0] - h_out[0] + k_size[0]) / 2)
        p2 = torch.tensor(((h_in[1] - 1) * stride[1] - h_out[1] + k_size[1]) / 2)
        padding_h = int(abs(torch.ceil(p1)))
        padding_w = int(abs(torch.ceil(p2)))
        return [padding_h, padding_w]

    def forward(self, z, y):
        z = z.reshape(self.batch_size, self.time_steps, self.n_noise_t)
        y = y[:, None, :].expand(self.batch_size, self.time_steps, self.n_noise_y)
        x = torch.cat([z, y], -1)
        # lstm input requires shape [batch, length, features]
        x, _ = self.lstm1(x)
        # batch norm layer input requires shape [batch, features, length]
        # pretty annoying
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn2(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(self.batch_size * self.time_steps, -1)
        x = self.dense_net(x)
        x = x.reshape(self.batch_size * self.time_steps, self.filter_size * 4, 8, 8)
        x = self.deconv_net(x)
        # x = self.final1(x)
        # x2 = self.final2(x)
        x = x.reshape(self.batch_size, self.time_steps, self.nchannel, self.x_height, self.x_width)
        # x2 = x2.reshape(self.batch_size, self.time_steps, self.x_height, self.x_width)
        return x