from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from simsg.capsule import *
import numpy as np



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.ch = 16
        self.dropout = nn.Dropout2d(p=args['drop_out'])
        self.leakyrelu = nn.LeakyReLU()
        if args['noise_source'] == 'input':
            self.fc = nn.Linear(args['noise_size'], args['image_size']*args['image_size'])
        if args['noise_source'] == 'broadcast_conv':
            self.conv_noise = nn.Conv2d(in_channels=args['noise_size'], out_channels=1, kernel_size=5, padding=2, stride=1)
        if (args['noise_source'] == 'dropout') or (args['noise_source'] == 'broadcast_latent'):
            # in_channels=1->3
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.ch, kernel_size=5, padding=2, stride=1)
        else:
            # in_channels=2->4
            self.conv1 = nn.Conv2d(in_channels=4, out_channels=self.ch, kernel_size=5, padding=2, stride=1)


        self.convcaps1 = convolutionalCapsule(in_capsules=1, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps2 = convolutionalCapsule(in_capsules=2, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps3 = convolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch, out_channels=self.ch * 2,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps4 = convolutionalCapsule(in_capsules=4, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])
        self.convcaps5   = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 4,
                                              stride=2, padding=2, kernel=5,
                                              nonlinearity=args['caps_nonlinearity'], batch_norm=args['batch_norm'],
                                              dynamic_routing=args['dynamic_routing'], cuda=args['cuda'])

    def forward(self, x, noise, args):
        batch_size = x.size(0)

        # whether to add noise as input/ broadcast
        if args['noise_source'] == 'input':
            noise = self.fc(noise)
            noise = noise.view(batch_size, 1, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast_conv':
            noise = self.leakyrelu(self.conv_noise(noise))
            x = torch.cat([x, noise], dim=1)
        elif args['noise_source'] == 'broadcast':
            x = torch.cat([x, noise], dim=1)

        if args['gan_nonlinearity'] == 'leakyRelu':
            x_1 = self.leakyrelu(self.conv1(x))
        else:
            x_1 = F.relu(self.conv1(x))

        x_1 = x_1.view(batch_size, 1, self.ch, x_1.size(2), x_1.size(3))
        x = self.convcaps1(x_1)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x_2 = self.convcaps2(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_2 = self.dropout(x_2)
        x = self.convcaps3(x_2)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        x_3 = self.convcaps4(x)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x_3 = self.dropout(x_3)
        x = self.convcaps5(x_3)
        if (args['drop_out_train']) and (args['state'] == 'train'):
            x = self.dropout(x)
        # x_out = x.view(batch_size, self.ch, x.size(3), x.size(4))

        # tanh or sigmoid
        if args['gan_last_nonlinearity'] == 'tanh':
            out =torch.tanh(self.conv2(x_out))
        else:
            out =torch.sigmoid(self.conv2(x_out))

        return out #, x_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # TODO
        pass

    def forward(self):
        # TODO
        pass
