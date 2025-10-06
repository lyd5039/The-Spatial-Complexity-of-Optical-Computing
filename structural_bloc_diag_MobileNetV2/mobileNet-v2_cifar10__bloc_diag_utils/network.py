import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re

from blocks import BaseBlock
from bloc_diag_utils import block_diagonal_zeros_torch, block_diagonal_ones_torch


class MobileNetV2(nn.Module):
    def __init__(self, layer_sizes, row_blocks, col_blocks, fc_if_bias, fc_nonlinearity, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.row_blocks = row_blocks # Sizes of the rows for each block for each layer
        self.col_blocks = col_blocks # Sizes of the columns for each block for each layer
        self.fc_if_bias = fc_if_bias
        self.fc_nonlinearity = fc_nonlinearity

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), layer_sizes[0], kernel_size = 1, bias = False) # default value of layer_sizes[0] = 1280
        self.bn1 = nn.BatchNorm2d(layer_sizes[0])
        ### fully connected layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=fc_if_bias))

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = False)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = False)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = layer(x)
            
            if re.search('every_layer_except_last', self.fc_nonlinearity, re.IGNORECASE) is not None:
                if layer is not self.layers[-1]: # add nonlinearity on every layer except the last one

                    if re.search('SiLU', self.fc_nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.fc_nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("unknown nonlinearity")

            elif re.search('only_first_layer', self.fc_nonlinearity, re.IGNORECASE) is not None:
                if layer is self.layers[1]: # add nonlinearity ONLY on the 1st layer

                    if re.search('SiLU', self.fc_nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.fc_nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("unknown nonlinearity")
            
            elif self.fc_nonlinearity == 'none':
                pass

        return x


    def get_off_diag_loss(self, device):
        off_diag_weights = []
        for i in range(len(self.layers)):
          if i >= len(self.row_blocks) or i >= len(self.col_blocks):
            # if the last few block-diagonalization are not specified
            # the last few layers' weights are not block-diagonalized
            pass
          elif self.row_blocks[i] == [] or self.col_blocks[i] == []:
            # if the block-diagonalization for a specific layer is not specified
            # that layer is not block-diagonalized
            pass
          else:
            off_diag_mask_matrix = block_diagonal_zeros_torch(self.row_blocks[i], self.col_blocks[i], device)
            off_diag_weights.append(self.layers[i].weight * off_diag_mask_matrix)

        off_diag_loss = 0
        for off_diag_weight_mat in off_diag_weights:
          off_diag_loss = off_diag_loss + torch.sum(torch.abs(off_diag_weight_mat))

        return off_diag_loss


    def copy_model_drop_off_diag(self, device):
        model_off_diag_dropped = copy.deepcopy(self)
        for i in range(len(self.layers)):
          if i >= len(self.row_blocks) or i >= len(self.col_blocks):
            # if the last few block-diagonalization are not specified
            # the last few layers' weights are not block-diagonalized
            pass
          elif self.row_blocks[i] == [] or self.col_blocks[i] == []:
            # if the block-diagonalization for a specific layer is not specified
            # that layer is not block-diagonalized
            pass
          else:
            diag_mask_matrix = block_diagonal_ones_torch(self.row_blocks[i], self.col_blocks[i], device)
            model_off_diag_dropped.layers[i].weight.data = model_off_diag_dropped.layers[i].weight.data * diag_mask_matrix

        return model_off_diag_dropped



if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from count import measure_model
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    print(x.shape)

    net = MobileNetV2(10, alpha = 1)
    y = net(x)

    print(x.shape)
    print(y.shape)

    f, c = measure_model(net, 32, 32)
    print("model size %.4f M, ops %.4f M" %(c/1e6, f/1e6))

    # size = 1
    # for param in net.parameters():
    #     arr = np.array(param.size())
        
    #     s = 1
    #     for e in arr:
    #         s *= e
    #     size += s

    # print("all parameters %.2fM" %(size/1e6) )