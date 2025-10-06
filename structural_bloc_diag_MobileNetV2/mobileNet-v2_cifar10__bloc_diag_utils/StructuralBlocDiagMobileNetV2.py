import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re

from blocks import BaseBlock
from bloc_diag_utils import block_diagonal_zeros_torch, block_diagonal_ones_torch
from bloc_diag_linear_layer import BlocDiagLinear


class StructuralBlocDiagMobileNetV2(nn.Module):
    def __init__(self, layer_sizes, row_block_sizes_list, col_block_sizes_list, fc_if_bias, fc_nonlinearity, alpha = 1):
        super(StructuralBlocDiagMobileNetV2, self).__init__()
        self.layer_sizes = layer_sizes
        self.row_block_sizes_list = row_block_sizes_list # Sizes of the rows for each block for each layer
        self.col_block_sizes_list = col_block_sizes_list # Sizes of the columns for each block for each layer
        self.fc_if_bias = fc_if_bias
        self.fc_nonlinearity = fc_nonlinearity

        if not (len(layer_sizes)-1 == len(row_block_sizes_list) == len(col_block_sizes_list)):
            raise ValueError("layer_sizes-1, row_block_sizes_list, and col_block_sizes_list must have the same length.")
        
        # Check that each layer's output matches the next layer's input
        for i in range(1, len(col_block_sizes_list)):
            if sum(row_block_sizes_list[i-1]) != sum(col_block_sizes_list[i]):
                raise ValueError("The output size of layer {} must match the input size of layer {}.".format(i-1, i))


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
        # fully connected layers
        self.layers = nn.ModuleList()
        # append the weight matrices, as BlocDiagLinear layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BlocDiagLinear(layer_sizes[i+1], layer_sizes[i], row_block_sizes_list[i], col_block_sizes_list[i], if_bias=fc_if_bias))
            ### NOTICE: Here the col_dim and row_dim are different from Pytorch LinearLayer convention
            ### because in BlocDiagLinear, there is a transpose in the matmul when computing forward

        # weights init
        #self.weights_init()


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


    def set_all_layers_from_full_matrices(self, full_matrices):
        if len(full_matrices) != len(self.layers):
            raise ValueError("Number of full matrices must match the number of layers.")
        for layer, matrix in zip(self.layers, full_matrices):
            # Convert the numpy matrix to a torch tensor if it's not already one
            if isinstance(matrix, np.ndarray):
                matrix = torch.from_numpy(matrix).float()
            layer.set_blocks_from_full_matrix(matrix)



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