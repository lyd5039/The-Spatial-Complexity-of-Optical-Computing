import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from bloc_diag_linear_layer import BlocDiagLinear

class StructuralBlocDiagModel(nn.Module):
    def __init__(self, layer_sizes, row_block_sizes_list, col_block_sizes_list, if_bias, nonlinearity):
        """
        Initializes a model composed of multiple BlocDiagLinear layers.

        Args:
            row_block_sizes_list (list of lists): Each sublist contains the row sizes for the blocks of a particular layer.
            col_block_sizes_list (list of lists): Each sublist contains the column sizes for the blocks of a particular layer.
            if_bias (bool): If True, adds a bias to each BlocDiagLinear layer.
        """
        super(StructuralBlocDiagModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.row_block_sizes_list = row_block_sizes_list
        self.col_block_sizes_list = col_block_sizes_list
        self.if_bias = if_bias
        self.nonlinearity = nonlinearity
        
        if not (len(layer_sizes)-1 == len(row_block_sizes_list) == len(col_block_sizes_list)):
            raise ValueError("layer_sizes-1, row_block_sizes_list, and col_block_sizes_list must have the same length.")
        
        # Check that each layer's output matches the next layer's input
        for i in range(1, len(col_block_sizes_list)):
            if sum(row_block_sizes_list[i-1]) != sum(col_block_sizes_list[i]):
                raise ValueError("The output size of layer {} must match the input size of layer {}.".format(i-1, i))

        # append the weight matrices, as BlocDiagLinear layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BlocDiagLinear(layer_sizes[i+1], layer_sizes[i], row_block_sizes_list[i], col_block_sizes_list[i], if_bias))


    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)

            if re.search('every_layer_except_last', self.nonlinearity, re.IGNORECASE) is not None:
                if layer is not self.layers[-1]: # add nonlinearity on every layer except the last one

                    if re.search('SiLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("unknown nonlinearity")

            if re.search('only_first_layer', self.nonlinearity, re.IGNORECASE) is not None:
                if layer is self.layers[0]: # add nonlinearity ONLY on the 1st layer

                    if re.search('SiLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("unknown nonlinearity")

        return x


    def set_all_layers_from_full_matrices(self, full_matrices):
        if len(full_matrices) != len(self.layers):
            raise ValueError("Number of full matrices must match the number of layers.")
        for layer, matrix in zip(self.layers, full_matrices):
            # Convert the numpy matrix to a torch tensor if it's not already one
            if isinstance(matrix, np.ndarray):
                matrix = torch.from_numpy(matrix).float()
            layer.set_blocks_from_full_matrix(matrix)