import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from bloc_circ_linear_layer import BlocCircLinear

class StructuralBlocCircModel(nn.Module):
    def __init__(self, layer_sizes, n_blocks_list, if_bias, nonlinearity):
        """
        Initializes a model composed of multiple BlockCirculantLinear layers.

        Args:
            layer_sizes (list of int): List containing the sizes of each layer.
            n_blocks_list (list of int): List containing the number of blocks for each layer.
            if_bias (bool): If True, adds a bias to each BlockCirculantLinear layer.
            nonlinearity (str): Specifies the nonlinearity to apply (e.g., 'SiLU', 'ReLU').
        """
        super(StructuralBlocCircModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.n_blocks_list = n_blocks_list
        self.if_bias = if_bias
        self.nonlinearity = nonlinearity
        
        if len(layer_sizes) - 1 != len(n_blocks_list):
            raise ValueError("layer_sizes-1 and n_blocks_list must have the same length.")

        # Append the weight matrices, as BlockCirculantLinear layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                BlocCircLinear(
                    row_dim=layer_sizes[i+1],
                    col_dim=layer_sizes[i],
                    n_blocks=n_blocks_list[i],
                    if_bias=if_bias
                )
            )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)

            if re.search('every_layer_except_last', self.nonlinearity, re.IGNORECASE) is not None:
                if layer is not self.layers[-1]:  # Add nonlinearity on every layer except the last one
                    if re.search('SiLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("Unknown nonlinearity")

            if re.search('only_first_layer', self.nonlinearity, re.IGNORECASE) is not None:
                if layer is self.layers[0]:  # Add nonlinearity ONLY on the 1st layer
                    if re.search('SiLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("Unknown nonlinearity")

        return x