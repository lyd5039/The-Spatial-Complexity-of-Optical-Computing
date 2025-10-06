import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .StructuralBlocDiagLinearLayer import StructuralBlocDiagLinear



class StructuralBlocDiagBoxHead(nn.Module):
    def __init__(self, layer_sizes, row_blocks, col_blocks, if_bias=True):
        super().__init__()
        if len(layer_sizes) != 3:
            raise ValueError(f"BoxHead should have `len(layer_sizes)` = 3, but got {len(layer_sizes)}.")

        self.row_blocks = row_blocks # Sizes of the rows for each block for each layer
        self.col_blocks = col_blocks # Sizes of the columns for each block for each layer

        self.fc6 = StructuralBlocDiagLinear(
            row_dim=layer_sizes[1], col_dim=layer_sizes[0],
            row_block_sizes=row_blocks[0], col_block_sizes=col_blocks[0],
            if_bias=if_bias)
        self.fc7 = StructuralBlocDiagLinear(
            row_dim=layer_sizes[2], col_dim=layer_sizes[1],
            row_block_sizes=row_blocks[1], col_block_sizes=col_blocks[1],
            if_bias=if_bias)
        self.layers = nn.ModuleList([self.fc6, self.fc7])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x



    def set_all_layers_from_full_matrices(self, full_matrices):
        if len(full_matrices) != len(self.layers):
            raise ValueError("Number of full matrices must match the number of layers.")

        for layer, matrix in zip(self.layers, full_matrices):
            # Convert the numpy matrix to a torch tensor if it's not already one
            if isinstance(matrix, np.ndarray):
                matrix = torch.from_numpy(matrix).float()
            layer.set_blocks_from_full_matrix(matrix)