import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re



class BlocDiagModel(nn.Module):
    def __init__(self, layer_sizes, row_blocks, col_blocks, if_bias, nonlinearity):
        super(BlocDiagModel, self).__init__()
        self.layers = nn.ModuleList()
        self.row_blocks = row_blocks # Sizes of the rows for each block for each layer
        self.col_blocks = col_blocks # Sizes of the columns for each block for each layer
        self.if_bias = if_bias
        self.nonlinearity = nonlinearity

        # Create layers based on the sizes provided in layer_sizes
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=if_bias))

    def get_off_diag_loss(self):
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
            off_diag_mask_matrix = block_diagonal_zeros_torch(self.row_blocks[i], self.col_blocks[i])
            off_diag_weights.append(self.layers[i].weight * off_diag_mask_matrix)

        off_diag_loss = 0
        for off_diag_weight_mat in off_diag_weights:
          off_diag_loss = off_diag_loss + torch.sum(torch.abs(off_diag_weight_mat))

        return off_diag_loss


    def copy_model_drop_off_diag(self):
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
            diag_mask_matrix = block_diagonal_ones_torch(self.row_blocks[i], self.col_blocks[i])
            model_off_diag_dropped.layers[i].weight.data = model_off_diag_dropped.layers[i].weight.data * diag_mask_matrix

        return model_off_diag_dropped


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
                if layer is self.layers[1]: # add nonlinearity ONLY on the 1st layer

                    if re.search('SiLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.silu(x)
                    elif re.search('ReLU', self.nonlinearity, re.IGNORECASE) is not None:
                        x = F.relu(x)
                    else:
                        raise ValueError("unknown nonlinearity")

        return x


def block_diagonal_zeros_torch(row_sizes, col_sizes):
    """
    Create a block diagonal matrix with blocks of ones, where each block has a different size, using PyTorch.

    Parameters:
    row_sizes (list of int): List specifying the number of rows for each block.
    col_sizes (list of int): List specifying the number of columns for each block.

    Returns:
    torch.Tensor: The resulting block diagonal matrix.
    """
    if len(row_sizes) != len(col_sizes):
        raise ValueError("row_sizes and col_sizes must have the same length")

    # Calculate the total size of the final matrix
    total_rows = sum(row_sizes)
    total_cols = sum(col_sizes)

    # Initialize a large matrix of zeros
    large_matrix = torch.ones(total_rows, total_cols)

    # Fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        large_matrix[start_row:end_row, start_col:end_col] = torch.zeros(row_sizes[i], col_sizes[i])
        start_row = end_row

    return large_matrix


def block_diagonal_ones_torch(row_sizes, col_sizes):
    """
    Create a block diagonal matrix with blocks of ones, where each block has a different size, using PyTorch.

    Parameters:
    row_sizes (list of int): List specifying the number of rows for each block.
    col_sizes (list of int): List specifying the number of columns for each block.

    Returns:
    torch.Tensor: The resulting block diagonal matrix.
    """
    if len(row_sizes) != len(col_sizes):
        raise ValueError("row_sizes and col_sizes must have the same length")

    # Calculate the total size of the final matrix
    total_rows = sum(row_sizes)
    total_cols = sum(col_sizes)

    # Initialize a large matrix of zeros
    large_matrix = torch.zeros(total_rows, total_cols)

    # Fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        large_matrix[start_row:end_row, start_col:end_col] = torch.ones(row_sizes[i], col_sizes[i])
        start_row = end_row

    return large_matrix