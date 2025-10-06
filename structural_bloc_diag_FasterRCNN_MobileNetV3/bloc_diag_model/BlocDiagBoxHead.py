import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import re



class BlocDiagBoxHead(nn.Module):
    def __init__(self, layer_sizes, row_blocks, col_blocks, if_bias=True):
        super().__init__()
        if len(layer_sizes) != 3:
            raise ValueError(f"BoxHead should have `len(layer_sizes)` = 3, but got {len(layer_sizes)}.")

        self.row_blocks = row_blocks # Sizes of the rows for each block for each layer
        self.col_blocks = col_blocks # Sizes of the columns for each block for each layer

        self.fc6 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=if_bias)
        self.fc7 = nn.Linear(layer_sizes[1], layer_sizes[2], bias=if_bias)
        self.layers = nn.ModuleList([self.fc6, self.fc7]) # stores references to fc6 and fc7

    def forward(self, x):
        x = x.flatten(start_dim=1)
    
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x



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
                off_diag_mask_matrix = block_diagonal_zeros_torch(
                    self.row_blocks[i], self.col_blocks[i], device=self.layers[i].weight.device)
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
                diag_mask_matrix = block_diagonal_ones_torch(
                    self.row_blocks[i], self.col_blocks[i], device=self.layers[i].weight.device)
                model_off_diag_dropped.layers[i].weight.data = model_off_diag_dropped.layers[i].weight.data * diag_mask_matrix

        return model_off_diag_dropped


def block_diagonal_zeros_torch(row_sizes, col_sizes, device):
    """
    Create a block diagonal matrix with blocks of zeros, where each block has a different size.

    Parameters:
    row_sizes (list of int): List specifying the number of rows for each block.
    col_sizes (list of int): List specifying the number of columns for each block.

    Returns:
    torch.Tensor: The resulting block diagonal matrix.
    """
    if len(row_sizes) != len(col_sizes):
        raise ValueError("row_sizes and col_sizes must have the same length")

    # the total size of the final matrix
    total_rows = sum(row_sizes)
    total_cols = sum(col_sizes)

    large_matrix = torch.ones(total_rows, total_cols, device=device)

    # fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        large_matrix[start_row:end_row, start_col:end_col] = torch.zeros(row_sizes[i], col_sizes[i])
        start_row = end_row

    return large_matrix


def block_diagonal_ones_torch(row_sizes, col_sizes, device):
    """
    Create a block diagonal matrix with blocks of ones, where each block has a different size.

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
    large_matrix = torch.zeros(total_rows, total_cols, device=device)

    # Fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        large_matrix[start_row:end_row, start_col:end_col] = torch.ones(row_sizes[i], col_sizes[i])
        start_row = end_row

    return large_matrix