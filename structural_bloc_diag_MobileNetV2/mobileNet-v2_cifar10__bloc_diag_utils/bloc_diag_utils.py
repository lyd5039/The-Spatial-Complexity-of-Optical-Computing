# -*- coding: utf-8 -*-
import torch


def translate_nonzero_blocks(input_str):
    string_nonzero_blocks = input_str.split(' | ')
    string_nonzero_blocks = [expr.split(', ') for expr in string_nonzero_blocks]
    string_nonzero_blocks = [item for sublist in string_nonzero_blocks for item in sublist] # Now, string_nonzero_blocks is a list of separated strings
    row_blocks = [eval(expr) for expr in string_nonzero_blocks[::2]] # even indexed elements
    col_blocks = [eval(expr) for expr in string_nonzero_blocks[1::2]] # odd indexed elements

    return row_blocks, col_blocks


def block_diagonal_zeros_torch(row_sizes, col_sizes, device):
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

    mat_diag_zeros = torch.ones(total_rows, total_cols, device=device)

    # Fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        mat_diag_zeros[start_row:end_row, start_col:end_col] = torch.zeros(row_sizes[i], col_sizes[i])
        start_row = end_row

    return mat_diag_zeros


def block_diagonal_ones_torch(row_sizes, col_sizes, device):
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

    mat_diag_ones = torch.zeros(total_rows, total_cols, device=device)

    # Fill the diagonal blocks
    start_row = 0
    for i in range(len(row_sizes)):
        end_row = start_row + row_sizes[i]
        start_col = sum(col_sizes[:i])
        end_col = start_col + col_sizes[i]
        mat_diag_ones[start_row:end_row, start_col:end_col] = torch.ones(row_sizes[i], col_sizes[i])
        start_row = end_row

    return mat_diag_ones