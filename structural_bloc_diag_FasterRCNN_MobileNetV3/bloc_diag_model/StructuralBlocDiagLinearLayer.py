import torch
import torch.nn as nn

class StructuralBlocDiagLinear(nn.Module):
    def __init__(self, row_dim, col_dim, row_block_sizes, col_block_sizes, if_bias=False):
        super().__init__()
        if len(row_block_sizes) != len(col_block_sizes):
            raise ValueError("Row and column block sizes must have the same number of elements.")

        # Validate dimensions against expected dimensions
        if sum(row_block_sizes) != row_dim:
            raise ValueError(f"Calculated total row_dim of the full_matrix, {sum(row_block_sizes)}, do not match expected {row_dim}.")
        if sum(col_block_sizes) != col_dim:
            raise ValueError(f"Calculated total col_dim of the full_matrix, {sum(col_block_sizes)}, do not match expected {col_dim}.")


        self.row_dim = row_dim  # Total rows in the full matrix
        self.col_dim = col_dim  # Total columns in the full matrix
        self.row_block_sizes = row_block_sizes
        self.col_block_sizes = col_block_sizes
        self.if_bias = if_bias
        self.blocks = nn.ParameterList(
            [nn.Parameter(torch.randn(row, col)) for row, col in zip(row_block_sizes, col_block_sizes)]
        )

        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(sum(row_block_sizes)))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        batch_size = x.size(0)

        weights = torch.stack(list(self.blocks), dim=0)  # (n_blocks, row_block_size, col_block_size)

        x_blocks = torch.split(x, self.col_block_sizes, dim=1)  # list of (batch_size, col_block_size)
        x_stacked = torch.stack(x_blocks, dim=1)  # (batch_size, n_blocks, col_block_size)

        outputs = torch.einsum('bni,nio->bno', x_stacked, weights.transpose(1, 2))  # (batch_size, n_blocks, row_block_size)
        output = outputs.reshape(batch_size, -1) # (batch_size, row_dim)

        if self.if_bias:
            output += self.bias
        return output


    def get_full_matrix(self):
        """
        For debug purposes
        Allows you to view the full matrix of the layer in the forward pass
        """
        total_rows = sum(self.row_block_sizes)
        total_cols = sum(self.col_block_sizes)
        full_matrix = torch.zeros(total_rows, total_cols)

        row_index = 0
        col_index = 0
        for block in self.blocks:
            block_rows, block_cols = block.size()
            full_matrix[row_index:row_index + block_rows, col_index:col_index + block_cols] = block.detach()
            row_index += block_rows
            col_index += block_cols
        return full_matrix.numpy()


    def set_blocks_from_full_matrix(self, full_matrix, input_bias):
        row_index = 0
        col_index = 0
        new_blocks = []
        for row_size, col_size in zip(self.row_block_sizes, self.col_block_sizes):
            # Extract the block from the full matrix
            block = full_matrix[row_index:row_index + row_size, col_index:col_index + col_size]
            new_blocks.append(nn.Parameter(block))
            row_index += row_size
            col_index += col_size
        self.blocks = nn.ParameterList(new_blocks)
        
        if self.if_bias and input_bias is not None:
            self.bias.data.copy_(input_bias)