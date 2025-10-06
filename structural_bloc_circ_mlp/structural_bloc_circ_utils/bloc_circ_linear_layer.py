import torch
import torch.nn as nn

class BlocCircLinear(nn.Module):
    def __init__(self, row_dim, col_dim, n_blocks, if_bias=False):
        super(BlocCircLinear, self).__init__()

        # Ensure the row and column dimensions are divisible by the number of blocks
        if row_dim % n_blocks != 0:
            raise ValueError("row_dim must be divisible by n_blocks.")
        if col_dim % n_blocks != 0:
            raise ValueError("col_dim must be divisible by n_blocks.")

        # Calculate the block sizes
        self.row_block_size = row_dim // n_blocks
        self.col_block_size = col_dim // n_blocks

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.n_blocks = n_blocks
        self.if_bias = if_bias
        
        # Consolidate the blocks into a single tensor for efficient processing
        ####### If I define blocks using a ParameterList, I cannot use blocks[indices] in forward
        self.blocks = nn.Parameter(
            torch.empty(n_blocks, self.row_block_size, self.col_block_size)
        )
        nn.init.xavier_uniform_(self.blocks)

        if self.if_bias:
            self.bias = nn.Parameter(torch.zeros(row_dim))
        else:
            self.register_parameter('bias', None)


    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the input

        # Reshape x into blocks
        x_blocks = x.view(batch_size, self.n_blocks, self.col_block_size)  # Shape: (batch_size, n_blocks, col_block_size)

        # Create an index tensor for the circular shifts
        indices = torch.arange(self.n_blocks).unsqueeze(1) - torch.arange(self.n_blocks).unsqueeze(0)
        indices = indices % self.n_blocks  # Apply modulo to get the correct circular shift indices

        # Gather the blocks according to the indices
        blocks_shifted = self.blocks[indices]  # Shape: (n_blocks, n_blocks, row_block_size, col_block_size)

        # Perform the block-wise multiplication
        y_blocks = torch.einsum('mnpq,bnq->bmp', blocks_shifted, x_blocks)
        # 'mnpq' represents the shape of blocks_shifted:
        #   - 'm': Refers to the first block dimension (corresponding to the circular shift index).
        #   - 'n': Refers to the second block dimension (aligns with the block index in x_blocks).
        #   - 'p': Refers to the output row dimension of the blocks.
        #   - 'q': Refers to the column dimension of the blocks, used for multiplication with the input.
        # 'bnq' represents the shape of x_blocks:
        #   - 'b': Refers to the batch dimension (each batch processed independently).
        #   - 'n': Refers to the block dimension (aligns with 'n' in blocks_shifted).
        #   - 'q': Refers to the vector dimension that aligns with 'q' in blocks_shifted.
        # The output 'bmp' means:
        #   - 'b': Batch size is retained in the output.
        #   - 'm': The circularly shifted result dimension (later summed over).
        #   - 'p': The output row dimension (from the matrix-vector multiplication).

        # Flatten y_blocks into a single vector y for each element in the batch
        y = y_blocks.view(batch_size, -1)  # Shape: (batch_size, row_dim)

        # Add bias if applicable
        if self.if_bias:
            y += self.bias

        return y


    ####### This is an inefficient implementation that you shouldn't use #######
    # def forward(self, x):
    #     # Initialize the full circulant matrix with zeros
    #     full_matrix = torch.zeros(self.row_dim, self.col_dim, device=x.device, dtype=x.dtype)

    #     # Fill the full circulant matrix with the blocks in a correct circulant pattern
    #     for i in range(self.n_blocks):
    #         for j in range(self.n_blocks):
    #             row_start = i * self.row_block_size
    #             col_start = j * self.col_block_size
    #             full_matrix[row_start:row_start+self.row_block_size, col_start:col_start+self.col_block_size] = self.blocks[(i-j) % self.n_blocks]

    #     output = torch.matmul(x, full_matrix.t())
    #     if self.if_bias:
    #         output += self.bias
    #     return output


    def get_full_circ_mat(self):
        # Initialize an empty tensor for the circulant matrix A
        A = torch.zeros(self.row_dim, self.col_dim)

        for i in range(self.n_blocks):
            # Calculate the row and column start indices for block placement
            row_start = i * self.row_block_size
            
            # Loop over each block and place them in the correct positions
            for j in range(self.n_blocks):
                col_start = j * self.col_block_size
                
                # Compute the index of the block to place
                block_index = (i - j) % self.n_blocks
                
                # Place the block in the appropriate position in A
                A[row_start:row_start + self.row_block_size, col_start:col_start + self.col_block_size] = self.blocks[block_index]

        return A


    def get_full_matrix(self):
        """
        For debug purposes
        Allows you to view the full matrix of the layer in the forward pass
        """
        full_matrix = torch.zeros(self.row_dim, self.col_dim)

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                row_start = i * self.row_block_size
                col_start = j * self.col_block_size
                full_matrix[row_start:row_start+self.row_block_size, col_start:col_start+self.col_block_size] = self.blocks[(i-j) % self.n_blocks]
        return full_matrix.detach().numpy()