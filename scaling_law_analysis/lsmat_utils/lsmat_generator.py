import numpy as np
from lsmat_to_graph import get_ports_pos



def random_sample_list_idx_satisfied(list_idx_satisfied, N_nonzero, rand_seed):
  np.random.seed(rand_seed) ### fix random seed

  array_idx_satisfied = np.array(list_idx_satisfied)

  # Check if N_nonzero is larger than the length of the original list
  if N_nonzero > array_idx_satisfied.size:
      raise ValueError("N is larger than the number of elements in the list.")

  # Randomly choose N elements from the array without replacement
  chosen_idx = np.random.choice(array_idx_satisfied.shape[0], N_nonzero, replace=False) # replace=False: don't allow replacement
  array_chosen_idx_satisfied = array_idx_satisfied[chosen_idx]

  return array_chosen_idx_satisfied.tolist()


### make a matrix of a desired sparsity
def get_dense_mat(row_dim, col_dim, N_nonzero, rand_seed):
    np.random.seed(rand_seed)
    if N_nonzero > row_dim * col_dim:
        raise ValueError("N_nonzero cannot be greater than the total number of elements in the matrix")

    output_mat = np.zeros((row_dim, col_dim), dtype=int)
    flat_matrix = output_mat.flatten()
    indices = np.random.choice(flat_matrix.size, N_nonzero, replace=False)
    flat_matrix[indices] = 1
    output_mat = flat_matrix.reshape((row_dim, col_dim))

    return output_mat


####### make row sparse matrices #######
def get_row_sparse_mat(row_dim, col_dim, N_activated_outputs, N_nonzero, rand_seed):
    np.random.seed(rand_seed)
    
    if N_nonzero != 'all' and N_nonzero > N_activated_outputs * col_dim:
        raise ValueError("N_nonzero cannot be greater than the total number of possible entries in the activated rows")

    # Select the rows to activate
    activated_row_indices = np.random.choice(np.arange(row_dim), N_activated_outputs, replace=False)

    output_mat = np.zeros((row_dim, col_dim), dtype=int)

    if N_nonzero == 'all':
        output_mat[activated_row_indices, :] = 1 # set all entries in activated rows to 1
    else:
        # Flatten the activated rows
        flat_activated_rows = output_mat[activated_row_indices, :].flatten()

        # Choose positions to set to 1 within the activated rows
        indices = np.random.choice(flat_activated_rows.size, N_nonzero, replace=False)
        flat_activated_rows[indices] = 1

        # Reshape and place the activated rows back into the matrix
        output_mat[activated_row_indices, :] = flat_activated_rows.reshape((N_activated_outputs, col_dim))

    N_satisfy_threshold = N_activated_outputs * col_dim
    return output_mat, N_satisfy_threshold



####### functions for generating cone-LSmats #######
def get_list_coupling_idx_satisfy_threshold(input_mat, nx_in,ny_in, nx_out,ny_out, separation_threshold, geq_or_leq, N_activated_outputs, rand_seed):
  """
  only outputs in array_activated_outputs are activated
  """
  _,list_xcor_in,list_ycor_in,_, _,list_xcor_out,list_ycor_out,_ = get_ports_pos(input_mat, nx_in,ny_in,nx_out,ny_out)

  # randomly activate N_activated_outputs outputs in the index array
  np.random.seed(rand_seed) ### fix random seed
  array_activated_outputs = np.zeros(nx_out*ny_out, dtype=int)
  array_activated_outputs[np.random.choice(np.arange(nx_out*ny_out), size=N_activated_outputs, replace=False)] = 1

  N_satisfy_threshold = 0
  list_idx_satisfied = []
  for ii in range(nx_out*ny_out):
    if array_activated_outputs[ii] == 1: # if output[ii] is activated
      for jj in range(nx_in*ny_in):
        inplane_separation = np.sqrt( (list_xcor_in[jj]-list_xcor_out[ii])**2 + (list_ycor_in[jj]-list_ycor_out[ii])**2 )
        if geq_or_leq == 'geq':
          if inplane_separation >= separation_threshold:
            N_satisfy_threshold += 1
            list_idx_satisfied.append([ii,jj])
        elif geq_or_leq == 'leq':
          if inplane_separation <= separation_threshold:
            N_satisfy_threshold += 1
            list_idx_satisfied.append([ii,jj])

  return list_idx_satisfied, N_satisfy_threshold


def get_LSmat(input_mat, nx_in,ny_in, nx_out,ny_out, separation_threshold, geq_or_leq, N_activated_outputs, N_activated_couplings, rand_seed):
  """
  relation between N_activated and the density rho:
  N_activated_outputs = np.ceil(rho_activated_outputs * nx_out*ny_out)
  N_activated_couplings = np.ceil(rho_activated_couplings * nx_in*ny_in * nx_out*ny_out)
  """

  list_idx_satisfied, N_satisfy_threshold = get_list_coupling_idx_satisfy_threshold(input_mat, nx_in,ny_in, nx_out,ny_out, separation_threshold, geq_or_leq, N_activated_outputs, rand_seed)

  if N_activated_couplings == 'all':
    N_activated_couplings = N_satisfy_threshold
  elif isinstance(N_activated_couplings, int):
    if N_activated_couplings > N_satisfy_threshold:
      raise ValueError(f"N_activated_couplings={N_activated_couplings} exceeds N_satisfy_threshold={N_satisfy_threshold}. Not enough couplings to be activated.")
  else:
    raise ValueError("N_activated_couplings must be 'all' or an integer")

  list_chosen_idx_satisfied = random_sample_list_idx_satisfied(list_idx_satisfied, N_activated_couplings, rand_seed)


  mask_mat = np.zeros_like(input_mat)
  for idx in list_chosen_idx_satisfied:
    mask_mat[idx[0],idx[1]] = 1

  return mask_mat, N_satisfy_threshold
