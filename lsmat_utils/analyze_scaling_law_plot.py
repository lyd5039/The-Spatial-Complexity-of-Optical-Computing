import numpy as np



def analyze_max_C_of_mat(input_df, nx_in_array, rho, max_d_inplane, mat_type):
  ### NOTICE: nx_in = sqrt(N_in)

  avg_max_C_array = np.empty(len(nx_in_array))
  std_max_C_array = np.empty(len(nx_in_array))
  theory_max_C_array = np.empty(len(nx_in_array))

  for i, nx_in in enumerate(nx_in_array):
    if mat_type == 'LSmat':
      sub_df = input_df[(input_df['nx_in'] == nx_in) & (input_df['mat_density'] == rho) & (input_df['max_d_inplane'] == max_d_inplane+0.05)]
      theory_max_C_array[i] = (8*np.sqrt(2) / (3*np.pi)) * rho * nx_in**3 * max_d_inplane - 1/2*rho * nx_in**2 * max_d_inplane**2
    elif mat_type == 'sparse mat' or mat_type == 'dense mat':
      sub_df = input_df[(input_df['nx_in'] == nx_in) & (input_df['mat_density'] == rho)]
      theory_max_C_array[i] = 1/2 * rho * nx_in**4

    avg_max_C_array[i] = np.mean(sub_df['max_C'])
    std_max_C_array[i] = np.std(sub_df['max_C'], ddof=0)

  return avg_max_C_array, std_max_C_array, theory_max_C_array