import numpy as np
import matplotlib.colors as mcolors



def lighten_color(color, amount=0.5):
    white = np.array([1, 1, 1])
    color = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex((1 - amount) * color + amount * white)


def analyze_max_C_of_mat(input_df, nx_in_array, rho, max_d_inplane, mat_type):
    ### NOTICE: nx_in = sqrt(N_in)

    avg_max_C_array = np.empty(len(nx_in_array))
    std_max_C_array = np.empty(len(nx_in_array))
    theory_max_C_array = np.empty(len(nx_in_array))

    for i, nx_in in enumerate(nx_in_array):
        if mat_type == 'LSmat':
            sub_df = input_df[(input_df['nx_in'] == nx_in) &
                              (input_df['percentage_activated_outputs'] == rho) &
                              (input_df['max(d_inplane)'] == max_d_inplane + 0.05)]
            ####### when max_d_inplane = 6, there are other simulation results with rho=low_density
            ####### need to be filtered out
            if max_d_inplane == 6:
                sub_df = sub_df[sub_df['mat_density'] > 0.02]
            theory_max_C_array[i] = 2 * max_d_inplane * (np.sqrt(2) * nx_in - max_d_inplane)

        elif mat_type == 'row sparse mat':
            sub_df = input_df[(input_df['nx_in'] == nx_in) & (input_df['percentage_activated_outputs'] == rho)]
            theory_max_C_array[i] = rho * nx_in**2

        elif mat_type == 'trivial sparse mat':
            sub_df = input_df[(input_df['nx_in'] == nx_in) & (input_df['mat_density'] == rho)]
            theory_max_C_array[i] = 1 * nx_in**2

        avg_max_C_array[i] = np.mean(sub_df['max(C)'])
        std_max_C_array[i] = np.std(sub_df['max(C)'], ddof=0)

    return avg_max_C_array, std_max_C_array, theory_max_C_array


####### methods for analyzing LSmats #######
def analyze_max_C_of_LSmat(input_df, nx_in_array, rho_row, mat_density, max_d_inplane, mat_type):
    ### NOTICE: nx_in = sqrt(N_in)

    if mat_type != 'LSmat':
        raise ValueError(f"Invalid mat_type '{mat_type}'. Only 'LSmat' is accepted for this function.")

    avg_max_C_array = np.empty(len(nx_in_array))
    std_max_C_array = np.empty(len(nx_in_array))
    theory_max_C_array = np.empty(len(nx_in_array))

    for i, nx_in in enumerate(nx_in_array):
        sub_df = input_df[(input_df['nx_in'] == nx_in) &
                          (input_df['percentage_activated_outputs'] == rho_row) &
                          (input_df['mat_density'] == mat_density) &
                          (input_df['max(d_inplane)'] == max_d_inplane + 0.05)]
        theory_max_C_array[i] = 2 * np.sqrt(2) * max_d_inplane * (nx_in - max_d_inplane)

        avg_max_C_array[i] = np.mean(sub_df['max(C)'])
        std_max_C_array[i] = np.std(sub_df['max(C)'], ddof=0)

    return avg_max_C_array, std_max_C_array, theory_max_C_array


def max_rho_of_LSmat(input_df, nx_in_array, rho, max_d_inplane, mat_type):
    avg_max_rho_array = np.empty(len(nx_in_array))
    theory_max_rho_array = np.empty(len(nx_in_array))

    if mat_type != 'LSmat':
        raise ValueError(f"Invalid mat_type '{mat_type}'. Only 'LSmat' is accepted for this function.")
    
    for i, nx_in in enumerate(nx_in_array):
        sub_df = input_df[
            (input_df['nx_in'] == nx_in) &
            (input_df['percentage_activated_outputs'] == rho) &
            (input_df['max(d_inplane)'] == max_d_inplane + 0.05)
        ]
        
        if max_d_inplane == 6:
            sub_df = sub_df[sub_df['mat_density'] > 0.02]
        
        theory_max_rho_array[i] = (
            (nx_in**2 - max_d_inplane * nx_in + max_d_inplane**2) *
            np.pi * max_d_inplane**2 / nx_in**4
        )
        
        avg_max_rho_array[i] = np.mean(sub_df['mat_density'])

    return avg_max_rho_array, theory_max_rho_array
