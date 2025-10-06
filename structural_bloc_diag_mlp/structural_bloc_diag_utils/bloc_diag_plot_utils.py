import os
import pandas as pd
import re



def bloc_diag_str_to_latex(input_str):
    # Split the input into left and right parts at the '|'
    left_part, right_part = input_str.split('|')

    # Regex to find all patterns like '[num]*count'
    pattern = re.compile(r'\[(\d+)\]\*(\d+)')

    # Find all matches for both parts
    left_matches = pattern.findall(left_part.strip())
    right_matches = pattern.findall(right_part.strip())

    if not left_matches or not right_matches:
        return "Invalid input format"


    left_dims = [int(x[0]) for x in left_matches]
    right_dims = [int(x[0]) for x in right_matches]
    left_n_mats = [int(x[1]) for x in left_matches]
    right_n_mats = [int(x[1]) for x in right_matches]
    if sum(left_n_mats) != sum(right_n_mats):
        raise ValueError("columns and rows of block matrices do not match.")


    # Generate the LaTeX part
    parts = []
    if len(right_n_mats) > len(left_n_mats) and len(left_n_mats) == 1:
        for right_n_mat, right_dim in zip(right_n_mats, right_dims):
            parts.append(f"{right_n_mat} \\times \\rm \\mathbb{{R}}^{{{left_dims[0]} \\times {right_dim}}}")
    elif len(left_n_mats) > len(right_n_mats) and len(right_n_mats) == 1:
        for left_n_mat, left_dim in zip(left_n_mats, left_dims):
            parts.append(f"{left_n_mat} \\times \\rm \\mathbb{{R}}^{{{left_dim} \\times {right_dims[0]}}}")
    elif len(right_n_mats) == 1 and len(left_n_mats) == 1:
        if sum(left_n_mats) == 1:
            parts.append(f"\\rm \\mathbb{{R}}^{{{left_dims[0]} \\times {right_dims[0]}}}")
        elif sum(left_n_mats) > 1:
            parts.append(f"{left_n_mats[0]} \\times \\rm \\mathbb{{R}}^{{{left_dims[0]} \\times {right_dims[0]}}}")

    # Join all parts with ' and ' or just return a single part
    if len(parts) > 1:
        result = '\\ and\\ '.join(parts)
    else:
        result = parts[0]

    return result


def load_df(df_path):
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        raise FileNotFoundError(f"No such file: '{df_path}'")
    
    return df


def calculate_accu_to_plot(df, config):
    subdf_config = df[df['config_name'] == config]
    if config == 'config0':
        avg_accu = subdf_config['test_accuracy_before_dropping'].mean()
        std_accu = subdf_config['test_accuracy_before_dropping'].std(ddof=0)
    else:
        avg_accu = subdf_config['test_accuracy_off_diag_dropped'].mean()
        std_accu = subdf_config['test_accuracy_off_diag_dropped'].std(ddof=0)
    
    return avg_accu, std_accu


def nonzero_blocks_to_axis_label(nonzero_blocks_str):
    """
    nonzero_blocks_str: unseparated string describing the bloc_diag model
    e.g.,
    [1]*100 | [7]*16+[8]*84, [1]*100 | [1]*100, [1]*10 | [10]*10, [10]*1 | [10]*1
    """
    nonzero_blocks_list_str = nonzero_blocks_str.split(", ")
    output_str = []
    for block_str in nonzero_blocks_list_str:
        try:
            # Replace ' and ' with ', '
            block_str = bloc_diag_str_to_latex(block_str).replace('\\ and\\ ', ', ')
            output_str.append(f"${block_str}$")
        except ValueError as e:
            output_str.append(f"${str(e)}$")

    # Create the final LaTeX formatted string with line breaks
    axis_label = '\n'.join(output_str)

    return axis_label


def calculate_optical_complexity(nonzero_blocks_str):
    """
    nonzero_blocks_str: unseparated string describing the bloc_diag model
    e.g.,
    [1]*100 | [7]*16+[8]*84, [1]*100 | [1]*100, [1]*10 | [10]*10, [10]*1 | [10]*1
    """
    ### convert to axis label format
    ### without line swapping \n
    nonzero_blocks_list_str = nonzero_blocks_str.split(", ")
    list_latex_str = []
    for block_str in nonzero_blocks_list_str:
        try:
            # Replace ' and ' with ', '
            block_str = bloc_diag_str_to_latex(block_str).replace('\\ and\\ ', ', ')
            list_latex_str.append(f"${block_str}$")
        except ValueError as e:
            list_latex_str.append(f"${str(e)}$")


    processed_str_list = []

    # Iterate over each element in the original list
    for element in list_latex_str:
        # Check if element contains a comma
        if ',' in element:
            # Strip the initial and final '$', split the element at the comma, and wrap each part back with '$'
            parts = element.strip('$').split(', ')
            split_elements = ['$' + part + '$' for part in parts]
            # Extend the processed list with the split elements
            processed_str_list.extend(split_elements)
        else:
            # If no comma, add the element as is to the processed list
            processed_str_list.append(element)
    print(processed_str_list)


    total_elements = 0
    total_alpha_u = 0
    unitary_matrices_info = []


    for processed_str in processed_str_list:
        # Regular expression to extract numbers from the LaTeX string
        numbers = list(map(int, re.findall(r'\d+', processed_str)))
        if len(numbers) == 2:  # Only dimensions provided, default to one matrix
            num_matrices = 1
            m, n = numbers
        elif len(numbers) == 3:
            num_matrices, m, n = numbers
        else:
            raise ValueError("The input string does not does not describe the number and dim of matrices.")

        unitary_mats_U = f"{num_matrices} x U({m})"
        unitary_mats_VT = f"{num_matrices} x U({n})"


        total_elements += num_matrices * m * n
        total_alpha_u += num_matrices * ((m-1) * m / 2 + (n-1) * n / 2)
        unitary_matrices_info.append(unitary_mats_U)
        unitary_matrices_info.append(unitary_mats_VT)
    
    return total_elements, unitary_matrices_info, total_alpha_u