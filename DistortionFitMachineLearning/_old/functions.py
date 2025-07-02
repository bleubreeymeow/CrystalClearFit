from pprint import pprint

import numpy as np
import tensorflow as tf

def shift_atoms_p63(cif_file, output_file="cif/temp_cif.cif", Pr1_1_dx=0, Pr1_1_dy=0, O1_1_dy=0, O1_1_dz=0, Ni1_1_dy=0,
                           Ni1_2_dy=0):
    """
    Returns the adjusted CIF file with the new atomic positions
    :param cif_file: CIF file to be adjusted
    :param output_file: Name of the new CIF file
    :param Pr1_1_dx: Pr Atom shift x
    :param Pr1_1_dy: Pr Atom shift y
    :param O1_1_dy: O Atom shift y
    :param O1_1_dz: O Atom shift z
    :param Ni1_1_dy: Ni Atom shift y
    :param Ni1_2_dy: Ni Atom shift y
    """
    modifications = {
        'Pr1_1': {'x': Pr1_1_dx, 'y': Pr1_1_dy},
        'O1_1': {'y': O1_1_dy, 'z': O1_1_dz},
        'Ni1_1': {'y': Ni1_1_dy},
        'Ni1_2': {'y': Ni1_2_dy},
    }

    with open(cif_file, 'r') as file:
        lines = file.readlines()

    atom_section = False
    modified_lines = []

    for line in lines:
        if line.startswith("_atom_site_fract_symmform"):
            atom_section = True
            modified_lines.append(line)
            continue

        if line.startswith("_iso_displacivemode_number"):
            atom_section = False
            modified_lines.append(line)
            continue

        if atom_section:
            parts = line.split()
            if len(parts) < 8:
                modified_lines.append(line)
                continue

            atom_label = parts[0]
            if atom_label in modifications:
                for i, key in enumerate(['x', 'y', 'z']):
                    if key in modifications[atom_label]:
                        parts[4 + i] = str(float(parts[4 + i]) + modifications[atom_label][key])
            modified_lines.append(" ".join(parts) + "\n")
        else:
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)
    return output_file

def transform_list_hkl_p63(hkl_list):
    """
    Function to transform a list of hkl vectors from the old cell to the new cell
    :param hkl_list: List of hkl vectors
    :return: List of hkl vectors in the new cell
    2c,-2a-2b,a-b
    """
    # Convert hkl_list to a TensorFlow tensor
    hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    # Apply the transformation using TensorFlow operations
    h_new = 2 * hkl_list[:, 2]
    k_new = -2 * (hkl_list[:, 0] + hkl_list[:, 1])
    l_new = hkl_list[:, 0] - hkl_list[:, 1]

    # Stack the new h, k, l components into a single tensor
    result = tf.stack([h_new, k_new, l_new], axis=1)

    return result


def get_atomic_form_factor(qnorm, atom):

    """
    Function to calculate the atomic form factor for a specific atom. Values for the Gaussian's are from
    International Tables for Crystallography, Vol. C, 2006.
    :param qnorm: Norm of the hkl vector |Q|
    :param atom: Type of atom (atm only Pr, Ni or O is possible)
    :return: The atomic form factor
    """
    # Define values for Pr, Ni, O atoms as TensorFlow constants
    Pr_vals = {
        'a': tf.constant([21.3727, 19.7491, 12.1329, 0.97578], dtype=tf.float32),
        'b': tf.constant([2.64520, 0.214299, 15.323, 36.4065], dtype=tf.float32),
        'c': tf.constant(1.77132, dtype=tf.float32)
    }
    Ni_vals = {
        'a': tf.constant([12.1271, 7.34625, 4.8940, 1.67865], dtype=tf.float32),
        'b': tf.constant([3.77755, 0.25070000000000003, 10.52465, 44.25235], dtype=tf.float32),
        'c': tf.constant(0.94775, dtype=tf.float32)
    }
    O_vals = {
        'a': tf.constant([3.7504, 2.84294, 1.54298, 1.652091], dtype=tf.float32),
        'b': tf.constant([16.5151, 6.59203, 0.319201, 42.3486], dtype=tf.float32),
        'c': tf.constant(0.24206, dtype=tf.float32)
    }

    # Choose atom values based on the input atom
    if atom == "Pr":
        vals_dict = Pr_vals
    elif atom == "Ni":
        vals_dict = Ni_vals
    else:
        vals_dict = O_vals

    # Start with the constant "c" term
    fq = vals_dict["c"]

    # Use element-wise operations instead of a loop
    a_vals = vals_dict["a"]
    b_vals = vals_dict["b"]

    # Compute the exponential terms
    exponential_terms = tf.exp(-b_vals * (qnorm / (4 * tf.constant(np.pi))) ** 2)

    # Multiply the "a" values with the corresponding exponential terms and sum them
    fq += tf.reduce_sum(a_vals * exponential_terms)

    return fq


def get_structure_factor(hkl, structure):
    """
    Function to calculate the structure factor for a certain Q vector hkl.
    :param hkl: hkl vector in style [h, k, l]
    :param structure: Structure of the crystal
    :return: Structure factor
    """
    qnorm = tf.linalg.norm(hkl)  # Calculate the magnitude of the Q vector
    F_hkl = tf.complex(0.0, 0.0)  # Initialize the structure factor as a complex number
    fq_ni = get_atomic_form_factor(qnorm, "Ni")
    fq_pr = get_atomic_form_factor(qnorm, "Pr")
    fq_o = get_atomic_form_factor(qnorm, "O")
    for atom, occ, pos in structure:
        # Get the atomic form factor for the current atom
        if atom == "Ni":
            fq = fq_ni
        elif atom == "Pr":
            fq = fq_pr
        else:
            fq = fq_o
        pos = tf.convert_to_tensor(pos, dtype=tf.float32)
        phase = tf.exp(tf.complex(0.0, -2 * np.pi) * tf.complex(tf.reduce_sum(hkl * pos), 0.0))
        fq = tf.cast(fq, tf.complex64)
        phase = tf.cast(phase, tf.complex64)
        # Add the contribution of this atom to the structure factor
        F_hkl += fq * phase

    return F_hkl

def shift_atoms(Pr1_1_dx, Pr1_1_dy, O1_1_dy, O1_1_dz, Ni1_1_dy, Ni1_2_dy):
    res = [
        ['Pr', 59, [0.25 + Pr1_1_dx,   0.375 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.75 + Pr1_1_dx,   0.875 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.75 - Pr1_1_dx,   0.625 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.25 - Pr1_1_dx,   0.125 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.75 - Pr1_1_dx,   0.375 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.25 - Pr1_1_dx,   0.875 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.25 + Pr1_1_dx,   0.625 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.75 + Pr1_1_dx,   0.125 - Pr1_1_dy, 0.75]],
        ['O', 8, [0., 0.75 + O1_1_dy, 0.5 + O1_1_dz]],
        ['O', 8, [0.5, 0.25 + O1_1_dy, 0.5 + O1_1_dz]],
        ['O', 8, [0., 0.25 - O1_1_dy, 0. + O1_1_dz]],
        ['O', 8, [0.5, 0.75 - O1_1_dy, 0. + O1_1_dz]],
        ['O', 8, [0.,  0.75 + O1_1_dy, 0. - O1_1_dz]],
        ['O', 8, [0.5, 0.25 + O1_1_dy, 0. - O1_1_dz]],
        ['O', 8, [0.,  0.25 - O1_1_dy, 0.5 - O1_1_dz]],
        ['O', 8, [0.5, 0.75 - O1_1_dy, 0.5 - O1_1_dz]],
        ['O', 8, [0., 0., 0]],
        ['O', 8, [0.5, 0.5, 0.]],
        ['O', 8, [0., 0., 0.5]],
        ['O', 8, [0.5, 0.5, 0.5]],
        ['O', 8, [0., 0.5, 0.]],
        ['O', 8, [0.5, 0., 0.]],
        ['O', 8, [0., 0.5, 0.5]],
        ['O', 8, [0.5, 0., 0.5]],
        ['Ni', 28, [0., 0.125 + Ni1_1_dy, 0.25]],
        ['Ni', 28, [0.5, 0.625 + Ni1_1_dy, 0.25]],
        ['Ni', 28, [0., 0.875 - Ni1_1_dy, 0.75]],
        ['Ni', 28, [0.5, 0.375 - Ni1_1_dy, 0.75]],
        ['Ni', 28, [0., 0.625 + Ni1_2_dy, 0.25]],
        ['Ni', 28, [0.5, 0.125 + Ni1_2_dy, 0.25]],
        ['Ni', 28, [0., 0.375 - Ni1_2_dy, 0.75]],
        ['Ni', 28, [0.5, 0.875 - Ni1_2_dy, 0.75]]
        ]
    return res
