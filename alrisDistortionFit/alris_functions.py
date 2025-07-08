import numpy as np
import tensorflow as tf


def transform_list_hkl_p63_p65(hkl_list):
    """
    Function to transform a list of hkl vectors from the old cell to the new cell, in the P63 structure as well as
    in the p65 structure.
    :param hkl_list: List of hkl vectors
    :return: List of hkl vectors in the new cell
    -9a,c,9b
    """
    # Convert hkl_list to a TensorFlow tensor
    hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    # Apply the transformation using TensorFlow operations
    h_new = -9 * hkl_list[:, 0]
    k_new = hkl_list[:, 2]
    l_new = 9 * hkl_list[:, 1]

    # Stack the new h, k, l components into a single tensor
    result = tf.stack([h_new, k_new, l_new], axis=1)

    return result


def get_atomic_form_factor(qnorm, atom):
    """
    Function to calculate the atomic form factor for a specific atom. Values for the Gaussian's are from
    International Tables for Crystallography, Vol. C, 2006.

    https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php#:~:text=Each%20diffraction%20peak%20corresponds%20to,intensity%20of%20the%20diffraction%20peak.
    
    :param qnorm: Norm of the hkl vector |Q|
    :param atom: Type of atom (atm only Pr, Ni or O is possible)
    :return: The atomic form factor

    Oxidation states:
    Pr: 4+
    Ba: 2+
    Cu: 2+
    O: 2+

    value of O is used becuase there is no O2-

    or

    Pr: 1+
    Ba: 2+
    Cu: 3+
    O: 2-


    """
    # Define values for Pr, Ni, O atoms as TensorFlow constants
    Pr_vals = {
        'a': tf.constant([20.9413, 20.0539, 12.4668, 0.296689], dtype=tf.float32),
        'b': tf.constant([2.54467, 0.202481, 14.8137, 45.4643], dtype=tf.float32),
        'c': tf.constant(1.24285, dtype=tf.float32),
    }
    Ba_vals = {
        'a': tf.constant([20.1807, 19.1136, 10.9054, 0.77634], dtype=tf.float32),
        'b': tf.constant([3.21367, 0.28331, 20.0558, 51.746], dtype=tf.float32),
        'c': tf.constant(3.02902, dtype=tf.float32),
    }
    Cu_vals = {
        'a': tf.constant([11.8168, 7.11181, 5.78135, 1.14523], dtype=tf.float32),
        'b': tf.constant([3.37484, 0.244078, 7.9876, 19.897], dtype=tf.float32),
        'c': tf.constant(1.14431, dtype=tf.float32),
    }
    O_vals = {
        'a': tf.constant([3.7504, 2.84294, 1.54298, 1.652091], dtype=tf.float32),
        'b': tf.constant([16.5151, 6.59203, 0.319201, 42.3486], dtype=tf.float32),
        'c': tf.constant(0.24206, dtype=tf.float32),
    }

    # Choose atom values based on the input atom
    if atom == "Pr":
        vals_dict = Pr_vals
    elif atom == "Ba":
        vals_dict = Ba_vals
    elif atom == "Cu":
        vals_dict = Cu_vals
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


def get_structure_factors(hkl_batch, structure):
    """
    Vectorized structure factor calculation.

    Parameters
    ----------
    hkl_batch : Tensor [N, 3]
        List of N hkl vectors
    structure : List of (atom, occupancy, position)
        Atomic basis of the crystal

    Returns
    -------
    Tensor [N] (complex64)
        Structure factors for each hkl
    """
    # Get atomic types and positions
    atoms = [a for a, _, _ in structure]
    positions = tf.stack([tf.convert_to_tensor(p, dtype=tf.float32) for _, _, p in structure])  # [A, 3]

    # Compute qnorms for each hkl vector (shape [N])
    qnorms = tf.norm(tf.cast(hkl_batch, tf.float32), axis=1)  # [N]
    # w = tf.constant(0.01, dtype=tf.float32)  # Debye-Waller factor old is 0.00159

    # Get per-atom form factors per hkl
    fq_table = {
        "Pr": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Pr"), tf.complex64), qnorms),
        "Ba": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Ba"), tf.complex64), qnorms),
        "Cu": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Cu"), tf.complex64), qnorms),
        "O": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "O"), tf.complex64), qnorms)
    }  # Each: [N]

    # Build full form factor matrix [N, A]
    fq_matrix = tf.stack([fq_table[atom] for atom in atoms], axis=1)  # shape [N, A]

    # Compute phase terms: [N, A]
    phase_arg = tf.tensordot(tf.cast(hkl_batch, tf.float32), tf.transpose(positions), axes=1)  # [N, A]
    phase = tf.exp(tf.complex(0.0, -2.0 * np.pi) * tf.cast(phase_arg, tf.complex64))  # [N, A]

    # Element-wise multiply and sum over atoms
    F_hkl = tf.reduce_sum(fq_matrix * phase, axis=1)  # [N]
    # Apply Debye-Waller factor
    # F_hkl = tf.cast(F_hkl, tf.complex64) * tf.cast(tf.exp(-w * qnorms ** 2), tf.complex64)  # [N]
    return F_hkl


def shift_atoms_p63(Pr1_1_dx, Pr1_1_dy, O1_1_dy, O1_1_dz, Ni1_1_dy, Ni1_2_dy):
    """
    Function to shift atoms in the P63 structure.
    :param Pr1_1_dx: Shift in the x direction for Pr atoms
    :param Pr1_1_dy: Shift in the y direction for Pr atoms
    :param O1_1_dy: Shift in the y direction for O1_1 atoms
    :param O1_1_dz: Shift in the z direction for O1_1 atoms
    :param Ni1_1_dy: Shift in the y direction for Ni1_1 atoms
    :param Ni1_2_dy: Shift in the y direction for Ni1_2 atoms
    :return:
    """
    res = [
        ['Pr', 59, [0.25 + Pr1_1_dx, 0.375 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.75 + Pr1_1_dx, 0.875 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.75 - Pr1_1_dx, 0.625 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.25 - Pr1_1_dx, 0.125 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.75 - Pr1_1_dx, 0.375 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.25 - Pr1_1_dx, 0.875 + Pr1_1_dy, 0.25]],
        ['Pr', 59, [0.25 + Pr1_1_dx, 0.625 - Pr1_1_dy, 0.75]],
        ['Pr', 59, [0.75 + Pr1_1_dx, 0.125 - Pr1_1_dy, 0.75]],
        ['O', 8, [0., 0.75 + O1_1_dy, 0.5 + O1_1_dz]],
        ['O', 8, [0.5, 0.25 + O1_1_dy, 0.5 + O1_1_dz]],
        ['O', 8, [0., 0.25 - O1_1_dy, 0. + O1_1_dz]],
        ['O', 8, [0.5, 0.75 - O1_1_dy, 0. + O1_1_dz]],
        ['O', 8, [0., 0.75 + O1_1_dy, 0. - O1_1_dz]],
        ['O', 8, [0.5, 0.25 + O1_1_dy, 0. - O1_1_dz]],
        ['O', 8, [0., 0.25 - O1_1_dy, 0.5 - O1_1_dz]],
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




def get_mode_amplitudes_p63(Pr1_1_dx, Pr1_1_dy, O1_1_dy, O1_1_dz, Ni1_1_dy, Ni1_2_dy):
    """
    Function to get the mode amplitudes for the given parameters in the P63 structure.
    :param Pr1_1_dx: Amplitude of the Pr atom in the x direction
    :param Pr1_1_dy: Amplitude of the Pr atom in the y direction
    :param O1_1_dy: Amplitude of the O atom in the y direction
    :param O1_1_dz: Amplitude of the O atom in the z direction
    :param Ni1_1_dy: Amplitude of the Ni atom in the y direction
    :param Ni1_2_dy: Amplitude of the Ni atom in the y direction
    :return: List of mode amplitudes
    """
    res = {
        "S1(a,a;0,0)[Pr1:d:dsp]A2u(a)": Pr1_1_dx,
        "M5-(0,a)[Pr1:d:dsp]Eu(a)": Pr1_1_dy,
        "S1(a,a;0,0)[O1:f:dsp]B3u(a)": -0.5 * O1_1_dy - 0.5 * O1_1_dz,
        "S1(a,a;0,0)[O1:f:dsp]B2u(a)": O1_1_dy - O1_1_dz,
        "S1(a,a;0,0)[Ni1:a:dsp]Eu(a)": Ni1_1_dy - Ni1_2_dy,
        "M5-(0,a)[Ni1:a:dsp]Eu(a)": -Ni1_1_dy - Ni1_2_dy,
    }
    return res



