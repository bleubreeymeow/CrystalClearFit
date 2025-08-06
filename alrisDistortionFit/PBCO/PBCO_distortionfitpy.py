import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from importlib import reload
import alris_functions2
reload(alris_functions2)
from alris_functions2 import shift_atoms, transform_list_hkl_p63_p65, get_structure_factors , atom_position_list
from itertools import chain
from matplotlib.markers import MarkerStyle
import multiprocessing as mp

global features ,labels , labels_err, matrix , max_mode_amps

def init_worker():

    seed = 1



    # Create separate seed
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    worker_seed = seed + worker_id
    tf.keras.utils.set_random_seed(seed=worker_seed)


def ran_iteration(iteration):
    global features, labels, labels_err, matrix, max_mode_amps
    n_dim = 3
    lr = 5e-1
    max_mode_amps = tf.constant([5.6,6.36,5.6,11.19,11.19,12.73,11.19,12.73,11.19,11.19,12.73,11.19,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,9.0,7.91,9.0,9.0,9.0,7.91,9.0,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,6.36,6.36,5.6,6.36,6.36,5.6,6.36,6.36,6.36,6.36,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,8.29,7.91,8.04,8.42,8.04,8.42,0.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,15.59,18.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,11.73,11.19,11.37,11.91,11.73,11.19,11.37,11.91,11.02,12.73,14.7,12.73,11.73,11.19,11.37,11.91,11.73,11.19,11.37,11.91,11.73,11.19,11.37,11.91,11.02,12.73,14.7,12.73,11.73,11.19,11.37,11.91,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,12.73,11.02,11.19,12.73,11.02,12.73,12.73,11.02,11.19,12.73,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,7.79,9.0,9.14,7.91,9.14,7.91,8.29,7.91,9.14,9.58,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,9.14,7.91,9.14,7.91,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,9.14,9.58,8.04,8.42,7.79,9.0,9.14,7.91,9.14,7.91,8.29,7.91,8.04,8.42,9.14,9.58,8.29,7.91,8.04,8.42,9.14,9.58,7.79,9.0,10.39,9.0,10.39,9.0,8.29,7.91,8.04,8.42,9.14,9.58,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,9.14,7.91,9.14,7.91,8.29,7.91,9.14,9.58,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,5.6,6.36,5.6,11.19,11.19,12.73,11.19,12.73,11.19,11.19,12.73,11.19,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,9.0,7.91,9.0,9.0,9.0,7.91,9.0,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,6.36,6.36,5.6,6.36,6.36,5.6,6.36,6.36,6.36,6.36,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,7.79,9.0,9.14,7.91,9.14,7.91,0.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,15.59,18.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,12.73,11.02,11.19,12.73,11.02,12.73,12.73,11.02,11.19,12.73,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,10.39,9.0,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,5.6,6.36,5.6,11.19,11.19,12.73,11.19,12.73,11.19,11.19,12.73,11.19,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,9.0,7.91,9.0,9.0,9.0,7.91,9.0,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,9.0,7.91,9.0,7.91,9.0,9.0,9.0,7.91,7.91,7.91,7.91,7.91,7.91,9.0,7.91,7.91,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,6.36,6.36,5.6,6.36,6.36,5.6,6.36,6.36,6.36,6.36,6.36,5.6,5.6,6.36,5.6,5.6,6.36,5.6,5.6,6.36,6.36,5.6,6.36,5.6,8.29,7.91,8.04,8.42,8.04,8.42,0.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,15.59,18.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.73,11.19,11.37,11.91,11.73,11.19,11.37,11.91,11.02,12.73,14.7,12.73,11.73,11.19,11.37,11.91,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,12.73,11.02,11.19,12.73,11.02,12.73,12.73,11.02,11.19,12.73,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,9.14,9.58,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,9.14,9.58,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,10.39,9.0,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,9.14,9.58,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,0.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,15.59,18.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,11.73,11.19,11.37,11.91,11.73,11.19,11.37,11.91,11.02,12.73,14.7,12.73,11.73,11.19,11.37,11.91,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,12.73,11.02,11.19,12.73,11.02,12.73,12.73,11.02,11.19,12.73,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,10.39,9.0,9.14,7.91,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,10.39,9.0,9.14,7.91,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,9.14,9.58,8.29,7.91,8.04,8.42,9.14,9.58,7.79,9.0,10.39,9.0,10.39,9.0,8.29,7.91,8.04,8.42,9.14,9.58,8.29,7.91,8.04,8.42,8.04,8.42,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,10.39,9.0,9.14,7.91,8.29,7.91,8.04,8.42,8.04,8.42,7.79,9.0,9.14,7.91,9.14,7.91,0.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,15.59,18.0,15.59,15.83,15.59,15.83,15.59,18.0,15.59,15.83,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,12.92,11.19,11.02,12.73,14.7,12.73,11.02,12.73,12.92,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,11.02,11.19,12.73,11.02,11.19,12.73,11.02,12.73,12.73,11.02,11.19,12.73,11.02,11.19,11.19,11.02,11.19,11.19,11.02,12.73,11.19,11.02,11.19,11.19,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,10.39,9.0,10.39,9.0,7.79,9.0,9.14,7.91,10.39,9.0,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91,7.79,9.0,10.39,9.0,9.14,7.91,7.79,9.0,9.14,7.91,9.14,7.91])

    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    n_epochs = 100


 # Create the model
    inputs = tf.keras.Input(shape=(n_dim,))
    outputs = FunAsLayer(matrix , max_mode_amps)(inputs)
    model = tf.keras.Model(inputs, outputs)


    # Compile the model with the custom loss function and metric
    model.compile(
        optimizer=optim,
        loss= 'mse', # MSE_weighted() if using errors
        metrics=[r_factor_metric],
        run_eagerly=False,  # Set to True for debugging, False for performance
    )

    history = model.fit(
        x=features,
        y=labels,  # replace with combined_labels if using errors
        batch_size = features.shape[0], # Use a smaller batch size features.shape[0]
        epochs=n_epochs,
        verbose='auto',
        shuffle=True, # not sure whether this matters
        # callbacks=[cb]
        sample_weight=labels_err  # Use sample weights if you have errors
    )

    final_loss = history.history['loss'][-1]
    best_model_pars = max_mode_amps * tf.tanh(model.layers[-1].get_weights()[0])

    return best_model_pars, final_loss





def fun_tf(hkl_list, pars, matrix):
    """
    Fast computation of structure factors with parameter-dependent structure.
    """

    # Get modified structure
    pars_tensor = tf.stack(pars)  # shape (params,)

    atom_shift_list = shift_atoms(matrix , (pars_tensor))
    atom_shift_list = atom_shift_list[:,0]
    atom_shift_list = tf.unstack(atom_shift_list)

    modified_struct = atom_position_list(*atom_shift_list)

    hkl_list = transform_list_hkl_p63_p65(hkl_list)

    sf_hkl = get_structure_factors(hkl_list, modified_struct)
    intensity = (abs(sf_hkl)) ** 2
    w = tf.constant(0.00032001553565274784, dtype=tf.float32)  # Debye-Waller factor 
    qnorms = tf.norm(tf.cast(hkl_list, tf.float32), axis=1)
    intensity = intensity * tf.exp(- w* qnorms ** 2)  # Apply Debye-Waller factor

    intensity = intensity / tf.reduce_sum(intensity) * 60
    return intensity


class FunAsLayer(tf.keras.layers.Layer):
    def __init__(self, matrix , max_mode_amps,**kwargs):
        super().__init__(**kwargs)
        self.max_mode_amps = max_mode_amps
        self.matrix = matrix

    def build(self, input_shape):
        self.param = self.add_weight(name='param', shape=(1290,), initializer=tf.keras.initializers.GlorotNormal(seed=1), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # Apply tanh to ensure parameters stay within the [-1, 1] range then multiply by max_mode_amps so each parameter is scaled corresponding to the element in max_mode_amps
        pretransform = tf.tanh(self.param)
        transformed_params = pretransform * self.max_mode_amps  # Scale parameters

        output = fun_tf(inputs, (transformed_params) , self.matrix)
        return tf.reshape(output , [-1])  # Ensure output is 1D
    

"""
# R-Score based on intensity
class RFactorLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.reduce_sum(y_true)
"""
    
# mean squared error
class PerSampleMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        squared_error = tf.square(y_true - y_pred)
        per_sample_mse = tf.reduce_mean(squared_error, axis=-1)
        return per_sample_mse  # shape (batch_size,)
  
# Define the custom metric function
def r_factor_metric(y_true, y_pred):
    labels = y_true
    return tf.reduce_sum(tf.abs(labels - y_pred)) / tf.reduce_sum(labels)


def make_sample_weights(experimental_data):
    labels = experimental_data["intensity_exp"].tolist()
    labels = labels / np.sum(labels) * 60 #Normalize labels
    vol_err = experimental_data["intensity_exp_err"].tolist()

    labels_err = []

    for label, err in zip(labels, vol_err):
        if label == 0:
            labels_err.append(10)  # Assign a high error for zero labels
        else:
            labels_err.append(1/(label))  # Inverse error for each label

    labels_err = tf.convert_to_tensor(labels_err, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    labels = tf.expand_dims(labels, axis=-1)  # Ensure labels are 2D
    labels_err = tf.expand_dims(labels_err, axis=-1)  # Ensure labels_err are 2D

    return labels, labels_err

def fn_distortion_fitting(labels , labels_err , matrix , features , n_dim , iteration_num):
    #initialise the histogram matrix
    histogram_matrix = tf.zeros([iteration_num , iteration_num], dtype=tf.float32)
    
    lr = [5e-1]
    best_pars_overall = None
    best_rf_overall = np.inf
    best_loss_overall = []

    for learning_rate in lr:

        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        n_epochs = 100
        histories = []
        n_iter = iteration_num

        min_loss = np.inf
        best_pars = None

        # List to store the loss values for each epoch
        all_losses = []


        for i in range(n_iter):
            # Create the model
            inputs = tf.keras.Input(shape=(n_dim,))
            outputs = FunAsLayer(matrix , max_mode_amps)(inputs)
            model = tf.keras.Model(inputs, outputs)


            # Compile the model with the custom loss function and metric
            model.compile(
                optimizer=optim,
                loss= 'mse', # MSE_weighted() if using errors
                metrics=[r_factor_metric],
                run_eagerly=False,  # Set to True for debugging, False for performance

            )
            
            history = model.fit(
            x=features,
            y=labels,  # replace with combined_labels if using errors
            batch_size = features.shape[0], # Use a smaller batch size features.shape[0]
            epochs=n_epochs,
            verbose='auto',
            shuffle=True, # not sure whether this matters
            # callbacks=[cb]
            sample_weight=labels_err  # Use sample weights if you have errors
            )

            histories.append(history)
            all_losses.append(history.history['loss'])
            # Check final loss
            final_loss = history.history['loss'][-1]
            print(model.layers[-1].get_weights()[0].shape)
            curren_model_pars = max_mode_amps * tf.tanh(model.layers[-1].get_weights()[0])
            print(f"Final loss: {final_loss:.3e}")
            '''
            print(f"Best parameters for iteration {i+1}:")
            for j, par in enumerate(curren_model_pars):
                print(f"Parameter {j+1}: {par.numpy():.4f}")
            '''
            #open the file and write the parameters


            if final_loss < min_loss:
                # Update best model parameters
                best_model_pars = max_mode_amps * tf.tanh(model.layers[-1].get_weights()[0])
                min_loss = final_loss
                rf = r_factor_metric(labels, fun_tf(features, best_model_pars , matrix))
                print(f"Iteration {i+1} - New best loss: {min_loss:.3e} (R-factor: {rf:.3e})")

        if min_loss < best_rf_overall:
            best_rf_overall = min_loss
            best_pars_overall = best_model_pars
            best_loss_overall = all_losses

        # Plotting the loss values
        plt.figure(figsize=(10, 6))

        # Plot the loss values for each iteration
        for i, loss_values in enumerate(all_losses):
            plt.plot(loss_values, label=f'Iteration {i+1}')


        return histogram_matrix




if __name__ == "__main__":
    # Load experimental data
    experimental_data = pd.read_csv('C:/Users/User/Desktop/uzh_intern/CrystalClearFit/alrisDistortionFit/PBCO/raw_data/combined_peaks.csv')
    matrix = np.loadtxt('C:/Users/User/Desktop/uzh_intern/CrystalClearFit/alrisDistortionFit/PBCO/new_PBCO_fit/matrix.txt', dtype=np.float32)

    n_features = experimental_data.shape[0]
    n_dim = 3

    hkl_list = experimental_data[["h", "k", "l"]].values.tolist()
    hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)

    labels, labels_err = make_sample_weights(experimental_data)

    histogram_matrix = fn_distortion_fitting(labels, labels_err, matrix, hkl_list, n_dim, iteration_num=1000)





