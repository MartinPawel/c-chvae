
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian




def decoder(samples_z, z_dim, y_dim, y_dim_partition, batch_size, types_list):

    samples = dict.fromkeys(['s', 'z', 'y', 'x'], [])
    gradients = dict.fromkeys(['g1', 'g2', 'g3'], [])

    samples['z'] = samples_z

    with tf.GradientTape() as g_1:
        g_1.watch(samples_z)
        # Create deterministic layer y
        samples['y'] = tf.layers.dense(inputs=samples_z, units=y_dim, activation=None,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_h1_', reuse=None)

    gradients['g1'] = g_1.gradient(samples['y'], samples_z)

    with tf.GradientTape() as g_2:
        g_2.watch(samples['y'])
        grouped_samples_y = y_partition(samples['y'], types_list, y_dim_partition)

    gradients['g2'] = g_2.gradient(grouped_samples_y, samples['y'])

    with tf.GradientTape() as g_3:
        g_3.watch(grouped_samples_y)
        # Compute the parameters h_y
        theta = theta_estimation_from_y(grouped_samples_y, types_list, batch_size, reuse=None)

    gradients['g3'] = g_3.gradient(theta, grouped_samples_y)


    return theta, samples, gradients


def y_partition(samples_y, types_list, y_dim_partition):
    grouped_samples_y = []
    # First element must be 0 and the length of the partition vector must be len(types_list)+1
    if len(y_dim_partition) != len(types_list):
        raise Exception("The length of the partition vector must match the number of variables in the data + 1")

    # Insert a 0 at the beginning of the cumsum vector
    partition_vector_cumsum = np.insert(np.cumsum(y_dim_partition), 0, 0)
    for i in range(len(types_list)):
        grouped_samples_y.append(samples_y[:, partition_vector_cumsum[i]:partition_vector_cumsum[i + 1]])

    return grouped_samples_y


def observed_data_layer(observed_data, output_dim, name, reuse):
    # Train a layer with the observed data and reuse it for the missing data
    obs_output = tf.layers.dense(inputs=observed_data, units=output_dim, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.05), name=name, reuse=reuse,
                                 trainable=True)

    return obs_output


def theta_estimation_from_y(samples_y, types_list, batch_size, reuse):
    theta = []

    # Independet yd -> Compute p(xd|yd)
    for i, d in enumerate(samples_y):

        observed_y = samples_y[i]
        nObs = tf.shape(observed_y)[0]

        # Different layer models for each type of variable
        if types_list[i]['type'] == 'real':
            params = theta_real(observed_y, types_list, i, reuse)

        elif types_list[i]['type'] == 'pos':
            params = theta_pos(observed_y, types_list, i, reuse)

        elif types_list[i]['type'] == 'count':
            params = theta_count(observed_y, types_list, i, reuse)

        elif types_list[i]['type'] == 'cat':
            params = theta_cat(observed_y, types_list, batch_size, i, reuse)

        elif types_list[i]['type'] == 'ordinal':
            params = theta_ordinal(observed_y, types_list, i, reuse)

        theta.append(params)

    return theta


def theta_real(observed_y, types_list, i, reuse):
    # Mean layer (To DO)
    h2_mean = observed_data_layer(observed_y, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
    # Sigma Layer (To DO)
    h2_sigma = observed_data_layer(observed_y, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i),
                                   reuse=reuse)

    return [h2_mean, h2_sigma]


def theta_pos(observed_y, types_list, i, reuse):
    # Mean layer
    h2_mean = observed_data_layer(observed_y, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)
    # Sigma Layer
    h2_sigma = observed_data_layer(observed_y, output_dim=types_list[i]['dim'], name='layer_h2_sigma' + str(i),
                                   reuse=reuse)

    return [h2_mean, h2_sigma]


def theta_count(observed_y, types_list, i, reuse):
    # Lambda Layer
    h2_lambda = observed_data_layer(observed_y, output_dim=types_list[i]['dim'], name='layer_h2' + str(i), reuse=reuse)

    return h2_lambda


def theta_cat(observed_y, types_list, batch_size, i, reuse):
    # Log pi layer, with zeros in the first value to avoid the identificability problem
    h2_log_pi_partial = observed_data_layer(observed_y, output_dim=int(types_list[i]['dim']) - 1,
                                            name='layer_h2' + str(i), reuse=reuse)
    h2_log_pi = tf.concat([tf.zeros([batch_size, 1]), h2_log_pi_partial], 1)

    return h2_log_pi


def theta_ordinal(observed_y, types_list, i, reuse):
    # Theta layer, Dimension of ordinal - 1
    h2_theta = observed_data_layer(observed_y, output_dim=int(types_list[i]['dim']) - 1, name='layer_h2' + str(i),
                                   reuse=reuse)
    # Mean layer, a single value
    h2_mean = observed_data_layer(observed_y, output_dim=1, name='layer_h2_sigma' + str(i), reuse=reuse)

    return [h2_theta, h2_mean]


def decoder_test_time(samples_z, z_dim, y_dim, y_dim_partition, batch_size, types_list):
    samples = dict.fromkeys(['s', 'z', 'y', 'x'], [])

    samples['z'] = samples_z

    # Create deterministic layer y
    samples['y'] = tf.layers.dense(inputs=samples_z, units=y_dim, activation=None,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_h1_',
                                   reuse=True)

    grouped_samples_y = y_partition(samples['y'], types_list, y_dim_partition)

    # Compute the parameters h_y
    theta = theta_estimation_from_y(grouped_samples_y, types_list, batch_size, reuse=True)

    return theta, samples