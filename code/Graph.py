
import tensorflow as tf
import numpy as np
import Helpers
import Encoder
import Decoder
import Evaluation
import Generator

# MASTER of Disaster

def C_HVAE_graph(types_file, learning_rate=1e-4, z_dim=1, y_dim=1, s_dim=1, y_dim_partition=[], nsamples=1000, p=2):

    # -----------------------------------------------------------------------------------#
    # Preliminaries

    # Load remaining placeholders
    print('[*] Defining placeholders')


    # Placeholder for batch_size (required for counterfactual search loop)
    batch_size = tf.placeholder(dtype=tf.int32)
    # Placeholder for Gumbel-softmax parameter
    tau = tf.placeholder(tf.float32, shape=())
    batch_data_list, types_list = Helpers.place_holder_types(types_file, batch_size)

    # Batch normalization of the data
    X_list, normalization_params, X_list_noisy = Helpers.batch_normalization(batch_data_list, types_list, batch_size)


    # Set dimensionality of Y
    if y_dim_partition:
        y_dim_output = np.sum(y_dim_partition)
    else:
        y_dim_partition = y_dim * np.ones(len(types_list), dtype=int)
        y_dim_output = np.sum(y_dim_partition)

    # -----------------------------------------------------------------------------------#
    # (HVAE) Encoder and Decoder for training time

    # Encoder
    print('[*] Defining Encoder...')
    samples, q_params = Encoder.encoder(X_list_noisy, batch_size, z_dim, s_dim, tau)

    samples_s = samples['s']
    samples_z = samples['z']
    p_params = dict()

    # Create the distribution of p(z|s)
    p_params['z'] = Encoder.z_distribution_GMM(samples['s'], z_dim, reuse=None)

    # Decoder
    print('[*] Defining Decoder...')
    theta, samples, gradient_decoder = Decoder.decoder(samples_z, z_dim, y_dim_output, y_dim_partition, batch_size, types_list)

    samples['s'] = samples_s
    # Compute loglik and output of the VAE
    log_p_x, samples['x'], p_params['x'] = Evaluation.loglik_evaluation(batch_data_list,
                                                             types_list,
                                                             theta,
                                                             normalization_params,
                                                             reuse=None)

    # Evaluate active vs passive variables
    degree_active = 0.95# must be less than 1 (not used in paper)
    delta_kl = Evaluation.kl_z_diff(p_params, q_params, degree_active, batch_size, z_dim)


    # -----------------------------------------------------------------------------------#
    # optimize ELBO

    print('[*] Defining Cost function...')
    ELBO, loss_reconstruction, KL_z, KL_s = Evaluation.cost_function(log_p_x,
                                                          p_params,
                                                          q_params,
                                                          types_list,
                                                          z_dim,
                                                          y_dim_output,
                                                          s_dim)

    optim = tf.train.AdamOptimizer(learning_rate).minimize(-ELBO)

    # -----------------------------------------------------------------------------------#
    # Generator function for test time sample generation
    samples_test, test_params, log_p_x_test, theta_test = Generator.samples_generator(batch_data_list,
                                                                            X_list,
                                                                            types_list,
                                                                            batch_size,
                                                                            z_dim,
                                                                            y_dim_output,
                                                                            y_dim_partition,
                                                                            s_dim,
                                                                            tau,
                                                                            normalization_params)

    # -----------------------------------------------------------------------------------#
    #  Decoder for test time counterfactuals
    #  'samples_perturbed': does not contain 'x' samples

    print('[*] Defining Test Time Decoder...')
    theta_perturbed, samples_perturbed = Decoder.decoder_test_time(samples_z,
                                                            z_dim,
                                                            y_dim_output,
                                                            y_dim_partition,
                                                            batch_size,
                                                            types_list)

    # Evaluation Function not necessary here
    '''log_p_x, samples_perturbed['x'], p_params_x_perturbed = Evaluation.loglik_evaluation(batch_data_list,
                                                                              types_list,
                                                                              theta_perturbed,
                                                                              normalization_params,
                                                                              reuse=True)'''

    # -----------------------------------------------------------------------------------#
    # Packing results

    tf_nodes = {'batch_size': batch_size,#feed
                'ground_batch': batch_data_list,#feed
                'tau_GS': tau,#feed,
                #'predict_proba': predict_proba,#feed
                'samples_z': samples_z,#feed
                'samples': samples,
                'log_p_x': log_p_x,
                'loss_re': loss_reconstruction,
                'loss': -ELBO,
                'optim': optim,
                'KL_s': KL_s,
                'KL_z': KL_z,
                'X': X_list,
                'p_params': p_params,
                'q_params': q_params,
                'samples_test': samples_test,
                'test_params': test_params,
                'log_p_x_test': log_p_x_test,
                'samples_perturbed': samples_perturbed,
                'theta_test': theta_test,
                'theta_perturbed': theta_perturbed,
                'normalization_params': normalization_params,
                'gradient_decoder': gradient_decoder,
                'delta_kl': delta_kl}

    return tf_nodes


# MASTER of Disaster for conditional density approximations

def C_CHVAE_graph(types_file, types_file_c, learning_rate=1e-3, z_dim=1, y_dim=1, s_dim=1, y_dim_partition=[], nsamples=1000, p=2, degree_active=0.95):

    # -----------------------------------------------------------------------------------#
    # Preliminaries

    # Load placeholders
    print('[*] Defining placeholders')

    # c: short for 'conditional'
    # Placeholder for batch_size (required for counterfactual search loop)
    batch_size = tf.placeholder(dtype=tf.int32)
    # Placeholder for Gumbel-softmax parameter
    tau = tf.placeholder(tf.float32, shape=())
    batch_data_list, types_list = Helpers.place_holder_types(types_file, batch_size)
    batch_data_list_c, types_list_c = Helpers.place_holder_types(types_file_c, batch_size)


    # Batch normalization of the data
    X_list, normalization_params, X_list_noisy = Helpers.batch_normalization(batch_data_list, types_list, batch_size)
    # Batch normalization of the data
    X_list_c, _, X_list_noisy_c = Helpers.batch_normalization(batch_data_list_c, types_list, batch_size)


    # Set dimensionality of Y
    if y_dim_partition:
        y_dim_output = np.sum(y_dim_partition)
    else:
        y_dim_partition = y_dim * np.ones(len(types_list), dtype=int)
        y_dim_output = np.sum(y_dim_partition)

    # -----------------------------------------------------------------------------------#
    # (HVAE) Encoder and Decoder for training time

    # Encoder
    print('[*] Defining Encoder...')
    samples, q_params = Encoder.encoder_c(X_list, X_list_c, batch_size, z_dim, s_dim, tau)

    samples_s = samples['s']
    samples_z = samples['z']
    p_params = dict()

    # Create the distribution of p(z|s)
    p_params['z'] = Encoder.z_distribution_GMM(samples['s'], z_dim, reuse=None)

    # Decoder
    print('[*] Defining Decoder...')
    theta, samples, gradient_decoder = Decoder.decoder(samples_z, z_dim, y_dim_output, y_dim_partition, batch_size, types_list)

    samples['s'] = samples_s
    # Compute loglik and output of the VAE
    log_p_x, samples['x'], p_params['x'] = Evaluation.loglik_evaluation(batch_data_list,
                                                             types_list,
                                                             theta,
                                                             normalization_params,
                                                             reuse=None)

    # -----------------------------------------------------------------------------------#
    # optimize ELBO

    print('[*] Defining Cost function...')
    ELBO, loss_reconstruction, KL_z, KL_s = Evaluation.cost_function(log_p_x,
                                                          p_params,
                                                          q_params,
                                                          types_list,
                                                          z_dim,
                                                          y_dim_output,
                                                          s_dim)

    optim = tf.train.AdamOptimizer(learning_rate).minimize(-ELBO)

    # -----------------------------------------------------------------------------------#
    # Generator function for test time sample generation
    samples_test, test_params, log_p_x_test, theta_test = Generator.samples_generator_c(batch_data_list,
                                                                            X_list, X_list_c,
                                                                            types_list,
                                                                            batch_size,
                                                                            z_dim,
                                                                            y_dim_output,
                                                                            y_dim_partition,
                                                                            s_dim,
                                                                            tau,
                                                                            normalization_params)

    # -----------------------------------------------------------------------------------#
    #  Decoder for test time counterfactuals
    #  'samples_perturbed': does not contain 'x' samples

    print('[*] Defining Test Time Decoder...')
    theta_perturbed, samples_perturbed = Decoder.decoder_test_time(samples_z,
                                                            z_dim,
                                                            y_dim_output,
                                                            y_dim_partition,
                                                            batch_size,
                                                            types_list)

    # Evaluation Function not necessary here
    degree_active = degree_active# must be less than 1
    delta_kl = Evaluation.kl_z_diff(p_params, q_params, degree_active, batch_size, z_dim)

    # -----------------------------------------------------------------------------------#
    # Packing results

    tf_nodes = {'batch_size': batch_size,  #feed
                'ground_batch': batch_data_list,  #feed
                'ground_batch_c': batch_data_list_c,  #feed
                'tau_GS': tau,  #feed,
                'samples_z': samples_z,  #feed
                'samples': samples,
                'log_p_x': log_p_x,
                'loss_re': loss_reconstruction,
                'loss': -ELBO,
                'optim': optim,
                'KL_s': KL_s,
                'KL_z': KL_z,
                'X': X_list,
                'p_params': p_params,
                'q_params': q_params,
                'samples_test': samples_test,
                'test_params': test_params,
                'log_p_x_test': log_p_x_test,
                'samples_perturbed': samples_perturbed,
                'theta_test': theta_test,
                'theta_perturbed': theta_perturbed,
                'normalization_params': normalization_params,
                'gradient_decoder': gradient_decoder,
                'delta_kl': delta_kl}

    return tf_nodes