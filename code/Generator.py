
import tensorflow as tf
import Encoder
import Decoder
import Evaluation

def samples_generator(batch_data_list, X_list, types_list, batch_size, z_dim, y_dim, y_dim_partition, s_dim, tau, normalization_params):

    samples_test = dict.fromkeys(['s' ,'z' ,'y' ,'x'] ,[])
    test_params = dict()
    X = tf.concat(X_list ,1)

    # Create the proposal of q(s|x^o)
    _, params = Encoder.s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=True)
    samples_test['s'] = tf.one_hot(tf.argmax(params ,1) ,depth=s_dim)

    # Create the proposal of q(z|s,x^o)
    _, params = Encoder.z_proposal_GMM_factorized(X_list, samples_test['s'], batch_size, z_dim, reuse=True)
    samples_test['z'] = params[0]

    # Create deterministic layer y
    samples_test['y'] = tf.layers.dense(inputs=samples_test['z'],
                                        units=y_dim,
                                        activation=None,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                        trainable=True,
                                        name= 'layer_h1_', reuse=True)

    grouped_samples_y = Decoder.y_partition(samples_test['y'], types_list, y_dim_partition)

    # Compute the parameters h_y
    theta = Decoder.theta_estimation_from_y(grouped_samples_y, types_list, batch_size, reuse=True)

    # Compute loglik and output of the VAE
    log_p_x, samples_test['x'], test_params['x'] = Evaluation.loglik_evaluation(batch_data_list,
                                                                     types_list,
                                                                     theta,
                                                                     normalization_params,
                                                                     reuse=True)

    return samples_test, test_params, log_p_x, theta



def samples_generator_c(batch_data_list, X_list, X_list_c, types_list, batch_size, z_dim, y_dim, y_dim_partition, s_dim, tau, normalization_params):

    samples_test = dict.fromkeys(['s' ,'z' ,'y' ,'x'] ,[])
    test_params = dict()
    X = tf.concat(X_list ,1)
    X_c = tf.concat(X_list_c, 1)

    # Create the proposal of q(s|x^o)
    _, params = Encoder.s_proposal_multinomial_c(X, X_c, batch_size, s_dim, tau, reuse=True)
    samples_test['s'] = tf.one_hot(tf.argmax(params, 1), depth=s_dim)

    # Create the proposal of q(z|s,x^o)
    _, params = Encoder.z_proposal_GMM_factorized_c(X_list, X_c, samples_test['s'], batch_size, z_dim, reuse=True)
    samples_test['z'] = params[0]

    # Create deterministic layer y
    samples_test['y'] = tf.layers.dense(inputs=samples_test['z'],
                                        units=y_dim,
                                        activation=None,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                        trainable=True,
                                        name='layer_h1_', reuse=True)

    grouped_samples_y = Decoder.y_partition(samples_test['y'], types_list, y_dim_partition)

    # Compute the parameters h_y
    theta = Decoder.theta_estimation_from_y(grouped_samples_y, types_list, batch_size, reuse=True)

    # Compute loglik and output of the VAE
    log_p_x, samples_test['x'], test_params['x'] = Evaluation.loglik_evaluation(batch_data_list,
                                                                     types_list,
                                                                     theta,
                                                                     normalization_params,
                                                                     reuse=True)

    return samples_test, test_params, log_p_x, theta




def samples_perturbation_z(batch_data_list, X_list, types_list, z_dim, y_dim, y_dim_partition, s_dim, tau,
                           normalization_params, nsamples, batch_size, p, l, h):
    # I ended up not using this one
    # should be: batch_size size = nsamples

    samples_test = dict.fromkeys(['s', 'z', 'y_tilde', 'z_tilde', 'x_tilde'], [])
    test_params = dict()
    X = tf.concat(X_list, 1)

    # -----------------------------------------------------------------------------------#
    # Encoder: Test Time

    # Create the proposal of q(s|x^o)
    _, params = Encoder.s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=True)
    samples_test['s'] = tf.one_hot(tf.argmax(params, 1), depth=s_dim)

    # Create the proposal of q(z|s,x^o)
    _, params = Encoder.z_proposal_GMM_factorized(X_list, samples_test['s'], batch_size, z_dim, reuse=True)
    samples_test['z'] = params[0]

    # -----------------------------------------------------------------------------------#
    # counterfactual step

    # z = samples_test['z']
    delta_z = tf.random_normal((nsamples, z_dim), 0, 1,
                               dtype=tf.float32)  # http://mathworld.wolfram.com/HyperspherePointPicking.html
    d = tf.add(tf.multiply(tf.random_uniform((nsamples, 1), 0, 1, dtype=tf.float32), (h - l)), l)  # length range [l, h)
    norm_p = tf.norm(delta_z, ord=p, axis=1)
    norm_p = tf.reshape(norm_p, [-1, 1])  # right format
    d_norm = tf.div(d, norm_p)  # rescale/normalize factor
    delta_z = tf.multiply(delta_z, d_norm)  # shape: (nsamples x z_dim)

    # -----------------------------------------------------------------------------------#
    # Decoder: Test Time

    # during counterfactual search
    z_tilde = tf.add(samples_test['z'], delta_z)  # gives (nsamples x z_dim) vector
    samples_test['z_tilde'] = tf.reshape(z_tilde, [-1, z_dim])  # use reshape to avoid rank error

    # Create deterministic layer y
    samples_test['y_tilde'] = tf.layers.dense(inputs=samples_test['z_tilde'],
                                              units=y_dim,
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                              trainable=True,
                                              name='layer_h1_', reuse=True)

    grouped_samples_y = Decoder.y_partition(samples_test['y_tilde'], types_list, y_dim_partition)

    # Compute the parameters h_y
    theta = Decoder.theta_estimation_from_y(grouped_samples_y, types_list, batch_size, reuse=True)

    # Compute loglik and output of the VAE
    log_p_x, samples_test['x_tilde'], test_params['x'] = Evaluation.loglik_evaluation(batch_data_list,
                                                                           types_list,
                                                                           theta,
                                                                           normalization_params,
                                                                           reuse=True)

    return samples_test, delta_z, d, theta