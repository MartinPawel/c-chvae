
import tensorflow as tf
import Loglik

def loglik_evaluation(batch_data_list, types_list, theta, normalization_params, reuse):

    log_p_x = []
    samples_x = []
    params_x = []

    # Independet yd -> Compute log(p(xd|yd))
    # batch data list is a list of tensors with different dimensions depending on data type

    for i, d in enumerate(batch_data_list):

        # Select the likelihood for the types of variables
        # For that we need to import 'loglik_models_missing_normalize' as function
        loglik_function = getattr(Loglik, 'loglik_' + types_list[i]['type'])

        out = loglik_function(d, types_list[i], theta[i], normalization_params[i],
                              kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_mean_dec_x' + str(i), reuse=reuse)

        log_p_x.append(out['log_p_x'])
        samples_x.append(out['samples'])
        params_x.append(out['params'])

    return log_p_x, samples_x, params_x



def loglik_evaluation_test(batch_data_list, theta, normalization_params, list_type):

    samples_x_perturbed = []
    params_x_perturbed = []

    # batch data list is a list of tensors with different dimensions depending on data type
    # needed here for loop; nothing else!

    for i, d in enumerate(batch_data_list):

        # Select the likelihood for the types of variables
        # For that we need to import 'loglik_models_missing_normalize' as function
        loglik_function = getattr(Loglik, 'loglik_test_' + list_type[i]['type'])

        out = loglik_function(theta[i], normalization_params[i], list_type[i])

        samples_x_perturbed.append(out['samples'])
        params_x_perturbed.append(out['params'])

    return samples_x_perturbed, params_x_perturbed




def cost_function(log_p_x, p_params, q_params, types_list, z_dim, y_dim, s_dim):
    # KL(q(s|x)|p(s))
    log_pi = q_params['s']
    pi_param = tf.nn.softmax(log_pi)
    KL_s = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi, labels=pi_param) + tf.log(float(s_dim))

    # KL(q(z|s,x)|p(z|s))
    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']
    KL_z = -0.5 * z_dim + 0.5 * tf.reduce_sum(
        tf.exp(log_var_qz - log_var_pz) + tf.square(mean_pz - mean_qz) / tf.exp(log_var_pz) - log_var_qz + log_var_pz,
        1)

    # Eq[log_p(x|y)]
    loss_reconstruction = tf.reduce_sum(log_p_x, 0)

    # Complete ELBO
    #ELBO = tf.reduce_mean(loss_reconstruction - KL_z - KL_s, 0)
    ELBO = tf.reduce_mean(1.20*loss_reconstruction - (KL_z + KL_s), 0)

    return ELBO, loss_reconstruction, KL_z, KL_s


def kl_z_diff(p_params, q_params, degree_active, batch_size, z_dim):
    # method to check whether one is within the polarized regime

    # parameters
    mean_pz, log_var_pz = p_params['z']
    mean_qz, log_var_qz = q_params['z']

    ones = tf.ones([batch_size, z_dim])

    # index according to global importance
    index = tf.greater(degree_active*ones, tf.reduce_mean(tf.exp(log_var_qz), 0))

    mean_qz_approx = tf.reshape(tf.boolean_mask(mean_qz, index), [batch_size, -1])
    mean_pz_approx = tf.reshape(tf.boolean_mask(mean_pz, index), [batch_size, -1])
    log_var_qz_approx = tf.reshape(tf.boolean_mask(log_var_qz, index), [batch_size, -1])
    log_var_pz_approx = tf.reshape(tf.boolean_mask(log_var_pz, index), [batch_size, -1])

    kl_approx = tf.reduce_mean(tf.reduce_sum(tf.exp(log_var_qz_approx - log_var_pz_approx) + tf.square(mean_pz_approx - mean_qz_approx) / tf.exp(log_var_pz_approx) - log_var_qz_approx + log_var_pz_approx, 1), 0)
    kl = tf.reduce_mean(tf.reduce_sum(tf.exp(log_var_qz - log_var_pz) + tf.square(mean_pz - mean_qz) / tf.exp(log_var_pz) - log_var_qz + log_var_pz, 1), 0)

    delta_kl = tf.divide(tf.abs(tf.subtract(kl_approx, kl)), kl)

    return [delta_kl, kl_approx, kl, index]

