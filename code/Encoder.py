
import tensorflow as tf

def encoder(X_list, batch_size, z_dim, s_dim, tau):

    samples = dict.fromkeys(['s', 'z', 'y', 'x'], [])
    q_params = dict()
    X = tf.concat(X_list, 1)

    # Create the proposal of q(s|x^o): categorical(x^~)
    samples['s'], q_params['s'] = s_proposal_multinomial(X, batch_size, s_dim, tau, reuse=None)

    # Create the proposal of q(z|s,x^o): N(mu(x^~,s), SIGMA(x^~,s))???
    samples['z'], q_params['z'] = z_proposal_GMM_factorized(X_list, samples['s'], batch_size, z_dim, reuse=None)

    return samples, q_params


def encoder_c(X_list, X_list_c, batch_size, z_dim, s_dim, tau):

    samples = dict.fromkeys(['s', 'z', 'y', 'x'], [])
    q_params = dict()
    X = tf.concat(X_list, 1)
    X_c = tf.concat(X_list_c, 1)

    # Create the proposal of q(s|x^o): categorical(x^~)
    samples['s'], q_params['s'] = s_proposal_multinomial_c(X, X_c, batch_size, s_dim, tau, reuse=None)

    # Create the proposal of q(z|s,x^o): N(mu(x^~,s), SIGMA(x^~,s))???
    samples['z'], q_params['z'] = z_proposal_GMM_factorized_c(X_list, X_c, samples['s'], batch_size, z_dim, reuse=None)

    return samples, q_params


def encoder_vae(X_list, X_list_c, batch_size, z_dim, s_dim, tau):

    samples = dict.fromkeys(['s', 'z', 'y', 'x'], [])
    q_params = dict()
    X = tf.concat(X_list, 1)
    X_c = tf.concat(X_list_c, 1)

    # Create the proposal of q(s|x^o): categorical(x^~)
    samples['s'], q_params['s'] = s_proposal_multinomial_c(X, X_c, batch_size, s_dim, tau, reuse=None)

    # Create the proposal of q(z|s,x^o): N(mu(x^~,s), SIGMA(x^~,s))???
    samples['z'], q_params['z'] = z_proposal_GMM_factorized_c(X_list, X_c, samples['s'], batch_size, z_dim, reuse=None)

    return samples, q_params



def z_proposal_GMM_factorized(X, samples_s, batch_size, z_dim, reuse):
    mean_qz = []
    log_var_qz = []

    for i, d in enumerate(X):
        observed_data = d
        observed_s = samples_s

        # Mean layer
        aux_m_qz = tf.layers.dense(inputs=tf.concat([observed_data, observed_s], 1), units=z_dim, activation=None,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                   name='layer_1_' + 'mean_enc_z' + str(i), reuse=reuse)


        # Logvar layers
        aux_lv_qz = tf.layers.dense(inputs=tf.concat([observed_data, observed_s], 1), units=z_dim, activation=None,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                    name='layer_1_' + 'logvar_enc_z' + str(i), reuse=reuse)

        mean_qz.append(aux_m_qz)
        log_var_qz.append(aux_lv_qz)

        # Input prior
    log_var_qz.append(tf.zeros([batch_size, z_dim]))
    mean_qz.append(tf.zeros([batch_size, z_dim]))

    # Compute full parameters, as a product of Gaussians distribution
    log_var_qz_joint = -tf.reduce_logsumexp(tf.negative(log_var_qz), 0)
    mean_qz_joint = tf.multiply(tf.exp(log_var_qz_joint),
                                tf.reduce_sum(tf.multiply(mean_qz, tf.exp(tf.negative(log_var_qz))), 0))

    # Avoid numerical problems
    # log_var_qz = tf.clip_by_value(log_var_qz, -15.0, 15.0)
    # Rep-trick
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    samples_z = mean_qz_joint + tf.multiply(tf.exp(log_var_qz_joint / 2), eps)

    return samples_z, [mean_qz_joint, log_var_qz_joint]


def z_proposal_GMM_factorized_c(X, X_c, samples_s, batch_size, z_dim, reuse):
    mean_qz = []
    log_var_qz = []

    for i, d in enumerate(X):
        observed_data = d
        observed_s = samples_s

        # Mean layer
        aux_m_qz = tf.layers.dense(inputs=tf.concat([observed_data, observed_s, X_c], 1), units=z_dim, activation=None,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                   name='layer_1_' + 'mean_enc_z' + str(i), reuse=reuse)

        # Logvar layers
        aux_lv_qz = tf.layers.dense(inputs=tf.concat([observed_data, observed_s, X_c], 1), units=z_dim, activation=None,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                    name='layer_1_' + 'logvar_enc_z' + str(i), reuse=reuse)

        mean_qz.append(aux_m_qz)
        log_var_qz.append(aux_lv_qz)

    # Input prior
    log_var_qz.append(tf.zeros([batch_size, z_dim]))
    mean_qz.append(tf.zeros([batch_size, z_dim]))

    # Compute full parameters, as a product of Gaussians distribution
    log_var_qz_joint = -tf.reduce_logsumexp(tf.negative(log_var_qz), 0)
    mean_qz_joint = tf.multiply(tf.exp(log_var_qz_joint),
                                tf.reduce_sum(tf.multiply(mean_qz, tf.exp(tf.negative(log_var_qz))), 0))

    # Avoid numerical problems
    # log_var_qz = tf.clip_by_value(log_var_qz, -15.0, 15.0)
    # Rep-trick
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    samples_z = mean_qz_joint + tf.multiply(tf.exp(log_var_qz_joint / 2), eps)

    return samples_z, [mean_qz_joint, log_var_qz_joint]


def z_proposal_distribution_GMM(x_list, x_list_c, samples_s, z_dim, reuse):
    # We propose a GMM for z

    x = tf.concat(x_list, 1)
    x_c = tf.concat(x_list_c, 1)

    h1 = tf.layers.dense(inputs=tf.concat([x, samples_s, x_c], 1), units=z_dim, activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                              name='layer_1_enc', reuse=reuse)

    # Mean layer
    aux_m_qz = tf.layers.dense(inputs=h1, units=z_dim, activation=None,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                               name='layer_2_' + 'mean_enc_z', reuse=reuse)

    # Logvar layers
    aux_lv_qz = tf.layers.dense(inputs=h1, units=z_dim, activation=None,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                                name='layer_2_' + 'logvar_enc_z', reuse=reuse)

    # Input prior

    log_var_qz.append(tf.zeros([batch_size, z_dim]))
    mean_qz.append(tf.zeros([batch_size, z_dim]))

    # Compute full parameters, as a product of Gaussians distribution
    log_var_qz_joint = -tf.reduce_logsumexp(tf.negative(log_var_qz), 0)
    mean_qz_joint = tf.multiply(tf.exp(log_var_qz_joint),tf.reduce_sum(tf.multiply(mean_qz, tf.exp(tf.negative(log_var_qz))), 0))

    # Avoid numerical problems
    # log_var_qz = tf.clip_by_value(log_var_qz, -15.0, 15.0)
    # Rep-trick
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    samples_z = mean_qz_joint + tf.multiply(tf.exp(log_var_qz_joint / 2), eps)

    return mean_pz, log_var_pz




def s_proposal_multinomial(X, batch_size, s_dim, tau, reuse):
    # Categorical(\pi(x^~))
    # We propose a categorical distribution to create a GMM for the latent space z
    log_pi = tf.layers.dense(inputs=X, units=s_dim, activation=None,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'enc_s',
                             reuse=reuse)

    # Gumbel-softmax trick (tau is temperature parameter)
    U = -tf.log(-tf.log(tf.random_uniform([batch_size, s_dim])))
    samples_s = tf.nn.softmax((log_pi + U) / tau)

    return samples_s, log_pi


def s_proposal_multinomial_c(X, X_c, batch_size, s_dim, tau, reuse):
    # Categorical(\pi(x^~))
    # We propose a categorical distribution to create a GMM for the latent space z
    log_pi = tf.layers.dense(inputs=tf.concat([X, X_c], 1), units=s_dim, activation=None,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.05), name='layer_1_' + 'enc_s',
                             reuse=reuse)

    # Gumbel-softmax trick (tau is temperature parameter)
    U = -tf.log(-tf.log(tf.random_uniform([batch_size, s_dim])))
    samples_s = tf.nn.softmax((log_pi + U) / tau)

    return samples_s, log_pi



def z_distribution_GMM(samples_s, z_dim, reuse):
    # We propose a GMM for z
    mean_pz = tf.layers.dense(inputs=samples_s, units=z_dim, activation=None,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.05),
                              name='layer_1_' + 'mean_dec_z', reuse=reuse)

    log_var_pz = tf.zeros([tf.shape(samples_s)[0], z_dim])

    # Avoid numerical problems
    log_var_pz = tf.clip_by_value(log_var_pz, -15.0, 15.0)

    return mean_pz, log_var_pz

