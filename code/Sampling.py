import numpy as np
import tensorflow as tf
import time
#import csv
#import pandas as pd
#import matplotlib
#import random
#from matplotlib import pyplot as plt
#import seaborn as sns
#from numpy import linalg as LA
#from scipy.spatial.distance import cdist
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.cluster import DBSCAN
#from sklearn import preprocessing
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm, tqdm_notebook

from recourse.builder import RecourseBuilder
from recourse.builder import ActionSet

# import functions
import Helpers
import Evaluation
import Graph

#import Encoder
#import Decoder
#import Generator
#import Loglik
#import LaugelEtAl
np.random.seed(619)

def print_loss(epoch, start_time, avg_loss, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, train_loglik: %.8f, KL_z: %.8f, KL_s: %.8f, ELBO: %.8f"
          % (epoch, time.time() - start_time, avg_loss, avg_KL_z, avg_KL_s, avg_loss - avg_KL_z - avg_KL_s))

# -----------------------------------------------------------------------------------#
############################# Running the C-CHVAE search   ##########################
# -----------------------------------------------------------------------------------#

def sampling(settings, types_dict, types_dict_c, out, ncounterfactuals, clf, n_batches_train, n_samples_train, k, n_input, degree_active):

    argvals = settings.split()
    args = Helpers.getArgs(argvals)

# Creating graph
    sess_HVAE = tf.Graph()

    with sess_HVAE.as_default():
        # args.model_name: excluded
        tf_nodes = Graph.C_CHVAE_graph(args.types_file, args.types_file_c,
                                   learning_rate=1e-3, z_dim=args.dim_latent_z,
                                   y_dim=args.dim_latent_y, s_dim=args.dim_latent_s,
                                   y_dim_partition=args.dim_latent_y_partition, nsamples=1000, p=2)

    # start session
    with tf.Session(graph=sess_HVAE) as session:
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        print('Initizalizing Variables ...')
        tf.global_variables_initializer().run()

        # -----------------------------------------------------------------------------------#
        # Apply on training data

        print('Training the CHVAE ...')
        if (args.train == 1):

            start_time = time.time()
            # Training cycle

            loglik_epoch = []
            KL_s_epoch = []
            KL_z_epoch = []
            for epoch in tqdm(range(args.epochs)):
                avg_loss = 0.
                avg_KL_s = 0.
                avg_KL_z = 0.
                samples_list = []
                p_params_list = []
                q_params_list = []
                log_p_x_total = []

                # Annealing of Gumbel-Softmax parameter
                tau = np.max([1.0 - 0.001 * epoch, 1e-3])

                # Randomize the data in the mini-batches
                train_data = out['training'][1]
                train_data_c = out['training'][2]
                random_perm = np.random.permutation(range(np.shape(train_data)[0]))
                train_data_aux = train_data[random_perm, :]
                train_data_aux_c = train_data_c[random_perm, :]

                for i in range(n_batches_train):
                    # Create inputs for the feed_dict
                    data_list = Helpers.next_batch(train_data_aux, types_dict, args.batch_size, index_batch=i)  # DONE
                    data_list_c = Helpers.next_batch(train_data_aux_c, types_dict_c, args.batch_size, index_batch=i)  # DONE

                    # Create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_c'], data_list_c)})
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['batch_size']] = args.batch_size

                    # Running VAE
                    _, X_list, loss, KL_z, KL_s, samples, log_p_x, p_params, q_params = session.run(
                        [tf_nodes['optim'],
                        tf_nodes['X'],
                        tf_nodes['loss_re'],
                        tf_nodes['KL_z'],
                        tf_nodes['KL_s'],
                        tf_nodes['samples'],
                        tf_nodes['log_p_x'],
                        tf_nodes['p_params'],
                        tf_nodes['q_params']],
                        feed_dict=feedDict)

                    # Collect all samples, distirbution parameters and logliks in lists
                    if i == 0:
                        samples_list = [samples]
                        p_params_list = [p_params]
                        q_params_list = [q_params]
                        log_p_x_total = [log_p_x]
                    else:
                        samples_list.append(samples)
                        p_params_list.append(p_params)
                        q_params_list.append(q_params)
                        log_p_x_total.append(log_p_x)

                    # Compute average loss
                    avg_loss += np.mean(loss)
                    avg_KL_s += np.mean(KL_s)
                    avg_KL_z += np.mean(KL_z)

                # Concatenate samples in arrays
                s_total, z_total, y_total, est_data = Helpers.samples_concatenation(samples_list)

                # Transform discrete variables back to the original values
                train_data_transformed = Helpers.discrete_variables_transformation(
                    train_data_aux[:n_batches_train * args.batch_size, :], types_dict)
                est_data_transformed = Helpers.discrete_variables_transformation(est_data, types_dict)

                # Create global dictionary of the distribution parameters
                p_params_complete = Helpers.p_distribution_params_concatenation(p_params_list,  # DONE
                                                                            types_dict,
                                                                            args.dim_latent_z,
                                                                            args.dim_latent_s)

                q_params_complete = Helpers.q_distribution_params_concatenation(q_params_list,  # DONE
                                                                            args.dim_latent_z,
                                                                            args.dim_latent_s)

                # Compute mean and mode of our loglik models: these correspond to the estimated values
                loglik_mean, loglik_mode = Helpers.statistics(p_params_complete['x'], types_dict)  # DONE

                # Try this for the errors
                error_train_mean = Helpers.error_computation(train_data_transformed, loglik_mean, types_dict)
                error_train_mode = Helpers.error_computation(train_data_transformed, loglik_mode, types_dict)
                error_train_samples = Helpers.error_computation(train_data_transformed, est_data_transformed, types_dict)

                # Display logs per epoch step
                if epoch % args.display == 0:
                    print_loss(epoch, start_time, avg_loss / n_batches_train, avg_KL_s / n_batches_train,
                           avg_KL_z / n_batches_train)
                    print("")

            # Plot evolution of test loglik
                loglik_per_variable = np.sum(np.concatenate(log_p_x_total, 1), 1) / n_samples_train

                loglik_epoch.append(loglik_per_variable)

            # -----------------------------------------------------------------------------------#
            # Apply on test data

            for i in range(1):
                samples_test_list = []
                test_params_list = []
                log_p_x_test_list = []
                data_c_list = []

                test_data_counter = out['test_counter'][1]
                test_data_c_counter = out['test_counter'][2]
                y_test_counter = out['test_counter'][3]
                n_samples_test = test_data_counter.shape[0]

                # Create test minibatch
                data_list = Helpers.next_batch(test_data_counter, types_dict, n_samples_test, index_batch=i)
                data_list_c = Helpers.next_batch(test_data_c_counter, types_dict_c, n_samples_test, index_batch=i)  # DONE

            # Constant Gumbel-Softmax parameter (where we have finished the annealing
                tau = 1e-3

            # Create feed dictionary
                feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], data_list)}
                feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_c'], data_list_c)})
                feedDict[tf_nodes['tau_GS']] = tau
                feedDict[tf_nodes['batch_size']] = ncounterfactuals  # n_samples_test

            # Get samples from the generator function (computing the mode of all distributions)
                samples_test, log_p_x_test, test_params, theta_test, normalization_params_test, X, delta_kl = session.run(
                    [tf_nodes['samples_test'],
                    tf_nodes['log_p_x_test'],
                    tf_nodes['test_params'],
                    tf_nodes['theta_test'],
                    tf_nodes['normalization_params'],
                    tf_nodes['X'],
                    tf_nodes['delta_kl']],
                    feed_dict=feedDict)

                samples_test_list.append(samples_test)
                test_params_list.append(test_params)
                log_p_x_test_list.append(log_p_x_test)
                data_c_list.append(data_list_c)

            # Concatenate samples in arrays
            s_total_test, z_total_test, y_total_test, samples_total_test = Helpers.samples_concatenation(samples_test_list)

            # Transform discrete variables back to the original values
            est_samples_transformed = Helpers.discrete_variables_transformation(samples_total_test, types_dict)

            # -----------------------------------------------------------------------------------#
            # Find k Attainable Counterfactuals
            print('[*] Find Attainable Counterfactuals...')

            counter_batch_size = 1  # counterfactual batch size (i.e. look for counterfactuals one by one)
            data_concat = []
            data_concat_c = []
            counterfactuals = []
            latent_tilde = []
            latent = []

            search_samples = args.search_samples
            p = args.norm_latent_space

            for i in tqdm(range(ncounterfactuals)):

                s = (k, n_input)  # preallocate k spots; # inputs
                sz = (k, args.dim_latent_z)
                s = np.zeros(s)
                sz = np.zeros(sz)
                ik = 0  # counter

                l = 0
                step = args.step_size

                x_adv, y_adv, z_adv, d_adv = None, None, None, None


                #scale test observations
                scaled_test, scaler_test = Helpers.standardize(test_data_counter)

                # get one test observation
                data_list = Helpers.next_batch(test_data_counter, types_dict, counter_batch_size, index_batch=i)
                data_list_c = Helpers.next_batch(test_data_c_counter, types_dict_c, counter_batch_size, index_batch=i)
                hat_y_test = np.repeat(y_test_counter[i] * 1, search_samples, axis=0)
                test_data_c_replicated = np.repeat(test_data_c_counter[i, :].reshape(1, -1), search_samples, axis=0)
                replicated_scaled_test = np.repeat(scaled_test[i, :].reshape(1, -1), search_samples, axis=0)


                # get replicated observations (observation replicated nsamples times)
                #replicated_scaled_test = Helpers.replicate_data_list(data_list_scaled, search_samples)
                replicated_data_list = Helpers.replicate_data_list(data_list, search_samples)
                replicated_data_list_c = Helpers.replicate_data_list(data_list_c, search_samples)
                replicated_z = np.repeat(z_total_test[i].reshape(-1, args.dim_latent_z), search_samples, axis=0)

                h = l + step
                # counter to stop
                count = 0
                counter_step = 1
                max_step = 500

                while True:

                    count = count + counter_step

                    if  (count > max_step) == True:
                        sz = None
                        s = None
                        z = z_total_test[i].reshape(-1, args.dim_latent_z)
                        break

                    if degree_active == 1: #choose all latent features for search

                        delta_z = np.random.randn(search_samples, replicated_z.shape[1])  # http://mathworld.wolfram.com/HyperspherePointPicking.html
                        d = np.random.rand(search_samples) * (h - l) + l  # length range [l, h)
                        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
                        d_norm = np.divide(d, norm_p).reshape(-1, 1)  # rescale/normalize factor
                        delta_z = np.multiply(delta_z, d_norm)
                        z_tilde = replicated_z + delta_z  # z tilde

                    else:

                        delta_z = np.random.randn(search_samples, replicated_z.shape[1])  # http://mathworld.wolfram.com/HyperspherePointPicking.html
                        d = np.random.rand(search_samples) * (h - l) + l  # length range [l, h)
                        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
                        d_norm = np.divide(d, norm_p).reshape(-1, 1)  # rescale/normalize factor
                        delta_z = np.multiply(delta_z, d_norm)

                        mask = np.tile(delta_kl[3][0, :] * 1,
                                       (search_samples, 1))  # only alter most important latent features
                        delta_z = np.multiply(delta_z, mask)

                        z_tilde = replicated_z + delta_z


                    # create feed dictionary
                    feedDict = {i: d for i, d in zip(tf_nodes['ground_batch'], replicated_data_list)}
                    feedDict.update({i: d for i, d in zip(tf_nodes['ground_batch_c'], replicated_data_list_c)})
                    feedDict[tf_nodes['samples_z']] = z_tilde
                    feedDict[tf_nodes['tau_GS']] = tau
                    feedDict[tf_nodes['batch_size']] = search_samples

                    theta_perturbed, samples_perturbed = session.run([tf_nodes['theta_perturbed'],
                                                                  tf_nodes['samples_perturbed']], feed_dict=feedDict)

                    x_tilde, params_x_perturbed = Evaluation.loglik_evaluation_test(X_list,
                                                                                theta_perturbed,
                                                                                normalization_params_test,
                                                                                types_dict)
                    x_tilde = np.concatenate(x_tilde, axis=1)
                    scaled_tilde = scaler_test.transform(x_tilde)
                    d_scale = np.sum(np.abs(scaled_tilde - replicated_scaled_test), axis=1)

                    x_tilde = np.c_[test_data_c_replicated, x_tilde]
                    y_tilde = clf.predict(x_tilde)

                    indices_adv = np.where(y_tilde == 0)[0]

                    if len(indices_adv) == 0:  # no candidate generated
                        l = h
                        h = l + step
                    elif all(s[k - 1, :] == 0):  # not k candidates generated

                        indx = indices_adv[np.argmin(d_scale[indices_adv])]
                        assert (y_tilde[indx] != 1)

                        s[ik, :] = x_tilde[indx, :]
                        sz[ik, :] = z_tilde[indx, :]
                        z = z_total_test[i].reshape(-1, args.dim_latent_z)

                        ik = ik + 1  # up the count
                        l = h
                        h = l + step
                    else:  # k candidates genereated
                        break

                data_concat.append(np.concatenate(data_list, axis=1))
                data_concat_c.append(np.concatenate(data_list_c, axis=1))
                counterfactuals.append(s)
                latent_tilde.append(sz)
                latent.append(z)

    cchvae_counterfactuals = np.array(counterfactuals)
    return cchvae_counterfactuals

