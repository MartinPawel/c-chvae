#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  27 05 2019

@based on Nazabal et al 2018

List of loglikelihoods for the types of variables considered in this paper.
Basically, we create the different layers needed in the decoder and during the
generation of new samples

The variable reuse indicates the mode of this functions
- reuse = None -> Decoder implementation
- reuse = True -> Samples generator implementation

"""

import tensorflow as tf
import numpy as np
import Helpers
from scipy.special import softmax
from scipy.special import expit


def loglik_real(batch_data, list_type, theta, normalization_params, kernel_initializer, name, reuse):
    
    output=dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data = batch_data
    
    data_mean, data_var = normalization_params
    data_var = tf.clip_by_value(data_var, epsilon, np.inf)
    
    est_mean, est_var = theta
    est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0) #Must be positive
    
    # Affine transformation of the parameters
    est_mean = tf.sqrt(data_var)*est_mean + data_mean
    est_var = data_var*est_var
    
    #Compute loglik
    log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data, est_mean)/est_var, 1) - int(list_type['dim'])*0.5*tf.log(2* np.pi) - 0.5*tf.reduce_sum(tf.log(est_var),1)
    
    #Outputs
    output['log_p_x'] = log_p_x
    output['params'] = [est_mean, est_var]
    output['samples'] = tf.contrib.distributions.Normal(est_mean, tf.sqrt(est_var)).sample()
        
    return output

def loglik_pos(batch_data, list_type, theta, normalization_params, kernel_initializer, name, reuse):
    
    #Log-normal distribution
    output = dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data_mean_log, data_var_log = normalization_params
    data_var_log = tf.clip_by_value(data_var_log, epsilon, np.inf)
    
    data = batch_data
    data_log = tf.log(1.0 + data)
    
    est_mean, est_var = theta
    est_var = tf.clip_by_value(tf.nn.softplus(est_var), epsilon, 1.0)
    
    # Affine transformation of the parameters
    est_mean = tf.sqrt(data_var_log)*est_mean + data_mean_log
    est_var = data_var_log*est_var
    
    #Compute loglik
    log_p_x = -0.5 * tf.reduce_sum(tf.squared_difference(data_log,est_mean)/est_var,1) \
        - 0.5*tf.reduce_sum(tf.log(2*np.pi*est_var),1) - tf.reduce_sum(data_log,1)
    
    output['log_p_x'] = log_p_x
    output['params'] = [est_mean, est_var]
    output['samples'] = tf.exp(tf.contrib.distributions.Normal(est_mean,tf.sqrt(est_var)).sample()) - 1.0
        
    return output

def loglik_cat(batch_data, list_type, theta, normalization_params, kernel_initializer, name, reuse):
    
    output=dict()
    
    #Data outputs
    data = batch_data
    
    log_pi = theta
    
    #Compute loglik
    log_p_x = -tf.nn.softmax_cross_entropy_with_logits(logits=log_pi,labels=data)
    
    output['log_p_x'] = log_p_x
    output['params'] = log_pi
    output['samples'] = tf.one_hot(tf.contrib.distributions.Categorical(probs=tf.nn.softmax(log_pi)).sample(),depth=int(list_type['dim']))
    
    return output
    
def loglik_ordinal(batch_data, list_type, theta, normalization_params, kernel_initializer, name, reuse):
    
    output=dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data = batch_data
    batch_size = tf.shape(data)[0]
    
    # We need to force that the outputs of the network increase with the categories
    partition_param, mean_param = theta
    mean_value = tf.reshape(mean_param,[-1,1])
    theta_values = tf.cumsum(tf.clip_by_value(tf.nn.softplus(partition_param), epsilon, 1e20),1)
    sigmoid_est_mean = tf.nn.sigmoid(theta_values - mean_value)
    mean_probs = tf.concat([sigmoid_est_mean,tf.ones([batch_size,1],tf.float32)],1) - tf.concat([tf.zeros([batch_size,1],tf.float32),sigmoid_est_mean],1)
    
    #Code needed to compute samples from an ordinal distribution
    true_values = tf.one_hot(tf.reduce_sum(tf.cast(data,tf.int32),1)-1,int(list_type['dim']))
    
    #Compute loglik
    log_p_x = tf.log(tf.clip_by_value(tf.reduce_sum(mean_probs*true_values,1),epsilon,1e20))
    
    output['log_p_x'] = log_p_x
    output['params'] = mean_probs
    output['samples'] = tf.sequence_mask(1+tf.contrib.distributions.Categorical(logits=tf.log(tf.clip_by_value(mean_probs,epsilon,1e20))).sample(), int(list_type['dim']),dtype=tf.float32)
    
    return output

def loglik_count(batch_data, list_type, theta, normalization_params, kernel_initializer, name, reuse):
    
    output=dict()
    epsilon = tf.constant(1e-6, dtype=tf.float32)
    
    #Data outputs
    data = batch_data
    
    est_lambda = theta
    est_lambda = tf.clip_by_value(tf.nn.softplus(est_lambda), epsilon, 1e20)
    
    log_p_x = -tf.reduce_sum(tf.nn.log_poisson_loss(targets=data, log_input=tf.log(est_lambda), compute_full_loss=True), 1)
    
    output['log_p_x'] = log_p_x
    output['params'] = est_lambda
    output['samples'] = tf.contrib.distributions.Poisson(est_lambda).sample()

    return output


def loglik_test_real(theta, normalization_params, list_type):

    output = dict()
    epsilon = 1e-6

    # Data outputs
    data_mean, data_var = normalization_params
    data_var = np.clip(data_var, epsilon, np.inf)

    # Estimated parameters
    est_mean, est_var = theta
    soft_plus_est_var = np.log(1 + np.exp(-np.abs(est_var))) + np.maximum(est_var, 0)
    est_var = np.clip(soft_plus_est_var, epsilon, 1.0)  # Must be positive

    # Affine transformation of the parameters
    est_mean = np.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    # Outputs
    output['samples'] = np.random.normal(est_mean, np.sqrt(est_var))
    output['params'] = [est_mean, est_var]

    return output


def loglik_test_pos(theta, normalization_params, list_type):

    # Log-normal distribution
    output = dict()
    epsilon = 1e-6

    # Data outputs
    data_mean_log, data_var_log = normalization_params
    data_var_log = np.clip(data_var_log, epsilon, np.inf)

    est_mean, est_var = theta
    soft_plus_est_var = np.log(1 + np.exp(-np.abs(est_var))) + np.maximum(est_var, 0)
    est_var = np.clip(soft_plus_est_var, epsilon, 1.0)

    # Affine transformation of the parameters
    est_mean = np.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    output['samples'] = np.exp(np.random.normal(est_mean, np.sqrt(est_var))) - 1.0
    output['params'] = [est_mean, est_var]

    return output


def loglik_test_cat(theta, normalization_params, list_type):
    output = dict()

    # Data outputs
    log_pi = theta

    est_cat = Helpers.cat_sample(log_pi)
    estimated_samples = Helpers.indices_to_one_hot(est_cat, int(list_type['dim']))

    output['samples'] = estimated_samples
    output['params'] = log_pi

    return output


def loglik_test_ordinal(theta, normalization_params, list_type):
    output = dict()
    epsilon = 1e-6

    # We need to force that the outputs of the network increase with the categories
    partition_param, mean_param = theta

    batch_size = mean_param.shape[0]

    mean_value = mean_param.reshape(-1, 1)
    soft_plus_partition_param = np.log(1 + np.exp(-np.abs(partition_param))) + np.maximum(partition_param, 0)

    theta_values = np.cumsum(np.clip(soft_plus_partition_param, epsilon, 1e20), axis=1)
    sigmoid_est_mean = expit(theta_values - mean_value)
    mean_probs = np.c_[sigmoid_est_mean, np.ones(batch_size)] - np.c_[np.zeros(batch_size), sigmoid_est_mean]
    mean_probs = np.clip(mean_probs, epsilon, 1e20)

    mean_logits = np.log(mean_probs/(1-mean_probs))

    pseudo_cat = 1 + Helpers.cat_sample(mean_logits)

    output['samples'] = Helpers.sequence_mask(pseudo_cat, batch_size, int(list_type['dim']))
    output['params'] = mean_probs

    return output



def loglik_test_count(theta, normalization_params, list_type):
    output = dict()
    epsilon = 1e-6

    est_lambda = theta
    soft_plus_lambda = np.log(1 + np.exp(-np.abs(est_lambda))) + np.maximum(est_lambda, 0)
    est_lambda = np.clip(soft_plus_lambda, epsilon, 1e20)

    output['samples'] = np.random.poisson(est_lambda)
    output['params'] = est_lambda

    return output

