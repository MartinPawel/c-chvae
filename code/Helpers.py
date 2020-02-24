
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.stats import moment
import csv
import argparse




# Argument Parser
def getArgs(argv=None):
    parser = argparse.ArgumentParser(description='Default parameters of the models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=100, help='Size of the batches')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs of the simulations')
    parser.add_argument('--train', type=int, default=1, help='Training model flag')
    parser.add_argument('--display', type=int, default=1, help='Display option flag')
    parser.add_argument('--save', type=int, default=1000, help='Save variables every save iterations')
    parser.add_argument('--restore', type=int, default=0, help='To restore session, to keep training or evaluation')
    parser.add_argument('--dim_latent_s', type=int, default=3, help='Dimension of the categorical space')
    parser.add_argument('--dim_latent_z', type=int, default=2, help='Dimension of the Z latent space')
    parser.add_argument('--dim_latent_y', type=int, default=5, help='Dimension of the Y latent space')
    parser.add_argument('--dim_latent_y_partition', type=int, nargs='+', help='Partition of the Y latent space')
    parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
    parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
    parser.add_argument('--data_file_c', type=str, default='MNIST_data', help='File with the conditioning data')
    parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
    parser.add_argument('--types_file_c', type=str, default='mnist_train_types2.csv', help='File with the types of the conditioning data')
    parser.add_argument('--classifier', type=str, default='RLinearR', help='Classification model (RandomForest, SVM or else RLinearR)')
    parser.add_argument('--classifier_two', type=str, default='RandomForest', help='Classification model (RandomForest, SVM or else RLinearR)')
    parser.add_argument('--norm_latent_space', type=int, default=2, help='To measure distance between latent variables')
    parser.add_argument('--step_size', type=float, default=0.5, help='Step size for Random Search')
    parser.add_argument('--search_samples', type=int, default=1000, help='Nunber search samples for counterfactual search')
    parser.add_argument('--data_y_file', type=str, default='cs_y_training', help='File with the y data')
    parser.add_argument('--ncounterfactuals', type=int, default=25, help='First #counterf. test data points for which we find counterf.')
    parser.add_argument('--boundary', type=float, default=-0.5, help='Boundary y = def. for simple classifier')
    parser.add_argument('--degree_active', type=float, default=1, help='active latent variable threshold')

    return parser.parse_args(argv)


def next_batch(data, types_dict, batch_size, index_batch):

    # Create minibath
    batch_xs = data[index_batch * batch_size:(index_batch + 1) * batch_size, :]

    # Slipt variables of the batches
    data_list = []
    initial_index = 0
    for d in types_dict:
        dim = int(d['dim'])
        data_list.append(batch_xs[:, initial_index:initial_index + dim])
        initial_index += dim

    return data_list

def next_batch_y(y, batch_size, index_batch):
    return y[index_batch * batch_size:(index_batch + 1) * batch_size, :]





def samples_concatenation(samples):
    for i, batch in enumerate(samples):
        if i == 0:
            samples_x = np.concatenate(batch['x'], 1)
            samples_y = batch['y']
            samples_z = batch['z']
            samples_s = batch['s']
        else:
            samples_x = np.concatenate([samples_x, np.concatenate(batch['x'], 1)], 0)
            samples_y = np.concatenate([samples_y, batch['y']], 0)
            samples_z = np.concatenate([samples_z, batch['z']], 0)
            samples_s = np.concatenate([samples_s, batch['s']], 0)

    return samples_s, samples_z, samples_y, samples_x


def discrete_variables_transformation(data, types_dict):
    ind_ini = 0
    output = []
    for d in range(len(types_dict)):
        ind_end = ind_ini + int(types_dict[d]['dim'])
        if types_dict[d]['type'] == 'cat':
            output.append(np.reshape(np.argmax(data[:, ind_ini:ind_end], 1), [-1, 1]))
        elif types_dict[d]['type'] == 'ordinal':
            output.append(np.reshape(np.sum(data[:, ind_ini:ind_end], 1) - 1, [-1, 1]))
        else:
            output.append(data[:, ind_ini:ind_end])
        ind_ini = ind_end

    return np.concatenate(output, 1)


def read_data(data_file, types_file):
    # Read types of data from data file
    with open(data_file, 'r') as f:
        data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
        data = np.array(data)

    # Read types of data from data file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]

    # Construct the data matrices
    data_complete = []
    for i in range(np.shape(data)[1]):

        if types_dict[i]['type'] == 'cat':
            # Get categories
            cat_data = [int(x) for x in data[:, i]]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['dim']))
            cat_data = new_categories[indexes]
            # Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0], len(new_categories)])
            aux[np.arange(np.shape(data)[0]), cat_data] = 1
            data_complete.append(aux)

        elif types_dict[i]['type'] == 'ordinal':
            # Get categories
            cat_data = [int(x) for x in data[:, i]]
            categories, indexes = np.unique(cat_data, return_inverse=True)
            # Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i]['dim']))
            cat_data = new_categories[indexes]
            # Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0], 1 + len(new_categories)])
            aux[:, 0] = 1
            aux[np.arange(np.shape(data)[0]), 1 + cat_data] = -1
            aux = np.cumsum(aux, 1)
            data_complete.append(aux[:, :-1])

        else:
            data_complete.append(np.transpose([data[:, i]]))

    n_samples = np.shape(data)[0]
    # n_variables = len(types_dict)

    data = np.concatenate(data_complete, 1)

    return data, types_dict, n_samples


def p_distribution_params_concatenation(params, types_dict, z_dim, s_dim):
    keys = params[0].keys()
    out_dict = {key: [] for key in keys}

    for i, batch in enumerate(params):

        for d, k in enumerate(keys):

            if k == 'z' or k == 'y':
                if i == 0:
                    out_dict[k] = batch[k]
                else:
                    out_dict[k] = np.concatenate([out_dict[k], batch[k]], 1)

            elif k == 'x':
                if i == 0:
                    out_dict[k] = batch[k]
                else:
                    for v in range(len(types_dict)):
                        if types_dict[v]['type'] == 'pos' or types_dict[v]['type'] == 'real':
                            out_dict[k][v] = np.concatenate([out_dict[k][v], batch[k][v]], 1)
                        else:
                            out_dict[k][v] = np.concatenate([out_dict[k][v], batch[k][v]], 0)

    return out_dict


def q_distribution_params_concatenation(params, z_dim, s_dim):
    keys = params[0].keys()
    out_dict = {key: [] for key in keys}

    for i, batch in enumerate(params):
        for d, k in enumerate(keys):
            out_dict[k].append(batch[k])

    out_dict['z'] = np.concatenate(out_dict['z'], 1)
    out_dict['s'] = np.concatenate(out_dict['s'], 0)

    return out_dict


def statistics(loglik_params, types_dict):
    loglik_mean = []
    loglik_mode = []

    for d, attrib in enumerate(loglik_params):
        if types_dict[d]['type'] == 'real':
            # Normal distribution (mean, sigma)
            loglik_mean.append(attrib[0])
            loglik_mode.append(attrib[0])
            # Only for log-normal
        elif types_dict[d]['type'] == 'pos':
            # Log-normal distribution (mean, sigma)
            loglik_mean.append(np.exp(attrib[0] + 0.5 * attrib[1]) - 1.0)
            loglik_mode.append(np.exp(attrib[0] - attrib[1]) - 1.0)
        elif types_dict[d]['type'] == 'count':
            # Poisson distribution (lambda)
            loglik_mean.append(attrib)
            loglik_mode.append(np.floor(attrib))

        else:
            # Categorical and ordinal (mode imputation for both)
            loglik_mean.append(np.reshape(np.argmax(attrib, 1), [-1, 1]))
            loglik_mode.append(np.reshape(np.argmax(attrib, 1), [-1, 1]))

    return np.transpose(np.squeeze(loglik_mean)), np.transpose(np.squeeze(loglik_mode))


def error_computation(x_train, x_hat, types_dict):
    error_observed = []
    ind_ini = 0
    for dd in range(len(types_dict)):

        # Mean classification error
        if types_dict[dd]['type'] == 'cat':
            ind_end = ind_ini + 1
            error_observed.append(np.mean(x_train[:, ind_ini:ind_end] != x_hat[:, ind_ini:ind_end]))

        # Mean "shift" error
        elif types_dict[dd]['type'] == 'ordinal':
            ind_end = ind_ini + 1
            error_observed.append(
                np.mean(np.abs(x_train[:, ind_ini:ind_end] - x_hat[:, ind_ini:ind_end])) / int(types_dict[dd]['dim']))

        # Normalized root mean square error
        else:
            ind_end = ind_ini + int(types_dict[dd]['dim'])
            norm_term = np.max(x_train[:, dd]) - np.min(x_train[:, dd])
            error_observed.append(
                np.sqrt(mean_squared_error(x_train[:, ind_ini:ind_end], x_hat[:, ind_ini:ind_end])) / norm_term)

        ind_ini = ind_end

    return error_observed


def place_holder_types(types_file, batch_size):
    # Read the types of the data from the files
    with open(types_file) as f:
        types_list = [{k: v for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]

    # Create placeholders for every data type, with appropriate dimensions
    batch_data_list = []
    for i in range(len(types_list)):
        batch_data_list.append(tf.placeholder(tf.float32, shape=(None, types_list[i]['dim'])))
    tf.concat(batch_data_list, axis=1)

    return batch_data_list, types_list


def batch_normalization(batch_data_list, types_list, batch_size):
    normalized_data = []
    normalization_parameters = []
    noisy_data = []

    for i, d in enumerate(batch_data_list):

        observed_data = d

        if types_list[i]['type'] == 'real':
            # We transform the data to a gaussian with mean 0 and std 1
            data_mean, data_var = tf.nn.moments(observed_data, 0)
            data_var = tf.clip_by_value(data_var, 1e-6, 1e20)  # Avoid zero values
            aux_X = tf.nn.batch_normalization(observed_data, data_mean, data_var, offset=0.0, scale=1.0,
                                              variance_epsilon=1e-6)

            aux_X_noisy = aux_X + tf.random_normal((batch_size, 1), 0, 0.05, dtype=tf.float32)

            normalized_data.append(aux_X)
            noisy_data.append(aux_X_noisy)
            normalization_parameters.append([data_mean, data_var])

        # When using log-normal
        elif types_list[i]['type'] == 'pos':

            # We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data_log = tf.log(1 + observed_data)
            data_mean_log, data_var_log = tf.nn.moments(observed_data_log, 0)
            data_var_log = tf.clip_by_value(data_var_log, 1e-6, 1e20)  # Avoid zero values
            aux_X = tf.nn.batch_normalization(observed_data_log, data_mean_log, data_var_log, offset=0.0, scale=1.0,
                                              variance_epsilon=1e-6)

            normalized_data.append(aux_X)
            normalization_parameters.append([data_mean_log, data_var_log])

        elif types_list[i]['type'] == 'count':

            # We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data_log = tf.log(1 + observed_data)
            data_mean_log, data_var_log = tf.nn.moments(observed_data_log, 0)
            data_var_log = tf.clip_by_value(data_var_log, 1e-6, 1e20)  # Avoid zero values
            aux_X = tf.nn.batch_normalization(observed_data_log, data_mean_log, data_var_log, offset=0.0, scale=1.0,
                                              variance_epsilon=1e-6)

            normalized_data.append(aux_X)
            normalization_parameters.append([data_mean_log, data_var_log])


        else:
            # Don't normalize the categorical and ordinal variables
            normalized_data.append(d)
            normalization_parameters.append(tf.convert_to_tensor([0.0, 1.0], dtype=tf.float32))  # No normalization here

            aux_X_noisy = d + tf.random_normal((batch_size, 1), 0, 0.05, dtype=tf.float32)
            noisy_data.append(aux_X_noisy)


    return normalized_data, normalization_parameters, noisy_data


# normalization function

def normalization_classification(batch_data_list, types_list):
    normalized_data = []
    normalization_parameters = []

    for i in range(len(types_list)):

        observed_data = batch_data_list[:, i]

        if types_list[i]['type'] == 'real':
            # We transform the data to a gaussian with mean 0 and std 1
            data_mean = np.mean(observed_data)
            data_var = moment(observed_data, 2)
            data_var = np.clip(data_var, 1e-6, 1e20)
            data_std = np.sqrt(data_var)
            aux_X = preprocessing.scale(observed_data)

            normalized_data.append(aux_X)
            normalization_parameters.append([data_mean, data_std])

        # When using log-normal
        elif types_list[i]['type'] == 'pos':
            #           #We transform the log of the data to a gaussian with mean 0 and std 1
            observed_data = observed_data
            data_mean = np.mean(observed_data)
            data_var = moment(observed_data, 2)
            data_var = np.clip(data_var, 1e-6, 1e20)  # Avoid zero values
            data_std = np.sqrt(data_var)

            aux_X = preprocessing.scale(observed_data)

            normalized_data.append(aux_X)
            normalization_parameters.append([data_mean, data_std])

        elif types_list[i]['type'] == 'count':

            # Input log of the data
            observed_data = observed_data
            data_mean = np.mean(observed_data)
            data_var = moment(observed_data, 2)
            data_var = np.clip(data_var, 1e-6, 1e20)  # Avoid zero values
            data_std = np.sqrt(data_var)

            aux_X = preprocessing.scale(observed_data)

            normalized_data.append(aux_X)
            normalization_parameters.append([data_mean, data_std])

        else:
            # Don't normalize the categorical and ordinal variables
            normalized_data.append(observed_data)
            normalization_parameters.append([0.0, 1.0])  # No normalization here

    return normalized_data, normalization_parameters



def replicate_data_list(data_list, num_replications):
    # data_list: expected to have 1 row
    # num_replications: expected to have #rows = nsamples
    new_data_list = []

    for i in range(len(data_list)):
        if i == 0:
            new_data_list = [np.repeat(data_list[i], num_replications, axis=0)]
        else:
            new_data_list.append(np.repeat(data_list[i], num_replications, axis=0))

    return new_data_list


# stylised classifier
def f_star(x_tilde, boundary):
    y = x_tilde[:,1] > boundary
    y = y*1
    return y


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def sequence_mask(pseudo_cat, dim_ord, batch_size):
    x = np.linspace(1, dim_ord, dim_ord).reshape(1, -1)
    x = ~(np.repeat(x, batch_size, axis=0).T > pseudo_cat).T
    x = x * 1
    return x


def cat_sample(logits):
    u = np.random.uniform(0, 1, logits.shape)
    return np.argmax(logits - np.log(-np.log(u)), axis=1)


def Compute_LOF(neighbors, x_train, x_test):
    # x_test: - np array
    # x_test_counterfactual: - np array
    # x_train: train data  - np array

    clf = LocalOutlierFactor(n_neighbors=neighbors, contamination=0.01, novelty=True)
    clf.fit(x_train)

    X_outlier = clf.predict(x_test)

    return X_outlier


def Connectedness(x_train, x_counter, number, epsilon, min_samples):
    x_counter.shape

    dbscan_list = []
    n, _ = x_counter.shape

    for i in range(n):
        density_control = np.r_[x_train[0:number, :], x_counter[i, :].reshape(1, -1)]
        density_pred = DBSCAN(eps=epsilon, min_samples=min_samples).fit(density_control)
        dbscan_list.append(density_pred.labels_[-1])

    not_connected = np.array(dbscan_list.count(-1)) / n #count occurcene of (-1) labels & divide by number of test set

    return not_connected, np.array(dbscan_list)


def Read_Split_Data(test_size, classifier, data_total, data_total_c, y_true, types_dict, types_dict_c, normalization):
    out = dict()

    # out_training: training x and y
    # out_test: test x and y
    # out_train_pos: x with corresponding positive predicted label on train set
    # out_test_counter: x with corresponding negative predicted label on test set


    # Split into test and train data
    train_data, test_data, train_data_c, test_data_c, y_train, y_test = train_test_split(data_total,
                                                                                         data_total_c,
                                                                                         y_true,
                                                                                         random_state=619,
                                                                                         test_size=test_size)

    n_train, _ = np.shape(train_data)
    df = np.r_[train_data, test_data]
    df_c = np.r_[train_data_c, test_data_c]

    df_norm, df_param = normalization_classification(df, types_dict)
    df_norm = np.transpose(np.array(df_norm))
    df_c_norm, df_c_param = normalization_classification(df_c, types_dict_c)
    df_c_norm = np.transpose(np.array(df_c_norm))

    train_data_norm = df_norm[0:n_train, :]
    test_data_norm = df_norm[n_train::, :]
    train_data_c_norm = df_c_norm[0:n_train, :]
    test_data_c_norm = df_c_norm[n_train::, :]

    # Concatenate free and conditioning features
    train_concat = np.c_[train_data_c, train_data]
    test_concat = np.c_[test_data_c, test_data]
    train_concat_norm = np.c_[train_data_c_norm, train_data_norm]
    test_concat_norm = np.c_[test_data_c_norm, test_data_norm]


    if normalization == True:
        train_concat_x = train_concat_norm
        test_concat_x = test_concat_norm

        # not normalized
        train_data_not = train_data
        train_data_c_not = train_data_c
        train_data_concat_not = np.c_[train_data_c_not, train_data_not]

        test_data_not = test_data
        test_data_c_not = test_data_c
        test_data_concat_not = np.c_[test_data_c_not, test_data_not]

        # normalized data
        train_data = train_data_norm
        train_data_c = train_data_c_norm
        test_data = test_data_norm
        test_data_c = test_data_c_norm

    else:
        train_concat_x = train_concat
        test_concat_x = test_concat


    # classifcation model training: Random forest or LR model: use default values
    if classifier == 'RandomForest':
        clf = RandomForestClassifier(random_state=619)

        param_grid = {'bootstrap': [True],
                        'max_depth': [3, 5, 7],
                        'min_samples_leaf': [5],
                        'min_samples_split': [4, 10],
                        'n_estimators': [50, 100]}

        grid = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            scoring='roc_auc',
                            cv=3,
                            n_jobs=-1,
                            verbose=2)

        grid.fit(train_concat_x, y_train.reshape(-1))
        clf = grid.best_estimator_


        inv_y_train = 1 - y_train
        ## grid search
        clf_ar = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=619)
        grid_ar = GridSearchCV(
            clf_ar, param_grid={'C': np.logspace(-4, 3)},
            cv=5,
            scoring='roc_auc',
            return_train_score=True)
        grid_ar.fit(train_concat_x, inv_y_train.reshape(-1))
        clf_ar = grid_ar.best_estimator_


    elif classifier == 'SVM':

        clf = SVC(random_state=619)

        tuned_parameters = [{'kernel': ['rbf'], 'C': [0.01, 1, 10]}]
        # tuned_parameters = [{'alpha': [0.0001, 0.001]}]

        grid = GridSearchCV(clf, tuned_parameters, cv=3, n_jobs=-1)
        grid.fit(train_concat_x, y_train.reshape(-1))
        clf = grid.best_estimator_
        print(grid.cv_results_)

        # for AR algorithm as placeholder
        inv_y_train = 1 - y_train
        ## grid search
        clf_ar = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=619)
        grid_ar = GridSearchCV(
            clf_ar, param_grid={'C': np.logspace(-4, 3)},
            cv=5,
            scoring='roc_auc',
            return_train_score=True)
        grid_ar.fit(train_concat_x, inv_y_train.reshape(-1))
        clf_ar = grid_ar.best_estimator_


    else:

        ## grid search
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=619)
        grid = GridSearchCV(
            clf, param_grid={'C': np.logspace(-4, 3)},
            cv=5,
            scoring='roc_auc',
            return_train_score=True)
        grid.fit(train_concat_x, y_train.reshape(-1))
        clf = grid.best_estimator_

    # for AR algorithm
        inv_y_train = 1 - y_train
        ## grid search
        clf_ar = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=619)
        grid_ar = GridSearchCV(
            clf_ar, param_grid={'C': np.logspace(-4, 3)},
            cv=5,
            scoring='roc_auc',
            return_train_score=True)
        grid_ar.fit(train_concat_x, inv_y_train.reshape(-1))
        clf_ar = grid_ar.best_estimator_



    # THESE GUYS/GURLS WILL NEED OUR HELP (TEST SET)
    index_predicted_denied = np.where(clf.predict(test_concat_x) == 1)[0]

    test_data_c_denied = test_data_c[index_predicted_denied, :]
    test_data_denied = test_data[index_predicted_denied, :]
    y_test_denied = y_test[index_predicted_denied]
    test_concat_x_denied = np.c_[test_data_c_denied, test_data_denied]
    ncounterfactuals, _ = test_concat_x_denied.shape

    # FROM THESE GUYS WE HAVE POSTIVE RECORD & they have predicted positive record (TRAINING SET)
    index_predicted_nodefault = (clf.predict(train_concat_x) == 0)
    index_true_nodefault = (y_train.reshape(-1) == 0)
    intersection_no = (index_predicted_nodefault * 1 + index_true_nodefault * 1)
    index_intersection_no = (intersection_no == 2)  #(nodefault + predicted nodefault) index

    train_data_c_pos = train_data_c[index_intersection_no, :]
    train_data_pos = train_data[index_intersection_no, :]
    train_concat_x_pos = np.c_[train_data_c_pos, train_data_pos]
    y_train_pos = y_train[index_intersection_no]

    if normalization == True:

        test_data_denied_not = test_data_not[index_predicted_denied, :]
        test_data_c_denied_not = test_data_c_not[index_predicted_denied, :]
        test_concat_x_denied_not = np.c_[test_data_c_denied_not, test_data_denied_not]

        train_data_c_pos_not = train_data_c_not[index_intersection_no, :]
        train_data_pos_not = train_data_not[index_intersection_no, :]
        train_concat_x_pos_not = np.c_[train_data_c_pos_not, train_data_pos_not]

    else:

        test_concat_x_denied_not = _

        test_data_denied_not = _
        test_data_c_denied_not = _
        train_concat_x_pos_not = _

        test_data_concat_not = _
        train_data_not = _
        train_data_c_not = _
        train_data_concat_not = _


    # return
    out['training'] = [train_concat_x, train_data, train_data_c, y_train]
    out['training_not'] = [train_data_concat_not, train_data_not, train_data_c_not, y_train]
    out['test'] = [test_concat_x, y_test]
    out['test_not'] = [test_data_concat_not, y_test]
    out['test_counter'] = [test_concat_x_denied, test_data_denied, test_data_c_denied, y_test_denied]
    out['test_counter_not'] = [test_concat_x_denied_not, test_data_denied_not, test_data_c_denied_not, y_test_denied]
    out['train_pos'] = [train_concat_x_pos, y_train_pos]
    out['train_pos_not'] = [train_concat_x_pos_not, y_train_pos]
    out['normalization_parameters'] = [df_param, df_c_param]

    return ncounterfactuals, clf, out, clf_ar, grid


def compute_cdf(data):
    # per free feature
    # relies on computing histogram first
    # num_bins: # bins in histogram
    # you can use bin_edges & norm_cdf to plot cdf

    n, p = np.shape(data)
    # num_bins = n
    norm_cdf = np.zeros((n, p))

    for j in range(p):
        counts, bin_edges = np.histogram(data[:, j], bins=n, normed=True)
        cdf = np.cumsum(counts)
        norm_cdf[:, j] = cdf / cdf[-1]
        # plt.plot (bin_edges[1:], norm_cdf)

    return bin_edges[1:], norm_cdf


def max_percentile_shift(norm_cdfs, norm_cdfs_counterfactual):
    # (3) in ustun et al
    delta_cdfs = np.abs(norm_cdfs - norm_cdfs_counterfactual)
    cost = np.max(delta_cdfs, 1)
    return cost


def total_percentile_shift(norm_cdfs, norm_cdfs_counterfactual):
    inv_counterfactual = norm_cdfs_counterfactual
    inv = norm_cdfs
    ratio = np.abs(inv_counterfactual - inv)
    cost = np.sum(ratio, 1)
    return cost


def total_log_percentile_shift(norm_cdfs, norm_cdfs_counterfactual):
    # (4) in ustun et al
    inv_counterfactual = np.clip(1-norm_cdfs_counterfactual, 0.01, 0.99)
    inv = np.clip(1-norm_cdfs, 0.01, 0.99)
    ratio = np.abs(np.log(np.clip((inv_counterfactual/inv), 0.01, 10)))
    cost = np.sum(ratio, 1)
    return cost


def denormalization(norm_para, norm_para_c, samples, samples_c):

    # norm_para & norm_para_c: numpy arrays
    # samples: numpy arrays

    n, p = np.shape(samples)
    n_c, p_c = np.shape(samples_c)

    norm_samples = np.zeros((n, p))
    norm_samples_c = np.zeros((n_c, p_c))

    for i in range(p):
        norm_samples[:, i] = (samples[:, i] - norm_para[i, 0])/norm_para[i, 1]

    for i in range(p_c):
        norm_samples_c[:, i] = (samples_c[:, i] - norm_para_c[i, 0])/norm_para_c[i, 1]

    return norm_samples, norm_samples_c


# standardize data
def standardize(data):
    scaler = StandardScaler()
    a = scaler.fit(data)
    a = scaler.transform(data)

    return a, scaler

# reduce dim of data
def reduce_dim(data, dim):
    pca = PCA(n_components= dim)
    components = pca.fit_transform(data)
    return components