import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist

# rejection sampling algorithm comes from LSE lecture notes
# alternatively see WOLFRAM: http://mathworld.wolfram.com/CirclePointPicking.html
# # http://mathworld.wolfram.com/HyperspherePointPicking.html

def unit_circumference_coordinates(r, n, coordinates):
    # r: radius
    # n: number of samples

    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)
    index = np.where((x1 ** 2 + x2 ** 2) < 1)  # accepted samples
    x1 = x1[index]
    x2 = x2[index]
    # coordinates
    x = ((x1) ** 2 - (x2) ** 2) / ((x1) ** 2 + (x2) ** 2) * r
    y = (2 * (x1) * (x2)) / ((x1) ** 2 + (x2) ** 2) * r

    a = coordinates[0]
    b = coordinates[1]  # 1x2 vector
    a = a + x
    b = b + y

    return a, b


def hyper_sphere_coordindates(n_search_samples, x, h, l, p):

    delta_x = np.random.randn(n_search_samples, x.shape[1])  # http://mathworld.wolfram.com/HyperspherePointPicking.html
    d = np.random.rand(n_search_samples) * (h - l) + l  # length range [l, h)
    norm_p = np.linalg.norm(delta_x, ord=p, axis=1)
    d_norm = np.divide(d, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_x = np.multiply(delta_x, d_norm)
    x_tilde = x + delta_x  # x tilde

    return x_tilde, d


def Laugel_Search(ncounterfactuals, out, search_samples, clf):

    # this function IS NOT GENERAL: works for "give me credit"
    x_tilde_star_list = []

    # Set parameters
    p = 2

    threshold = 200

    for i in range(ncounterfactuals):

        # Test data
        test_data_replicated = np.repeat(out['test_counter'][1][i, :].reshape(1, -1), search_samples, axis=0)
        test_data_c_replicated = np.repeat(out['test_counter'][2][i, :].reshape(1, -1), search_samples, axis=0)

        l = 0
        step = 0.5
        h = l + step

        # counter to stop
        count = 0
        counter_step = 1


        while True:

            count = count + counter_step

            if (count > threshold) is True:
                x_tilde_star = None
                break

            # STEP 1 of Algorithm
            # sample points on hyper sphere around test point
            x_tilde, _ = hyper_sphere_coordindates(search_samples, test_data_replicated, h, l, p)
            # one way: #x_tilde = np.ceil(x_tilde); another x_tilde = np.around(x_tilde,1)
            x_tilde = np.c_[test_data_c_replicated, x_tilde]

            # STEP 2 of Algorithm
            # compute l_1 distance
            distances = np.abs((x_tilde - np.c_[test_data_c_replicated, test_data_replicated])).sum(axis=1)

            # counterfactual labels
            y_tilde = clf.predict(x_tilde)
            cla_index = np.where(y_tilde != 1)

            x_tilde_candidates = x_tilde[cla_index]
            candidates_dist = distances[cla_index]

            if len(candidates_dist) == 0:  # no candidate generated
                l = h
                h = l + step
            else:  # certain candidates generated
                min_index = np.argmin(candidates_dist)
                x_tilde_star = x_tilde_candidates[min_index]
                break

        x_tilde_star_list.append(x_tilde_star)
    X_test_counterfactual = np.array(x_tilde_star_list)

    return X_test_counterfactual