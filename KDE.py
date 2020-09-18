import numpy as np
from tqdm import tqdm
import json
import math
from preproc_data import load_mnist, load_cifar
import matplotlib.pyplot as plt

# @profile
# compute the mean log probability over the test data
def eval_KDE(train_set, test_set, sigma):
    # compute the constant terms of the likelihood 
    denom = -1*2.0*math.pow(sigma,2)
    norm = -1.0/2 * math.log(2.0*math.pi*math.pow(sigma,2))
    prior = math.log(1.0/train_set.shape[0])

    # variable to record the mean log prob
    mean_log_prob = 0.0
    
    # for each example in test compute the likelihood - average 
    for test_ex in tqdm(test_set, desc=f"Sigma: {sigma}"):
    # for test_ex in test_set:
        # import pdb; pdb.set_trace()
    
        # compute the squared distance between the test vector and each example in the train matrix
        cond_prob_features = np.square(test_ex - train_set)

        # divide the distances by the denominator
        cond_prob_features_norm = np.true_divide(cond_prob_features, denom) + norm

        # add the probabilities over the features
        cond_prob_features = np.sum(cond_prob_features_norm, axis=1)

        # check that we have reduced the right dimension
        assert cond_prob_features.shape == (train_set.shape[0], )

        # add the prior
        cond_prob_features = cond_prob_features + prior

        # to reduce the risk of overflows we perform the following operations using a log-sum-exp numerical trick 
        shift_constant = np.max(cond_prob_features)

        # raise the values to exp, while shifting the values to ensure no overflow occurs
        likelihoods = np.exp(cond_prob_features - shift_constant)

        mean_log_prob += math.log(np.sum(likelihoods)) + shift_constant
        
        assert mean_log_prob < math.inf

    return mean_log_prob/test_set.shape[0]

# @profile
def eval_KDE_matrix(train_set, test_set, sigma, batch_size):
    # compute the constant terms of the likelihood 
    denom = -1*2.0*math.pow(sigma,2)
    norm = -1.0/2 * math.log(2.0*math.pi*math.pow(sigma,2))
    prior = math.log(1.0/train_set.shape[0])

    # variable to record the mean log probability over the given dataset
    mean_log_prob = 0.0

    for b in range(0, test_set.shape[0], batch_size):
        # import pdb; pdb.set_trace()

        # compute the distance between each test example and all train examples 
        diff_matrix = np.square(test_ex[:, np.newaxis] - train_set)

        # divide the distances by the denominator
        cond_prob_features_norm = np.true_divide(diff_matrix, denom) + norm

        # add the probabilities over the features
        cond_prob_features = np.sum(cond_prob_features_norm, axis=2)

        # check that we have reduced over the right dimension
        assert cond_prob_features.shape == (batch_size, train_set.shape[0], )

        # add the prior
        cond_prob_features = cond_prob_features + prior

        # find the maximum value at each test example
        shift_constant = np.array([np.ones(train_set.shape[0])*np.max(e) for e in cond_prob_features])

        # raise the values to exp, while shifting the values to ensure no overflow occurs
        likelihoods = np.exp(cond_prob_features - shift_constant)

        log_probs = np.add(np.log(np.sum(likelihoods, axis=1)), shift_constant[:,0])

        mean_log_prob += np.sum(log_probs)

    return mean_log_prob/test_set.shape[0]


# fit the best sigma value
def fit_sigma(train_set, valid_set):
    # sigma values over which we grid search
    config['sigma_grid'] = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3]

    mean_log_prob = []
    for sigma in config['sigma_grid']:
        mean_log_prob.append(eval_KDE(train_set, valid_set, sigma))

    plt.figure()
    plt.plot(config['sigma_grid'], mean_log_prob)
    plt.savefig(f"{config['dset_name']}_sigma_grid.pdf", format='pdf')

    return sorted(zip(config['sigma_grid'], mean_log_prob), key=lambda x: x[1])[-1][0]

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # load the mnist data
    config['dset_name'] = "mnist"
    train_set, valid_set, test_set = load_mnist(config)

    # best_sigma = fit_sigma(train_set, valid_set)
    best_sigma = 0.5
    
    mean_log_prob = eval_KDE(train_set, test_set, best_sigma)
    print(mean_log_prob)

    # mean_log_prob = eval_KDE_matrix(train_set, test_set, best_sigma, 5)
    # print(mean_log_prob)

    # load the cifar100 data
    config['dset_name'] = "cifar"
    train_set, valid_set, test_set = load_cifar(config)

    best_sigma = fit_sigma(train_set, valid_set)
    # best_sigma = 0.25

    mean_log_prob = eval_KDE(train_set, test_set, best_sigma)
    print(mean_log_prob)
