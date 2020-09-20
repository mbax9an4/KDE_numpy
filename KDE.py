import numpy as np
from tqdm import tqdm
import json
import math
import time
from preproc_data import load_mnist, load_cifar
import matplotlib.pyplot as plt

# flag necessary for line profiler
# @profile
# compute the mean log probability over the test/validation data
def eval_KDE(train_set, test_set, sigma):
    # compute the constant terms of the likelihood computation
    denom = -1*2.0*math.pow(sigma,2)
    norm = -1.0/2 * math.log(2.0*math.pi*math.pow(sigma,2))
    prior = math.log(1.0/train_set.shape[0])

    # variable to record the mean log prob
    mean_log_prob = 0.0
    
    # for each example in test compute the likelihood. tqdm is used to produce print a progress bar of the computation
    for test_ex in tqdm(test_set, desc=f"Sigma: {sigma}"):
    # line-profiler does not work with tqdm so use the line below instead
    # for test_ex in test_set:
    
        # compute the squared distance between the test vector and all examples in the train matrix
        cond_prob_features = np.square(test_ex - train_set)

        # divide the distances by the denominator
        cond_prob_features_norm = np.true_divide(cond_prob_features, denom) + norm

        # add the probabilities over each feature
        cond_prob_features = np.sum(cond_prob_features_norm, axis=1)

        # check that we have reduced the right dimension
        assert cond_prob_features.shape == (train_set.shape[0], )

        # add the prior
        cond_prob_features = cond_prob_features + prior

        # to reduce the risk of overflows we perform the following operations using a log-sum-exp numerical trick, thus we find the largest value in the vector
        shift_constant = np.max(cond_prob_features)

        # raise the values to exp, while shifting the values to ensure no overflow occurs
        likelihoods = np.exp(cond_prob_features - shift_constant)

        # sum over the training example likelihoods, undo the exp, and shift the result by the constant found earlier 
        mean_log_prob += math.log(np.sum(likelihoods)) + shift_constant
        
        # check that the log-sum-exp trick worked and that the probabilities were successfully computed
        assert mean_log_prob < math.inf

    # return the mean log prob over the test/validation set probabilities
    return mean_log_prob/test_set.shape[0]

# @profile
# Gaussian KDE implementation where we compute the log probability over a batch of test examples at the same time
# I check to see if computing these operations over several test examples at the same time is more efficient than the for loop in the eval_KDE function
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
        diff_matrix = np.square(test_ex[b:b+batch_size, np.newaxis] - train_set)

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


# fit the best sigma value and plot the results
def fit_sigma(config):
    mean_log_probs = []
    std_log_probs = []

    # go through all the sigma values specified in the config file and calculate the mean log probability on the sampled validation set
    for sigma in config['sigma_grid']:
        # variable to record the mean log probability values for each random seed
        mlp = []

        # the random seed affects the sampled training/validation datasets, where were fixed for this experiment to facilitate reproducibility
        for rand_seed in [1, 10, 20, 30, 40]:
            config['rand_seed'] = rand_seed

            # load the dataset specified in the config file
            if config['dset_name'] == "mnist":
                train_set, valid_set, test_set = load_mnist(config)
            elif "cifar" in config['dset_name']: 
                train_set, valid_set, test_set = load_cifar(config)

            mlp.append(eval_KDE(train_set, valid_set[:10], sigma))

        # record the mean and std of the mean log prob over the 5 random seeds specified
        mean_log_probs.append(np.mean(mlp))
        std_log_probs.append(np.std(mlp))

    # plot the recorded means and std for the sigma values evaluated and save the plot as a pdf
    plt.figure()
    plt.errorbar(config['sigma_grid'], mean_log_probs, yerr=std_log_probs, ecolor='k', alpha=0.6, marker='.', capsize=2)
    plt.xlabel('Sigma')
    plt.ylabel('Mean log probability')
    plt.tight_layout()
    plt.savefig(f"{config['dset_name']}_sigma_grid.pdf", format='pdf')

    # find the sigma value that has produced the best mean log prob
    optimal = sorted(zip(config['sigma_grid'], mean_log_probs), key=lambda x: x[1])[-1]
    config['best_sigma'] = optimal[0]
    print(f"The highest mean log probability (of {optimal[1]}), on the {config['dset_name']} dataset, has been recorded when sigma={optimal[0]}.")

    return optimal[0]

if __name__ == "__main__":
    # load the experiment config file
    with open("config.json", "r") as f:
        config = json.load(f)

    if config['search_sigma'] == 1:
        # perform a grid search to tune the value of sigma on the specified dataset
        best_sigma = fit_sigma(config)
    else:
        # I give the option for the random seed in the config to be a list of values, in which case we evaluate the mean log prob for each and report the average value and the average time it took to run
        if isinstance(config['rand_seed'], list):
            eval_metrics = []

            random_seeds = config['rand_seed']
            for r in random_seeds:
                config['rand_seed'] = r

                # load the dataset specified in the config file
                if config['dset_name'] == "mnist":
                    train_set, valid_set, test_set = load_mnist(config)
                elif "cifar" in config['dset_name']: 
                    train_set, valid_set, test_set = load_cifar(config)
            
                # compute the mean log prob and time the function
                start = time.time()
                mean_log_prob = eval_KDE(train_set, test_set, config['best_sigma'])
                end = time.time()

                eval_metrics.append((mean_log_prob, end-start))

            print(f"The mean log probability computed on the {config['dset_name']} test set with a sigma={config['best_sigma']} is {np.mean([i[0] for i in eval_metrics])} (+/-{np.std([i[0] for i in eval_metrics])}) and it took on average {np.mean([i[1] for i in eval_metrics])} (+/-{np.std([i[1] for i in eval_metrics])}) seconds to run.")
        else:
            # load the dataset specified in the config file
            if config['dset_name'] == "mnist":
                train_set, valid_set, test_set = load_mnist(config)
            elif "cifar" in config['dset_name']: 
                train_set, valid_set, test_set = load_cifar(config)
        
            # compute the mean log prob and time the function
            start = time.time()
            mean_log_prob = eval_KDE(train_set, test_set, config['best_sigma'])
            end = time.time()

            print(f"The mean log probability computed on the {config['dset_name']} test set with a sigma={config['best_sigma']} is {mean_log_prob} and it took {end-start} seconds to run.")