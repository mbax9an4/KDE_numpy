from KDE import eval_KDE, eval_KDE_matrix
from preproc_data import load_mnist, load_cifar
import numpy as np
import random
import json
import time
from sklearn.neighbors import KernelDensity

# build and fit a Gaussian KDE implemented in sklearn using the specified algorithm
def sklearn_KDE(train_set, test_set, algorithm, sigma):
    KDE_sklearn = KernelDensity(bandwidth=sigma, algorithm=algorithm)
    KDE_sklearn.fit(train_set)
    log_sklearn = KDE_sklearn.score(test_set)
    return log_sklearn/test_set.shape[0]

# verify that the mean log probbaility computed with my method is similar to that computed using sklearn (checking against both algorithms implemented)
def validate_result(config):
    eval_metrics = {
        'kd_tree':[],
        'ball_tree':[],
        'mine':[]}

    for r in [1, 10, 20]:
        config['rand_seed'] = r

        # load the dataset specified in the config file, this is where the seeds for numpy and python are also set
        if config['dset_name'] == "mnist":
            train_set, valid_set, test_set = load_mnist(config)
        elif "cifar" in config['dset_name']: 
            train_set, valid_set, test_set = load_cifar(config)

        # evaluate the sklearn KDE implementation with the two algorithms implemented
        for algorithm in ['kd_tree', 'ball_tree']:
            start = time.time()
            mlp = sklearn_KDE(train_set, test_set, algorithm, config['best_sigma'])
            end = time.time()

            # record the mean log prob computed by the method and the time it took to run
            eval_metrics[algorithm].append((mlp, end-start))
        
        # evaluate my method as well
        start = time.time()
        mlp = eval_KDE(train_set, test_set, config['best_sigma'])
        end = time.time()

        # record the mean log prob computed by the method and the time it took to run
        eval_metrics['mine'].append((mlp, end-start))

    # report the results
    print(f"The average mean log probability on the {config['dset_name']} dataset, computed using the sklearn (kd_tree) implementation was {np.mean([i[0] for i in eval_metrics['kd_tree']])} (+/-{np.std([i[0] for i in eval_metrics['kd_tree']])}) and it took on average {np.mean([i[1] for i in eval_metrics['kd_tree']])} (+/-{np.std([i[1] for i in eval_metrics['kd_tree']])}) seconds to run.")

    print(f"The average mean log probability on the {config['dset_name']} dataset, computed using the sklearn (ball_tree) implementation was {np.mean([i[0] for i in eval_metrics['ball_tree']])} (+/-{np.std([i[0] for i in eval_metrics['ball_tree']])}) and it took on average {np.mean([i[1] for i in eval_metrics['ball_tree']])} (+/-{np.std([i[1] for i in eval_metrics['ball_tree']])}) seconds to run.")

    print(f"The average mean log probability on the {config['dset_name']} dataset, computed using my implementation was {np.mean([i[0] for i in eval_metrics['mine']])} (+/-{np.std([i[0] for i in eval_metrics['mine']])}) and it took on average {np.mean([i[1] for i in eval_metrics['mine']])} (+/-{np.std([i[1] for i in eval_metrics['mine']])}) seconds to run.")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # validate the results on both datasets
    for dset_name in ['mnist', 'cifar']:
        config['dset_name'] = dset_name
        validate_result(config)