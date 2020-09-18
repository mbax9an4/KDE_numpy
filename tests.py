from KDE import eval_KDE, eval_KDE_matrix
from preproc_data import load_mnist, load_cifar
import numpy as np
import random
import json
import time
from sklearn.neighbors import KernelDensity

def validate_result():
    with open("config.json", "r") as f:
        config = json.load(f)

    for r in [2, 10, 20]:
        rand_seed = r
        config['rand_seed'] = r

        # set the python and numpy seeds
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        config['dset_name'] = "mnist"
        # load the mnist data
        train_set, valid_set, test_set = load_mnist(config)
        test_set = test_set[:2000]

        start = time.time()
        KDE_sklearn = KernelDensity(bandwidth=1)
        KDE_sklearn.fit(train_set)
        log_sklearn = KDE_sklearn.score(test_set)
        end = time.time()
        print(f'with a random seed of {r} the sklearn mean log prob on mnist is {log_sklearn/test_set.shape[0]} and it took {end-start}s')
        # print('sklearn mean log prob -740.8322056334938')

        best_sigma = 1
        start = time.time()
        mean_log_prob = eval_KDE(train_set, test_set, best_sigma)
        end = time.time()
        print(f'with a random seed of {r} my mean log prob on mnist is {mean_log_prob} and it took {end-start}s')

        start = time.time()
        mean_log_prob = eval_KDE_matrix(train_set, test_set, best_sigma, config['batch_size'])
        end = time.time()
        print(f'with a random seed of {r} my matrix mean log prob on mnist is {mean_log_prob} and it took {end-start}s')


        config['dset_name'] = "cifar"
        # load the mnist data
        train_set, valid_set, test_set = load_cifar(config)
        test_set = test_set[:2000]

        start = time.time()
        KDE_sklearn = KernelDensity(bandwidth=1)
        KDE_sklearn.fit(train_set)
        log_sklearn = KDE_sklearn.score(test_set)
        end = time.time()
        print(f'with a random seed of {r} the sklearn mean log prob on cifar is {log_sklearn/test_set.shape[0]} and it took {end-start}s')
        # print('sklearn mean log prob -2868.813658031753')

        best_sigma = 1
        start = time.time()
        mean_log_prob = eval_KDE(train_set, test_set, best_sigma)
        end = time.time()
        print(f'with a random seed of {r} my mean log prob on cifar is {mean_log_prob} and it took {end-start}s') 

        start = time.time()
        mean_log_prob = eval_KDE_matrix(train_set, test_set, best_sigma, config['batch_size'])
        end = time.time()
        print(f'with a random seed of {r} my matrix mean log prob on cifar is {mean_log_prob} and it took {end-start}s')


if __name__ == "__main__":
    validate_result()