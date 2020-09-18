import gzip
import pickle
import random 
import numpy as np

# load the mnist dataset
def load_mnist(config):
    with gzip.open('mnist.pkl.gz', 'r') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if config['sample_indices'] == 1:
        train_set, valid_set = get_train_val(config, train_set[0])
    else:
        train_set, valid_set = load_indices_file(config, train_set[0])

    return train_set, valid_set, np.array(test_set[0], dtype=np.float32)


# load the cifar 100 dataset
def load_cifar(config):
    with open('cifar-100-python/train', 'rb') as f:
        train_set = pickle.load(f, encoding="latin1")

    with open('cifar-100-python/test', 'rb') as f:
        test_set = pickle.load(f, encoding="latin1")
    
    if config['sample_indices'] == 1:
        train_set, valid_set = get_train_val(config, train_set['data'])
    else:
        train_set, valid_set = load_indices_file(config, train_set['data'])
        
    train_set, valid_set, test_set = preproc_data(train_set, valid_set, np.array(test_set['data'], dtype=np.float32))
    
    return train_set, valid_set, test_set

def load_indices_file(config, train_set):
    # save the indices for reproducibility, as samples can differ depending on hardware
    with open(f"{config['dset_name']}_indices.txt", 'r') as f:
        indices = np.loadtxt(f, dtype=int)

    train_data = np.array(train_set, dtype=np.float32)[indices[:int(config['n_samples']/2)]]
    valid_data = np.array(train_set, dtype=np.float32)[indices[int(config['n_samples']/2):]]

    # check that the training and validation sets are the same
    assert len(train_data) == 10000 and len(valid_data) == 10000

    return train_data, valid_data


# split the training data into training and validation, by randomly sampling from the original training set
def get_train_val(config, train_set):
    # fix the random seed
    rand_seed = config['rand_seed']

    # set the python and numpy seeds
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    # get the number of examples in the original training dataset
    n_examples = len(train_set)

    # sample the training and validation indices
    indices = random.sample(range(0,n_examples), config['n_samples'])

    # save the indices for reproducibility, as samples can differ depending on hardware
    with open(f"{config['dset_name']}_indices.txt", 'w') as f:
        np.savetxt(f, indices)

    train_data = np.array(train_set, dtype=np.float32)[indices[:int(config['n_samples']/2)]]
    valid_data = np.array(train_set, dtype=np.float32)[indices[int(config['n_samples']/2):]]

    # check that the training and validation sets are the same
    assert len(train_data) == int(config['n_samples']/2) and len(valid_data) == int(config['n_samples']/2)

    return train_data, valid_data

# preprocess the data, by scaling the pixel values between 0 and 1
def preproc_data(train_set, valid_set, test_set):
    # get the maximum pixel value across the training, testing, and validation data
    max_pixel = max(np.max(train_set), np.max(valid_set), np.max(test_set))

    # scale the pixel values by the maximum value in the dataset
    train_set = np.true_divide(train_set, max_pixel)
    valid_set = np.true_divide(valid_set, max_pixel)
    test_set = np.true_divide(test_set, max_pixel)

    return train_set, valid_set, test_set