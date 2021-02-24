from os import path
import pandas
from sklearn.preprocessing import normalize
import numpy as np
import threading
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random
import statistics
from scipy import stats

LOCK = threading.Lock()
DATASET = []
KFOLDS = 10
NUM_BENIGN_INSTANCES = 766570
NUM_MALICIOUS_INSTANCES = 0
RAW_PARTITION_SIZES = {}
size_of_partitions = 0
PARTITION_X = []
PARTITION_Y = []
NUM_FEATURES = 76
NUM_CLASSES = 0
MAX_FEATURES = {}
CLASSES = []
split_index = 0


def add_sample(sample):
    global DATASET
    global LOCK

    LOCK.acquire()
    DATASET.append(sample)
    LOCK.release()


def get_dataset():
    global DATASET

    return DATASET


def size():
    global DATASET

    return len(DATASET)


def read_from_file(w):
    print("Loading dataset started")
    global DATASET
    global LOCK
    global split_index

    files = ['../data/KDDTrain+.txt', '../data/KDDTest+.txt']

    for f in files:
        while True:
            if not path.exists(f):
                f = input(f + " does not exist. Please re-enter another file name:\n")
            else:
                break
        dataframe = pandas.read_csv(f, engine='python')
        dataframe.fillna(0)

        dataframe.iloc[1] = dataframe.iloc[1].astype('category')
        dataframe.iloc[1] = dataframe.iloc[1].cat.codes

        dataframe.iloc[4] = dataframe.iloc[2].astype('category')
        dataframe.iloc[2] = dataframe.iloc[2].cat.codes

        dataframe.iloc[3] = dataframe.iloc[3].astype('category')
        dataframe.iloc[3] = dataframe.iloc[3].cat.codes

        DATASET.extend(dataframe.values)

        if split_index == 0:
            split_index = len(DATASET)

    print('Loading dataset finished')
    partition_data_set(w)


def partition_data_set(w):
    #   1. convert all string labels to int
    #   2. for each label, create a new dictionary key
    #   3. add all individual instances to dictionary
    #   4. for each label, partition instances into folds
    #   5. convert all malciious instances to labels of 1
    #   6. extend the folds dictionary with individual partitions

    print('Dataset partitioning started')

    global NUM_MALICIOUS_INSTANCES
    global NUM_BENIGN_INSTANCES
    global RAW_PARTITION_SIZES
    global DATASET
    global KFOLDS
    global MAX_FEATURES
    global NUM_CLASSES
    global CLASSES
    global size_of_partitions

    X, Y = [], []

    for d in DATASET:
        X.append(np.array(d[-2]))
        Y.append(d[-2])

    X = np.array(X)
    Y = np.array(Y)

    # Normalize the x vector
    X_minmax = normalize(X, norm="max")

    # Transform y vector into a matrix
    encoder = LabelEncoder()
    encoder.fit(Y)
    CLASSES = encoder.classes_
    Y_encoded = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    transformed_y = np_utils.to_categorical(Y_encoded)

    NUM_CLASSES = len(transformed_y[0])

    instances_x = {}
    instances_y = {}

    for i in range(len(Y)):
        if Y[i] not in instances_x:
            instances_x[Y[i]] = []
            instances_y[Y[i]] = []
        instances_x[Y[i]].append(X_minmax[i])
        instances_y[Y[i]].append(transformed_y[i])

    # Get the number of malicious and benign instances
    # Set the number of benign instances to malicious * 2
    num_mal = calculate_mean_stdev(instances_x)

    NUM_MALICIOUS_INSTANCES = num_mal
    NUM_BENIGN_INSTANCES = num_mal
    if NUM_BENIGN_INSTANCES > len(instances_x['normal']):
        NUM_BENIGN_INSTANCES = len(instances_x['normal'])

    partitions_X = {}
    partitions_Y = {}

    partitions_X[0] = X_minmax[:split_index]
    partitions_Y[0] = transformed_y[:split_index]
    partitions_X[1] = X_minmax[split_index:]
    partitions_X[1] = transformed_y[split_index:]

    global PARTITION_X
    global PARTITION_Y

    PARTITION_X = partitions_X
    PARTITION_Y = partitions_Y

    size_of_partitions = len(partitions_X[0])

    w.write("Number of normal Instances: " + str(NUM_BENIGN_INSTANCES) + "\n")
    w.write("Number of Malicious Instances: " + str(NUM_MALICIOUS_INSTANCES) + "\n")
    w.flush()
    w.close()

    print('Dataset partitioning finished')


def calculate_mean_stdev(instances_x):
    global MAX_FEATURES

    num_mal = 0
    for key, value in instances_x.items():
        if key != 'normal':
            num_mal = num_mal + len(value)

            MAX_FEATURES[key] = []

            for idx in zip(*value):
                mean = sum(idx) / len(idx)
                median = statistics.median(idx)
                std = statistics.stdev(idx)
                mad = stats.median_absolute_deviation(idx)

                max_val = max(idx)
                min_val = min(idx)

                MAX_FEATURES[key].append((min_val, max_val))

        # MAX_FEATURES[key] = [max(idx) for idx in zip(*value)]
    return num_mal


def get_partitions():
    global PARTITION_X
    global PARTITION_Y

    return PARTITION_X, PARTITION_Y


def get_number_of_features():
    global NUM_FEATURES

    return NUM_FEATURES