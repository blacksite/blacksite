from os import path
import pandas
from sklearn.preprocessing import normalize
import numpy as np
import threading
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random

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


def read_from_file(w, filename):
    print("Loading dataset started")
    global DATASET
    global LOCK

    if filename:
        # load the dataset
        dataset = []
        files = []
        if ',' in filename:
            files = filename.split(',')
        else:
            files.append(filename)

        for f in files:
            while True:
                if not path.exists(f):
                    f = input(f + " does not exist. Please re-enter another file name:\n")
                else:
                    break
            dataframe = pandas.read_csv(f, engine='python')
            dataframe.fillna(0)
            LOCK.acquire()
            DATASET.extend(dataframe.values)
            LOCK.release()

    print('Loading dataset finished')
    partition_dataset_multiclass(w)


def partition_dataset(w):
    #   1. convert all string labels to int
    #   2. for each label, create a new dictionary key
    #   3. add all individual instances to dictionary
    #   4. for each label, partition instances into folds
    #   5. convert all malciious instances to labels of 1
    #   6. extend the folds dictionary with individual partitions

    print('Datast partitioning started')
    global NUM_MALICIOUS_INSTANCES
    global NUM_BENIGN_INSTANCES
    global RAW_PARTITION_SIZES
    global DATASET
    global KFOLDS
    global MAX_FEATURES
    global size_of_partitions

    X, Y = [], []

    for d in DATASET:
        X.append(np.array(d[3:-1]))
        Y.append(d[-1])
        if "Flow Duration" in d[3:-1]:
            print(str(d[3:-1]))

    X = np.array(X)
    Y = np.array(Y)

    X_minmax = normalize(X, norm="max")

    instances = {}

    for i in range(len(Y)):
        if Y[i] not in instances:
            instances[Y[i]] = []
        instances[Y[i]].append(X_minmax[i])

    # Get the number of malicious and benign instances
    # Set the number of benign instances to malicious * 2
    num_mal = 0
    for key, value in instances.items():
        if key != 'Benign':
            num_mal = num_mal + len(value)

            MAX_FEATURES[key] = [max(idx) for idx in zip(*value)]


    NUM_MALICIOUS_INSTANCES = num_mal
    NUM_BENIGN_INSTANCES = num_mal
    if NUM_BENIGN_INSTANCES > len(instances['Benign']):
        NUM_BENIGN_INSTANCES = len(instances['Benign'])

    # Limit the number of benign instances
    # instances['Benign'] = instances['Benign'][:NUM_BENIGN_INSTANCES]

    # Randomly select Benign instances to match the number of malicious instances
    instances['Benign'] = random.sample(instances['Benign'], NUM_BENIGN_INSTANCES)

    partition_sizes = {}

    for key, value in instances.items():
        partition_sizes[key] = int(len(value)/KFOLDS)

    RAW_PARTITION_SIZES = partition_sizes

    partitions_X = {}
    partitions_Y = {}

    for i in range(KFOLDS):
        if i not in partitions_X:
            partitions_X[i] = []
            partitions_Y[i] = []
        for key, ins in instances.items():
            for x in range(partition_sizes[key]):
                partitions_X[i].append(ins.pop(0))
                if key == 'Benign':
                    partitions_Y[i].append(0)
                else:
                    partitions_Y[i].append(1)

    global PARTITION_X
    global PARTITION_Y

    PARTITION_X = partitions_X
    PARTITION_Y = partitions_Y

    size_of_partitions = len(partitions_X[0])

    w.write("Number of Benign Instances: " + str(NUM_BENIGN_INSTANCES) + "\n")
    w.write("Number of Malicious Instances: " + str(NUM_MALICIOUS_INSTANCES) + "\n")
    w.write("Number of folds: {:s}\n".format(str(KFOLDS)))
    for key, value in RAW_PARTITION_SIZES.items():
        w.write("Number of {:s} Instances per fold {:s}\n".format(key, str(value)))
    w.flush()
    w.close()

    print('Dataset partitioning finished')


def partition_dataset_multiclass(w):
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
        X.append(np.array(d[3:-1]))
        Y.append(d[-1])
        if "Flow Duration" in d[3:-1]:
            print(str(d[3:-1]))

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
    dummy_y = np_utils.to_categorical(Y_encoded)

    NUM_CLASSES = len(dummy_y[0])

    instances_x = {}
    instances_y = {}

    for i in range(len(Y)):
        if Y[i] not in instances_x:
            instances_x[Y[i]] = []
            instances_y[Y[i]] = []
        instances_x[Y[i]].append(X_minmax[i])
        instances_y[Y[i]].append(dummy_y[i])

    # Get the number of malicious and benign instances
    # Set the number of benign instances to malicious * 2
    num_mal = 0
    for key, value in instances_x.items():
        if key != 'Benign':
            num_mal = num_mal + len(value)

            MAX_FEATURES[key] = [max(idx) for idx in zip(*value)]

    NUM_MALICIOUS_INSTANCES = num_mal
    NUM_BENIGN_INSTANCES = num_mal
    if NUM_BENIGN_INSTANCES > len(instances_x['Benign']):
        NUM_BENIGN_INSTANCES = len(instances_x['Benign'])

    # Limit the number of benign instances
    # instances['Benign'] = instances['Benign'][:NUM_BENIGN_INSTANCES]

    # Randomly select Benign instances to match the number of malicious instances
    instances_x['Benign'] = random.sample(instances_x['Benign'], NUM_BENIGN_INSTANCES)

    partition_sizes = {}

    for key, value in instances_x.items():
        partition_sizes[key] = int(len(value)/KFOLDS)

    RAW_PARTITION_SIZES = partition_sizes

    partitions_X = {}
    partitions_Y = {}

    for i in range(KFOLDS):
        if i not in partitions_X:
            partitions_X[i] = []
            partitions_Y[i] = []
        for key, ins in instances_x.items():
            for x in range(partition_sizes[key]):
                partitions_X[i].append(ins.pop(0))
                partitions_Y[i].append(instances_y[key].pop(0))

    global PARTITION_X
    global PARTITION_Y

    PARTITION_X = partitions_X
    PARTITION_Y = partitions_Y

    size_of_partitions = len(partitions_X[0])

    w.write("Number of Benign Instances: " + str(NUM_BENIGN_INSTANCES) + "\n")
    w.write("Number of Malicious Instances: " + str(NUM_MALICIOUS_INSTANCES) + "\n")
    w.write("Number of folds: {:s}\n".format(str(KFOLDS)))
    for key, value in RAW_PARTITION_SIZES.items():
        w.write("Number of {:s} Instances per fold {:s}\n".format(key, str(value)))
    w.flush()
    w.close()

    print('Dataset partitioning finished')


def get_partitions():
    global PARTITION_X
    global PARTITION_Y

    return PARTITION_X, PARTITION_Y


def get_number_of_features():
    global NUM_FEATURES

    return NUM_FEATURES