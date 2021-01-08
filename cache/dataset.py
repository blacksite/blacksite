from os import path
import pandas
from sklearn.preprocessing import normalize, MinMaxScaler
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
MAX_FEATURES = []
CLASSES = []
std_devs = {}


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
    print("Loading data set started")
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

            if 'Day4' in f:
                del dataframe['Flow ID']
                del dataframe['Src IP']
                del dataframe['Src Port']
                del dataframe['Dst IP']
            LOCK.acquire()
            DATASET.extend(dataframe.values)
            LOCK.release()

    print('Loading data set finished')
    partition_data_set(w)


def partition_data_set(w):
    #   1. convert all string labels to int
    #   2. for each label, create a new dictionary key
    #   3. add all individual instances to dictionary
    #   4. for each label, partition instances into folds
    #   5. convert all malciious instances to labels of 1
    #   6. extend the folds dictionary with individual partitions

    print('Data set partitioning started')

    global NUM_MALICIOUS_INSTANCES
    global NUM_BENIGN_INSTANCES
    global RAW_PARTITION_SIZES
    global DATASET
    global KFOLDS
    global MAX_FEATURES
    global NUM_CLASSES
    global CLASSES
    global size_of_partitions

    x, y = [], []

    for d in DATASET:
        x.append(np.array(d[3:-1]))
        y.append(d[-1])
        if "Flow Duration" in d[3:-1]:
            print(str(d[3:-1]))

    # Convert X and Y into numpy arrays
    x = np.array(x, dtype='f8')
    y = np.array(y)

    # Normalize X features
    scalar = MinMaxScaler()
    scalar.fit(x)
    x_normalized = scalar.transform(x)

    calculate_total_mean_mad(x_normalized)

    # Transform y vector into a matrix
    encoder = LabelEncoder()
    encoder.fit(y)
    CLASSES = encoder.classes_
    y_encoded = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_encoded = np_utils.to_categorical(y_encoded)

    NUM_CLASSES = len(y_encoded[0])

    instances_x = {}
    instances_y = {}

    for i in range(len(y)):
        if y[i] not in instances_x:
            instances_x[y[i]] = []
            instances_y[y[i]] = []
        instances_x[y[i]].append(x_normalized[i])
        instances_y[y[i]].append(y_encoded[i])

    # Get the number of malicious and benign instances
    # Set the number of benign instances to malicious * 2
    num_mal = calculate_mean_stdev(instances_x)

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

    partitions_x = {}
    partitions_y = {}

    # Iterate through all folds
    for i in range(KFOLDS):

        # Check if the current index is in our partitions
        # If not, create a partition with the key i
        if i not in partitions_x:
            partitions_x[i] = []
            partitions_y[i] = []

        # for all key,value pairs in our instances_x dictionary
        for key, ins in instances_x.items():

            # For x in the range of the partition size of this type of sample
            for x in range(partition_sizes[key]):

                # Get a random index to pop from the sample list
                index = random.randrange(len(ins))

                # Pop the randomly selected sample and append to our partitions array
                partitions_x[i].append(ins.pop(index))
                partitions_y[i].append(instances_y[key].pop(index))

    global PARTITION_X
    global PARTITION_Y

    PARTITION_X = partitions_x
    PARTITION_Y = partitions_y

    size_of_partitions = len(partitions_x[0])

    w.write("Number of Benign Instances: " + str(NUM_BENIGN_INSTANCES) + "\n")
    w.write("Number of Malicious Instances: " + str(NUM_MALICIOUS_INSTANCES) + "\n")
    w.write("Number of folds: {:s}\n".format(str(KFOLDS)))
    for key, value in RAW_PARTITION_SIZES.items():
        w.write("Number of {:s} Instances per fold {:s}\n".format(key, str(value)))
    w.flush()
    w.close()

    print('Dataset partitioning finished')


# def calculate_mean_stdev(instances_x):
#     # global MAX_FEATURES
#     global std_devs
#
#     num_mal = 0
#     for key, value in instances_x.items():
#         if key != 'Benign':
#             num_mal = num_mal + len(value)
#
#             MAX_FEATURES[key] = []
#
#             for idx in zip(*value):
#                 mean = sum(idx) / len(idx)
#                 median = statistics.median(idx)
#                 std = statistics.stdev(idx)
#                 mad = stats.median_absolute_deviation(idx)
#
#                 max_val = max(idx)
#                 min_val = min(idx)
#
#                 MAX_FEATURES[key].append((median, mad, std))
#
#         # MAX_FEATURES[key] = [max(idx) for idx in zip(*value)]
#     return num_mal


def calculate_mean_stdev(instances_x):
    global MAX_FEATURES
    global std_devs

    MAX_FEATURES = [(0, 0)] * len(instances_x['Benign'][0])

    num_mal = 0
    for key, value in instances_x.items():
        if key != 'Benign':
            num_mal = num_mal + len(value)

            i = 0
            for idx in zip(*value):
                mean = sum(idx) / len(idx)
                median = statistics.median(idx)
                std = statistics.stdev(idx)
                mad = stats.median_absolute_deviation(idx)

                max_val = max(idx)
                min_val = min(idx)

                MAX_FEATURES[i] = (min(MAX_FEATURES[i][1], min_val), max(MAX_FEATURES[i][0], max_val))

                i += 1

        # MAX_FEATURES[key] = [max(idx) for idx in zip(*value)]
    return num_mal


def calculate_total_mean_mad(x):
    global MAX_FEATURES
    global std_devs

    for idx in zip(*x):
        mean = sum(idx) / len(idx)
        median = statistics.median(idx)
        std = statistics.stdev(idx)
        mad = stats.median_absolute_deviation(idx)

        max_val = max(idx)
        min_val = min(idx)

        MAX_FEATURES.append((median, mad))


def get_max_features():
    global MAX_FEATURES
    return MAX_FEATURES


def get_partitions():
    global PARTITION_X
    global PARTITION_Y
    return PARTITION_X, PARTITION_Y


def get_number_of_features():
    global NUM_FEATURES
    return NUM_FEATURES
