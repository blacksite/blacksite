from os import listdir
from os.path import isfile, join
import pandas
from sklearn.preprocessing import normalize, MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random
import statistics
from scipy import stats

DATA_SET = {}
RNN_DATA_SET = {}
RNN_FOLDS_X = {}
RNN_FOLDS_Y = {}
DEVICE_FOLDS_X = {}
DEVICE_FOLDS_Y = {}
NUM_FOLDS = 10
NUM_BENIGN_INSTANCES = 0
NUM_MALICIOUS_INSTANCES = 0
NUM_CLASSES = 0
NUM_FEATURES = 0
MIN_MAX_FEATURES = {}
CLASSES = []
INCLUSION_LIST = ['192.168.1.7', '10.0.0.5', '10.0.0.6', '10.0.0.7', '10.0.0.8', '10.0.0.10',
                  '10.0.0.11', '10.0.0.12', '10.0.0.13', '10.0.0.23', '10.0.0.14', '10.0.0.15', '10.0.0.16',
                  '10.0.0.17']
ENCODER = None


def add_sample(sample):
    global DATA_SET
    DATA_SET.append(sample)


def get_dataset():
    global DATA_SET
    return DATA_SET


def read_from_file(w, path):
    print('Loading data set started')

    files = [f for f in listdir(path) if isfile(join(path, f))]

    ds = []
    for f in files:
        print(f)
        file_path = path + '\\' + f
        df = pandas.read_csv(file_path, engine='python')
        df.fillna(0)
        ds.extend(df.values)

    print('Loading data set finished')
    partition_based_on_ip(ds)
    partition_based_on_mal_type(ds)
    create_folds()


def partition_based_on_mal_type(ds):
    global RNN_DATA_SET
    global NUM_FEATURES

    x = normalize_data_set(ds)
    y = convert_labels(ds)

    for i in range(len(ds)):
        sample = ds[i]
        label = sample[-1]

        if label not in RNN_DATA_SET:
            RNN_DATA_SET[label] = []

        values = list(x[i])

        # Add encoded label to normalized values
        encoded_label = list(y[i]).index(max(list(y[i])))
        values.append(encoded_label)

        RNN_DATA_SET[label].append(values)


def partition_based_on_ip(ds):
    global DATA_SET
    global NUM_FEATURES

    x = normalize_data_set(ds)
    y = convert_labels(ds)

    for i in range(len(ds)):
        sample = ds[i]

        # Get source and destination IP address
        src_ip = sample[0]
        dst_ip = sample[1]

        # Get unencoded label
        label = sample[-1]

        # Get normalized values
        values = list(x[i])
        NUM_FEATURES = len(values)

        # Add encoded label to normalized values
        encoded_label = list(y[i]).index(max(list(y[i])))
        values.append(encoded_label)
        # Duplicate the values for destination address data set
        dup_values = values.copy()

        if src_ip not in DATA_SET:
            DATA_SET[src_ip] = {}

        if label not in DATA_SET[src_ip]:
            DATA_SET[src_ip][label] = []

        if dst_ip not in DATA_SET:
            DATA_SET[dst_ip] = {}

        if label not in DATA_SET[dst_ip]:
            DATA_SET[dst_ip][label] = []

        DATA_SET[src_ip][label].append(values)
        DATA_SET[dst_ip][label].append(dup_values)


def normalize_data_set(ds):
    x = []

    for d in ds:
        x.append(np.array(d[5:-1]))

    # Convert X and Y into numpy arrays
    x = np.array(x, dtype='f8')

    # Normalize X features
    scalar = MinMaxScaler()
    scalar.fit(x)
    x_normalized = scalar.transform(x)

    return x_normalized


def convert_labels(ds):
    global NUM_CLASSES
    global CLASSES
    global ENCODER

    y = []

    for d in ds:
        y.append(d[-1])

    y = np.array(y)

    # Transform y vector into a matrix
    encoder = LabelEncoder()
    encoder.fit(y)
    CLASSES = encoder.classes_
    y_encoded = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_encoded = np_utils.to_categorical(y_encoded)

    NUM_CLASSES = len(y_encoded[0])
    ENCODER = encoder

    return y_encoded


def create_folds():
    global DATA_SET
    global RNN_DATA_SET
    global DEVICE_FOLDS_X
    global DEVICE_FOLDS_Y
    global RNN_FOLDS_X
    global RNN_FOLDS_Y
    global NUM_FOLDS
    global ENCODER

    remove_exclusion_from_data_set()
    check_min_data_set()
    calculate_label_min_max()

    for ip_addr, ip_dict in DATA_SET.items():
        DEVICE_FOLDS_Y[ip_addr] = {}
        DEVICE_FOLDS_X[ip_addr] = {}

        for label, samples in ip_dict.items():
            if label != 'Benign':
                total = len(samples)
                samples_per_fold = int(total / NUM_FOLDS)

                for fold_num in range(NUM_FOLDS):
                    if fold_num not in DEVICE_FOLDS_Y[ip_addr]:
                        DEVICE_FOLDS_Y[ip_addr][fold_num] = []
                        DEVICE_FOLDS_X[ip_addr][fold_num] = []

                    if fold_num not in RNN_FOLDS_Y:
                        RNN_FOLDS_Y[fold_num] = []
                        RNN_FOLDS_X[fold_num] = []

                    for x in range(samples_per_fold):
                        # Get a random index to pop from the sample list
                        index = random.randrange(len(samples))

                        # Create a duplicate of the sample for the RNN
                        s = samples.pop(index)
                        encoded_label = s[-1]
                        s = s[:-1]

                        benign_index = random.randrange(len(ip_dict['Benign']))
                        benign_s = ip_dict['Benign'][benign_index]
                        encoded_benign_label = benign_s[-1]
                        benign_s = benign_s[:-1]

                        # Add the sample and the duplicate to the corresponding fold dictionaries
                        DEVICE_FOLDS_X[ip_addr][fold_num].append(s)
                        DEVICE_FOLDS_Y[ip_addr][fold_num].append(encoded_label)
                        DEVICE_FOLDS_X[ip_addr][fold_num].append(benign_s)
                        DEVICE_FOLDS_Y[ip_addr][fold_num].append(encoded_benign_label)

    for label, samples in RNN_DATA_SET.items():
        if label != 'Benign':
            samples_per_fold = int(len(samples)/NUM_FOLDS)

            for fold_num in range(NUM_FOLDS):
                for x in range(samples_per_fold):
                    # Get a random index to pop from the sample list
                    index = random.randrange(len(samples))

                    # Create a duplicate of the sample for the RNN
                    s = samples.pop(index)
                    encoded_label = s[-1]
                    s = s[:-1]

                    benign_index = random.randrange(len(RNN_DATA_SET['Benign']))
                    benign_s = RNN_DATA_SET['Benign'][benign_index]
                    encoded_benign_label = benign_s[-1]
                    benign_s = benign_s[:-1]

                    RNN_FOLDS_X[fold_num].append(s)
                    RNN_FOLDS_Y[fold_num].append(encoded_label)
                    RNN_FOLDS_X[fold_num].append(benign_s)
                    RNN_FOLDS_Y[fold_num].append(encoded_benign_label)


def remove_exclusion_from_data_set():
    global DATA_SET
    global INCLUSION_LIST

    ds = {}
    for key, value in DATA_SET.items():
        if key in INCLUSION_LIST:
            ds[key] = value

    DATA_SET = ds


def check_min_data_set():
    for ip_addr, value in DATA_SET.items():
        for label, samp_list in value.items():
            if len(samp_list) < NUM_FOLDS:
                print ("Not enough samples for IP Address " + ip_addr + " " + label)


def calculate_label_min_max():
    global DATA_SET
    global MIN_MAX_FEATURES

    for ip_addr, value in DATA_SET.items():
        all_samps = []

        for label, samps in value.items():
            all_samps.extend(samps[:][0:-1])

        MIN_MAX_FEATURES[ip_addr] = []

        for idx in zip(*all_samps):
            median = statistics.median(idx)
            min_val = min(idx)
            max_val = max(idx)
            mad = stats.median_absolute_deviation(idx)

            MIN_MAX_FEATURES[ip_addr].append((min_val, max_val))


def get_min_max_features():
    global MIN_MAX_FEATURES
    return MIN_MAX_FEATURES


def get_min_max_features(device):
    global MIN_MAX_FEATURES
    return MIN_MAX_FEATURES[device]


def get_device_folds():
    global DEVICE_FOLDS_X
    global DEVICE_FOLDS_Y
    return DEVICE_FOLDS_X, DEVICE_FOLDS_Y


def get_rnn_folds():
    global RNN_FOLDS_X
    global RNN_FOLDS_Y
    return RNN_FOLDS_X, RNN_FOLDS_Y


def get_number_of_features():
    global NUM_FEATURES
    return NUM_FEATURES


def get_classes():
    global CLASSES
    return CLASSES


def get_number_of_classes():
    global NUM_CLASSES
    return NUM_CLASSES


def get_original_data_set():
    global DATA_SET
    return DATA_SET
