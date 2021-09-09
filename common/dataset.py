from os import path
import pandas
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import threading
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import random
import math


class DataSet:

    def __init__(self):
        self.lock = threading.Lock()
        self.data_set = []
        self.ais_instances_x = []
        self.ais_instances_y = []
        self.dnn_instances_x = {}
        self.dnn_instances_y = {}
        self.number_of_features = 76
        self.number_of_classes = 0
        self.mean_mad = []
        self.min_max = {}
        self.classes = []

    def read_from_file(self, w, filename):
        print("Loading data set started")

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
                self.data_set.extend(dataframe.values)

        print('Loading data set finished')
        self.partition(w)

    def partition(self, w):
        #   1. convert all string labels to int
        #   2. for each label, create a new dictionary key
        #   3. add all individual instances to dictionary
        #   4. for each label, partition instances into folds
        #   5. convert all malciious instances to labels of 1
        #   6. extend the folds dictionary with individual partitions

        print('Data set partitioning started')

        x, y = [], []

        for d in self.data_set:
            # if "LOIC-UDP".lower() not in d[-1].lower() and "Infiltration".lower() not in d[-1].lower() and "Bot".lower()\
            #         not in d[-1].lower() and 'Benign'.lower() not in d[-1].lower():
            x.append(np.array(self.replace_nan_inf(d[3:-1])))
            y.append(d[-1])
            if "Flow Duration" in d[3:-1]:
                print(str(d[3:-1]))

        # self.add_benign_samples_from_file(x, y)

        # Convert X and Y into numpy arrays
        x = np.array(x, dtype='f8')
        y = np.array(y)

        # Normalize X features
        scalar = MinMaxScaler()
        scalar.fit(x)
        x_normalized = scalar.transform(x)

        # Print the min vector and max vector to be read in by Panda
        w.write(str(scalar.data_min_[0]))
        for i in range(1, len(scalar.data_min_)):
            w.write(',' + str(scalar.data_min_[i]))
        w.write('\n')

        w.write(str(scalar.data_max_[0]))
        for i in range(1, len(scalar.data_max_)):
            w.write(',' + str(scalar.data_max_[i]))
        w.write('\n')

        w.flush()
        w.close()

        # Transform y vector into a matrix
        encoder = LabelEncoder()
        encoder.fit(y)
        self.classes = encoder.classes_
        y_encoded = encoder.transform(y)

        self.number_of_classes = len(y_encoded)

        local_instances_x = {}
        local_instances_y = {}

        for i in range(len(y)):
            if y[i] not in local_instances_x:
                local_instances_x[y[i]] = []
                local_instances_y[y[i]] = []
            local_instances_x[y[i]].append(x_normalized[i])
            local_instances_y[y[i]].append(y_encoded[i])

        self.generate_min_max_list(local_instances_x)

        # take only 10% of each type for the ais validation samples,
        # not to be less than 1000
        # not to exceed 10000 samples
        for key, samples in local_instances_x.items():
            size = len(samples)
            labels = local_instances_y[key]
            if size < 1000:
                self.ais_instances_x.extend(samples)
                self.ais_instances_y.extend(labels)
            else:
                cutoff = int(len(samples) * 0.1)
                if cutoff < 1000:
                    cutoff = 1000
                elif cutoff > 10000:
                    cutoff = 10000
                self.ais_instances_x.extend(samples[:cutoff])
                self.ais_instances_y.extend(labels[:cutoff])

        # Partition dataset for dnn training,
        # not to be less than 1000
        # not to exceed 10000 samples
        for key, samples in local_instances_x.items():
            if key != 'Benign':  # and key != 'FTP-BruteForce':
                if key not in self.dnn_instances_x:
                    self.dnn_instances_x[key] = []
                    self.dnn_instances_y[key] = []

                size = len(samples)
                if size < 1000:
                    cutoff = len(samples)
                else:
                    cutoff = int(len(samples) * 0.1)
                    if cutoff < 1000:
                        cutoff = 1000
                    elif cutoff > 10000:
                        cutoff = 10000

                # for the given malicious key, determine how many instances I need per fold
                for i in range(cutoff):
                    # Get a random index to pop from the sample list
                    index = random.randrange(len(samples))

                    # Add the random sample to the current partition
                    self.dnn_instances_x[key].append(samples.pop(index))
                    self.dnn_instances_y[key].append(1)

                    # Get a random index to pop from the sample list
                    index = random.randrange(len(local_instances_x['Benign']))

                    # Add the random sample to the current partition
                    self.dnn_instances_x[key].append(local_instances_x['Benign'].pop(index))
                    self.dnn_instances_y[key].append(0)

    def add_benign_samples_from_file(self, x, y):
        filename = '../data/benign.csv'

        data_frame = pandas.read_csv(filename, engine='python')
        data_frame.fillna(0)

        data_set = data_frame.values

        for d in data_set.data_set:
            x.append(np.array(self.replace_nan_inf(d[7:-1])))
            y.append('Benign')

    def generate_min_max_list(self, local_instances_x):

        num_mal = 0
        for key, value in local_instances_x.items():
            if key != 'Benign':
                temp = []
                num_mal = num_mal + len(value)

                i = 0
                for idx in zip(*value):
                    max_val = max(idx)
                    min_val = min(idx)

                    temp.append((min_val, max_val))

                    i += 1
                self.min_max[key] = temp

        return num_mal

    def get_number_of_features(self):
        return self.number_of_features

    def replace_nan_inf(self, x):
        for i in range(len(x)):
            val = float(x[i])

            if math.isinf(val) or math.isnan(val):
                x[i] = 0.0
                # print(val)
            else:
                x[i] = val
        return x

    def get_classes(self):
        return self.classes

    def get_min_max_features_by_type(self, key):
        return self.min_max[key]
