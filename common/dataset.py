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
        self.instances_x = {}
        self.instances_y = {}
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
        # convert integers to dummy variables (i.e. one hot encoded)
        y_encoded = np_utils.to_categorical(y_encoded)

        self.number_of_classes = len(y_encoded[0])

        for i in range(len(y)):
            if y[i] not in self.instances_x:
                self.instances_x[y[i]] = []
                self.instances_y[y[i]] = []
            self.instances_x[y[i]].append(x_normalized[i])

            if y[i] == 'Benign':
                self.instances_y[y[i]].append(0)
            else:
                self.instances_y[y[i]].append(1)

        self.generate_min_max_list()

        # Add random benign sample to each malicious instance data set
        for key, value in self.instances_x.items():
            if key != 'Benign':
                current_length = len(value)
                for i in range(current_length):
                    index = random.randrange(len(self.instances_x['Benign']) - 1)

                    benign_sample = self.instances_x['Benign'].pop(index)
                    label = self.instances_y['Benign'].pop(index)
                    self.instances_x[key].append(benign_sample)
                    self.instances_y[key].append(label)

        # Delete the Benign lists in the instances_x and instances_y dictionaries
        del self.instances_x['Benign']
        del self.instances_y['Benign']

    def add_benign_samples_from_file(self, x, y):
        filename = '../data/benign.csv'

        data_frame = pandas.read_csv(filename, engine='python')
        data_frame.fillna(0)

        data_set = data_frame.values

        for d in data_set.data_set:
            x.append(np.array(self.replace_nan_inf(d[7:-1])))
            y.append('Benign')

    def generate_min_max_list(self):

        num_mal = 0
        for key, value in self.instances_x.items():
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
