import pandas
from keras.models import Sequential
from keras.layers import Dense
import keras.metrics as km
from sklearn.preprocessing import normalize
from common.database import MongoDBConnect
import threading
import pymongo
import logging
from common.instance import Instance
from common.detector import Detector
import numpy as np
from tensorflow import keras
from os import path
import os

# Disable GPU otimization
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Class variables
DB = MongoDBConnect()
DNN = None
DNN_LOCK = threading.Lock()
DB_LOCK = threading.Lock()
DNN_TRAINING_THRESHOLD = 30
FILENAME = 'deepneuralnetwork.blk'
ACCURACY_THRESHOLD = 0.8
NUM_BENIGN_INSTANCES = 766570
NUM_MALICIOUS_INSTANCES = 0
BATCH_SIZE = 80
KFOLDS = 10
RAW_PARTITION_SIZES = {}


def define_model():
    # create and fit the DNN network
    model = Sequential()
    model.add(Dense(70, input_dim=76, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=["accuracy", km.TruePositives(), km.FalsePositives(), km.TrueNegatives(),
                           km.FalseNegatives()])
    return model


def partition_dataset(dataset):
    #   1. convert all string labels to int
    #   2. for each label, create a new dictionary key
    #   3. add all individual instances to dictionary
    #   4. for each label, partition instances into folds
    #   5. convert all malciious instances to labels of 1
    #   6. extend the folds dictionary with individual partitions

    global NUM_MALICIOUS_INSTANCES
    global NUM_BENIGN_INSTANCES
    global RAW_PARTITION_SIZES

    X, Y = [], []

    for d in dataset:
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

    NUM_MALICIOUS_INSTANCES = num_mal
    NUM_BENIGN_INSTANCES = NUM_MALICIOUS_INSTANCES * 2
    if NUM_BENIGN_INSTANCES > len(instances['Benign']):
        NUM_BENIGN_INSTANCES = len(instances['Benign'])

    #   Limit the number of benign instances
    instances['Benign'] = instances['Benign'][:NUM_BENIGN_INSTANCES]

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

    return partitions_X, partitions_Y


def train_dnn(filename=None):
    global NUM_BENIGN_INSTANCES
    global NUM_MALICIOUS_INSTANCES
    global RAW_PARTITION_SIZES

    print('Training DNN')

    save_file = input("Please enter the file name to save the results\n")

    while True:
        if path.exists("results/" + save_file + ".csv"):
            option = input("That file exists, would you like to overwrite it: y/n\n")
            if option == "y":
                break
            else:
                save_file = input("Please enter the file name to save the results\n")
        else:
            break

    w = open("results/" + save_file + ".csv", "w")

    print("Loading dataset")
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
            dataset.extend(dataframe.values)
    else:
        dataset = DB.get_all_dataset()

    print("Loading complete")
    print("Partitioning dataset")
    partitions_X, partitions_Y = partition_dataset(dataset)
    print("Partitioning complete")
    print("Starting training")

    w.write("Number of Benign Instances: " + str(NUM_BENIGN_INSTANCES) + "\n")
    w.write("Number of Malicious Instances: " + str(NUM_MALICIOUS_INSTANCES) + "\n")
    w.write("Number of folds: {:s}".format(str(KFOLDS)))
    for key, value in RAW_PARTITION_SIZES.items():
        w.write("Number of {:s} Instances: {:s} per fold\n".format(key, str(value)))
    w.write("\n")
    w.write("{:^10s},{:^10s},{:^10s},{:^10s},{:^10s},{:^10s}\n".format("BCE", "Accuracy", "TP", "FP", "TN", "FN"))
    w.flush()

    # batches = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # for b in batches:
    #     estimator = KerasClassifier(build_fn=define_model, epochs=100, batch_size=b, verbose=2)
    #     kfold = KFold(n_splits=10, shuffle=True)
    #     results = cross_val_score(estimator, np.array(training_instances), np.array(training_labels), cv=kfold)
    #     print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #
    #     w.write("{:s}".format(str(b)))
    #     for a in results:
    #         w.write(',{:s}'.format(str(a)))
    #     w.write('\n')
    #     w.flush()

    for i in range(KFOLDS):
        # create training and testing x and y datasets from the kFolds
        training_x, training_y = [], []
        test_x, test_y = partitions_X[i], partitions_Y[i]

        for x in range(KFOLDS):
            if x == i:
                continue

            training_x.extend(partitions_X[i])
            training_y.extend(partitions_Y[i])

        # begin training the model
        model = define_model()
        model.fit(np.array(training_x), np.array(training_y), BATCH_SIZE, epochs=100, verbose=2)
        results = model.evaluate(np.array(test_x), np.array(test_y))
        w.write('{:^10.2f}'.format(results[0]))
        w.write(',{:^10.2f}'.format(results[1] * 100.0))
        for x in range(2, len(results)):
            w.write(',{:^10.0f}'.format(results[x]))
        w.write('\n')
        w.flush()

    w.flush()
    w.close()

    global DNN
    DNN = model


def save_dnn(filename):
    global DNN
    DNN.save("model/" + filename + ".dnn");


def load_dnn(filename):
    global DNN

    while True:
        if not path.exists(filename):
            filename = input("The entered file does not exist. Please re-enter a file name\n")
        else:
            break

    DNN = keras.models.load_model(filename)


def train_initial_detectors():
    global DNN
    global DB_LOCK
    global DB
    global DNN_LOCK

    detectors = DB.get_all_detectors()

    for key, detector in detectors.items():
        if detector.get_type() == 'INITIAL':
            if DNN:
                DNN_LOCK.acquire()
                classification = DNN.predict(detector.get_value())
                DNN_LOCK.release()

                if classification < 0.5:
                    DB_LOCK.acquire()
                    DB.remove_detector(detector)
                    DB_LOCK.release()
                else:
                    detector.set_type('IMMATURE')
                    DB_LOCK.acquire()
                    DB.update_detector(detector)
                    DB_LOCK.release()
            else:
                print("No DNN available")
                exit(-1)


def retrain_detectors_callback():
    global DNN
    global DB_LOCK
    global DB
    global DNN_LOCK

    # When a detector is added to the detector set table
    # Classify the detector
    # If classified bening, remove detector

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_detectors_collection().watch(pipeline) as stream:
            for value in stream:
                detector = Detector(value['_id'], value['VALUE'], value['TYPE'], value['LIFE'])
                if detector.get_type() == 'INITIAL':
                    if DNN:
                        DNN_LOCK.acquire()
                        classification = DNN.predict(detector.get_value())
                        DNN_LOCK.release()

                        if classification < 0.5:
                            DB_LOCK.acquire()
                            DB.remove_detector(detector)
                            DB_LOCK.release()
                        else:
                            detector.set_type('IMMATURE')
                            DB_LOCK.acquire()
                            DB.update_detector_type(detector)
                            DB_LOCK.release()
                    else:
                        print("No DNN available")
                        exit(-1)

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_detectors_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for value in stream:
                    detector = Detector(value['_id'], value['VALUE'], value['TYPE'], value['LIFE'])
                    if detector.get_type() == 'INITIAL':
                        if DNN:
                            DNN_LOCK.acquire()
                            classification = DNN.predict(detector.get_value())
                            DNN_LOCK.release()

                            if classification < 0.5:
                                DB_LOCK.acquire()
                                DB.remove_detector(detector)
                                DB_LOCK.release()
                            else:
                                detector.set_type('IMMATURE')
                                DB_LOCK.acquire()
                                DB.update_detector_type(detector)
                                DB_LOCK.release()
                        else:
                            print("No DNN available")
                            exit(-1)


def classify_instance(suspicious_instance):
    global DNN
    global DB_LOCK
    global DB
    global DNN_LOCK

    if DNN:
        detector = DB.get_single_detector(suspicious_instance.get_detector_id())

        if DNN:
            DNN_LOCK.acquire()
            classification = DNN.predict(suspicious_instance.get_value())
            DNN_LOCK.release()

            if classification > 0.5:
                if detector.get_type() == 'IMMATURE':
                    detector.set_type('MATURE')
                    DB_LOCK.acquire()
                    DB.update_detector_type(detector)
                    DB_LOCK.release()

                suspicious_instance.set_type('CONFIRMATION_NEEDED')
                DB_LOCK.acquire()
                DB.add_confirmation_instance(suspicious_instance)
                DB_LOCK.release()

            DB_LOCK.acquire()
            DB.remove_suspicious_instance(suspicious_instance)
            DB_LOCK.release()
        else:
            print("No DNN available")
            exit(-1)
    else:
        print("No DNN available")
        exit(-1)


def evaluate_initial_suspicious_instances():
    # Evaluate suspicious instances in the database at startup

    suspicious_instances = DB.get_all_suspicious_instances()

    for key, suspicious_instance in suspicious_instances.items():
        classify_instance(suspicious_instance)


def evaluate_suspicious_instances_callback():
    global DNN
    global DB_LOCK
    global DB
    global DNN_LOCK

    # Monitor the suspicious instances table
    # When a new suspicious instance is added
    # Classify the instance
    # If the instance is classified malicious, forward to confirmation_instances table

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_suspicious_instance_collection().watch(pipeline) as stream:
            for insert_change in stream:
                full_doc = insert_change['fullDocument']
                instance = Instance(full_doc['_id'], full_doc['VALUE'], full_doc['TYPE'], full_doc['DETECTOR_id'])
                for key, value in full_doc:
                    if key != '_id' and key != 'VALUE' and key != 'TYPE' and key != 'DETECTOR_id':
                        instance.add_feature(key, value)

                classify_instance(instance)

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_suspicious_instance_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    full_doc = insert_change['fullDocument']
                    instance = Instance(full_doc['_id'], full_doc['VALUE'], full_doc['TYPE'], full_doc['DETECTOR_id'])
                    for key, value in full_doc:
                        if key != '_id' and key != 'VALUE' and key != 'TYPE' and key != 'DETECTOR_id':
                            instance.add_feature(key, value)

                    classify_instance(instance)


def retrain_dnn_callback():
    global DNN
    global DB_LOCK
    global DB
    global DNN_LOCK

    # Retrain the DNN when a new instance is added to the dataset
    # By the human-in-the-loop

    global DNN
    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        sum = 0
        with DB.get_dataset_collection().watch(pipeline) as stream:
            for insert_change in stream:
                sum += 1
                if sum >= DNN_TRAINING_THRESHOLD:
                    temp = train_dnn()
                    DNN_LOCK.acquire()
                    DNN = temp
                    DNN_LOCK.release()

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_dataset_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    sum += 1
                    if sum >= DNN_TRAINING_THRESHOLD:
                        temp = train_dnn()
                        DNN_LOCK.acquire()
                        DNN = temp
                        DNN_LOCK.release()