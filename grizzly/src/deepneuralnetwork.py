import theano
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from database import MongoDBConnect
import threading
import pymongo
import logging
from instance import Instance
from detector import Detector

DB = MongoDBConnect()
DNN = None
DNN_LOCK = threading.Lock()
DB_LOCK = threading.Lock()
DNN_TRAINING_THRESHOLD = 30
FILENAME = 'deepneuralnetwork.blk'
ACCURACY_THRESHOLD = 0.8


def train_dnn(filename=None):
    print('Training DNN')

    dataset = None

    if filename:
        # load the dataset
        dataset = loadtxt(filename, delimiter=',')
    else:
        dataset = DB.get_all_dataset()

    # split into input (X) and output (y) variables
    X = dataset[:, 0:1]
    y = dataset[:, 1]

    # define the keras model
    temp_dnn = Sequential()
    temp_dnn.add(Dense(12, input_dim=80, activation='relu'))
    temp_dnn.add(Dense(8, activation='relu'))
    temp_dnn.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    temp_dnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # fit the keras model on the dataset
    temp_dnn.fit(X, y, epochs=150, batch_size=10)

    # evaluate the keras model
    accuracy = temp_dnn.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

    return temp_dnn


def save_dnn(filename):
    # serialize model to JSON
    model_json = DNN.to_json()
    with open(filename + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    filename.save_weights(filename + ".h5")
    print("Saved model to disk")


def load_dnn(filename):
    # load json and create model
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    DNN = model_from_json(loaded_model_json)
    # load weights into new model
    DNN.load_weights(filename + ".h5")
    print("Loaded model from disk")


def train_initial_detectors():
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