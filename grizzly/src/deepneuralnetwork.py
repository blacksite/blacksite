from database import MongoDBConnect
import threading
import pymongo
import logging

DB = MongoDBConnect()
DNN = None
LOCK = threading.Lock()
DNN_TRAINING_THRESHOLD = 30
FILENAME = 'deepneuralnetwork.blk'
ACCURACY_THRESHOLD = 0.8


def train_dnn():
    print('Training DNN')

    accuracy = 0.0

    while accuracy < ACCURACY_THRESHOLD:
        temp = None

        accuracy = test_dnn(temp)

    return temp


def test_dnn(temp):
    print('Testing DNN')

    #calculate the accuracy of temp
    accuracy = None

    return accuracy


def save_dnn():
    print('Saving DNN')


def load_dnn():
    print('Loading DNN from ' + FILENAME)


def train_initial_detectors():
    detectors = DB.get_all_detectors()

    for key, detector in detectors.items():
        if detector['TYPE'] == 'INITIAL':
            if DNN:
                LOCK.acquire()
                # classification = DNN.classify( detector['Value'])
                classification = None
                LOCK.release()

                if classification < 0.5:
                    DB.remove_detector(detector)
                else:
                    detector['TYPE'] = 'IMMATURE'
                    DB.update_detector(detector)
            else:
                print("No DNN available")
                exit(-1)


def retrain_detectors_callback():
    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_detectors_collection().watch(pipeline) as stream:
            for detector in stream:
                if detector['TYPE'] == 'INITIAL':
                    if DNN:
                        LOCK.acquire()
                        # classification = DNN.classify( detector['Value'])
                        classification = None
                        LOCK.release()

                        if classification < 0.5:
                            DB.remove_detector(detector)
                        else:
                            detector['TYPE'] = 'IMMATURE'
                            DB.update_detector(detector)
                    else:
                        print("No DNN available")
                        exit(-1)

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for detector in stream:
                    if detector['TYPE'] == 'INITIAL':
                        if DNN:
                            LOCK.acquire()
                            # classification = DNN.classify( detector['Value'])
                            classification = None
                            LOCK.release()

                            if classification < 0.5:
                                DB.remove_detector(detector)
                            else:
                                detector['TYPE'] = 'IMMATURE'
                                DB.update_detector(detector)
                        else:
                            print("No DNN available")
                            exit(-1)


def evaluate_initial_suspicious_instances():
    suspicious_instances = DB.get_all_suspicious_instances()

    for key, suspicious_instance in suspicious_instances.items():
        if suspicious_instance['TYPE'] == 'INITIAL':
            if DNN:
                detector = DB.get_one(suspicious_instance['DETECTOR_id'], DB.get_detectors_collection())

                if DNN:
                    LOCK.acquire()
                    # classification = DNN.classify( suspicious_instance)
                    classification = None
                    LOCK.release()

                    if classification > 0.5:
                        if detector['TYPE'] == 'IMMATURE':
                            detector['TYPE'] == 'MATURE'
                            DB.update_detector(detector)

                        DB.add_suspicious_instance(suspicious_instance)

                    DB.remove_new_instance(suspicious_instance)
                else:
                    print("No DNN available")
                    exit(-1)
            else:
                print("No DNN available")
                exit(-1)


def evaluate_suspicious_instances_callback():
    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_new_instance_collection().watch(pipeline) as stream:
            for insert_change in stream:
                instance = insert_change['fullDocument']
                detector = DB.get_one(instance['DETECTOR_id'], DB.get_detectors_collection())

                if DNN:
                    LOCK.acquire()
                    # classification = DNN.classify( suspicious_instance)
                    classification = None
                    LOCK.release()

                    if classification > 0.5:
                        if detector['TYPE'] == 'IMMATURE':
                            detector['TYPE'] == 'MATURE'
                            DB.update_detector(detector)

                        DB.add_suspicious_instance(instance)

                    DB.remove_new_instance(instance)
                else:
                    print("No DNN available")
                    exit(-1)

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_new_instance_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    instance = insert_change['fullDocument']
                    detector = DB.get_one(instance['DETECTOR_id'], DB.get_detectors_collection())

                    if DNN:
                        LOCK.acquire()
                        # classification = DNN.classify( suspicious_instance)
                        classification = None
                        LOCK.release()

                        if classification > 0.5:
                            if detector['TYPE'] == 'IMMATURE':
                                detector['TYPE'] == 'MATURE'
                                DB.update_detector(detector)

                            DB.add_suspicious_instance(instance)

                        DB.remove_new_instance(instance)
                    else:
                        print("No DNN available")
                        exit(-1)


def retrain_dnn_callback():
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
                    LOCK.acquire()
                    DNN = temp
                    LOCK.release()

                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    sum += 1
                    if sum >= DNN_TRAINING_THRESHOLD:
                        temp = train_dnn()
                        LOCK.acquire()
                        DNN = temp
                        LOCK.release()


class EvaluateSuspiciousInstanceThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        evaluate_suspicious_instances_callback()


class EvaluateInitialSuspiciousInstanceThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        evaluate_initial_suspicious_instances()


class RetrainDNNThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        retrain_dnn_callback()


class RetrainDetectorsThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        retrain_detectors_callback()


class TrainInitialDetectorsThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        train_initial_detectors()
