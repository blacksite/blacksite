from database import MongoDBConnect
import random
import time
import threading
import pymongo
from pymongo import MongoClient
import logging
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize

DB = MongoDBConnect()
DNN = None
LOCK = threading.Lock()


def train_dnn():
    print('Training DNN')


def test_dnn(dataset):
    print('Testing DNN')


def load_dnn(filename):
    print('Loading DNN')


def get_detector(_id):
    return DB.get_one(_id, DB.get_detectors_collection())


def evaluate_new_instances():
    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_new_instance_collection().watch(pipeline) as stream:
            for insert_change in stream:
                # print(insert_change)
                instance = insert_change['fullDocument']
                detector = get_detector(instance['detector'])

                if DNN:
                    # Evaluate instance['Value']
                    classification = None

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
        # The ChangeStream encountered an unrecoverable error or the
        # resume attempt failed to recreate the cursor.
        if resume_token is None:
            # There is no usable resume token because there was a
            # failure during ChangeStream initialization.
            logging.error('...' + str(error))
        else:
            # Use the interrupted ChangeStream's resume token to create
            # a new ChangeStream. The new stream will continue from the
            # last seen insert change without missing any events.
            with DB.get_collection().watch(
                    pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    print(insert_change)

                    instance = insert_change['fullDocument']
                    detector = get_detector(instance['detector'])

                    # Evaluate instance['Value']
                    LOCK.acquire()
                    classification = DNN
                    LOCK.release()

                    if classification > 0.5:
                        if detector['TYPE'] == 'IMMATURE':
                            detector['TYPE'] == 'MATURE'
                            DB.update_detector(detector)

                        DB.add_suspicious_instance(instance)

                    DB.remove_new_instance(instance)


class EvaluateNewInstance (threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        evaluate_new_instances()


class TrainDNN (threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        temp = train_dnn()
        LOCK.acquire()
        DNN = temp
        LOCK.release()


if __name__ == "__main__":
    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == 0:
            load_dnn('Test')
            break
        elif option == 1:
            train_dnn()
            break
        else:
            print('Invalid input')
            option = input(
                '0: Load Deep Neural Network\n'
                '1: Train New Deep Neural Network\n'
            )

    evaluation_thread = EvaluateThread(1, "EvaluationThread")
    evaluation_thread.start()

    update_thread = UpdateThread(2, "UpdateThread")
    update_thread.start()

    print("Still running")

    evaluation_thread.join()
    update_thread.join()


