from database import MongoDBConnect
import random
import time
import threading
import pymongo
import logging

MAX_DETECTORS = 1000
DETECTOR_LENGTH = 32
INITIAL_DETECTOR_LIFESPAN = 300
IMMATURE_DETECTOR_LIFESPAN = 604800
MATURE_DETECTOR_LIFESPAN = 2592000
MEMORY_DETECTOR_LIFESPAN = 31536000
CURRENT_DETECTORS = ""

DB = MongoDBConnect()


def generate_detector():
    # Variable to store the
    # string
    value = ""

    # Loop to find the string
    # of desired length
    for i in range(DETECTOR_LENGTH):
        # randint function to generate
        # 0, 1 randomly and converting
        # the result into str
        temp = str(random.randint(0, 1))

        # Concatenatin the random 0, 1
        # to the final result
        value += temp

    detector = {"VALUE": value, "TYPE": "INITIAL", 'LIFE': time.time()}

    return detector


def remove_detector(detector):
    # print(detector['TYPE'] + ' detector removed')
    DB.delete_detector(detector)
    del CURRENT_DETECTORS[detector['_id']]
    add_detector()


def add_detector():
    temp_detector = generate_detector()
    d = DB.add_detector(temp_detector)
    temp_detector['_id'] = d.inserted_id
    CURRENT_DETECTORS[d.inserted_id] = temp_detector
    # print('Detector added')


def generate_initial_detectors():
    num_detectors = len(CURRENT_DETECTORS)
    while (num_detectors < MAX_DETECTORS):
        add_detector()
        num_detectors = len(CURRENT_DETECTORS)
        print('{:.2f}%'.format(float(num_detectors)/MAX_DETECTORS*100))


def evaluate_detector_lifespans():
    print('Detector lifespan evaluation started')
    while True:
        num_detectors = len(CURRENT_DETECTORS)
        while (num_detectors < MAX_DETECTORS):
            add_detector()
            num_detectors = len(CURRENT_DETECTORS)

        keys = []

        for k in CURRENT_DETECTORS.keys():
            keys.append(k)

        sum = 0
        for key in keys:
            value = CURRENT_DETECTORS[key]
            lifetime = time.time() - value['LIFE']
            if value['TYPE'] == 'INITIAL':
                if lifetime > INITIAL_DETECTOR_LIFESPAN:
                    remove_detector(value)
                    sum += 1
            elif value['TYPE'] == 'IMMATURE':
                if lifetime > IMMATURE_DETECTOR_LIFESPAN:
                    remove_detector(value)
                    sum += 1
            elif value['TYPE'] == 'MATURE':
                if lifetime > MATURE_DETECTOR_LIFESPAN:
                    remove_detector(value)
                    sum += 1
            elif value['TYPE'] == 'MEMORY':
                if lifetime > MEMORY_DETECTOR_LIFESPAN:
                    remove_detector(value)
                    sum += 1

        if sum > 0:
            print('{:d} detectors were replaced'.format(sum))


def update_detectors_callback():
    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'update'}}]
        with DB.get_collection().watch(pipeline=pipeline, full_document='updateLookup') as stream:
            for update_change in stream:
                # print(insert_change)
                detector = update_change['fullDocument']
                CURRENT_DETECTORS[detector['_id']] = detector
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


def delete_detectors_callback():



def evaluate_initial_new_instances():
    print('Evaluating resident new instances')


def evaluate_new_instances_callback():
    print('Evaluation new instances')





class EvaluateDetectorLifespanThread (threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        evaluate_detector_lifespans()


class UpdateThread (threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        update_detectors_callback()


if __name__ == "__main__":
    CURRENT_DETECTORS = DB.get_all()

    print('Generating initial detectors')
    generate_initial_detectors()
    print('Finished generating initial detectors')

    evaluation_thread = EvaluateDetectorLifespanThread(1, "EvaluateDetectorLifespanThread")
    evaluation_thread.start()

    update_thread = UpdateThread(2, "UpdateThread")
    update_thread.start()

    print("Still running")

    evaluation_thread.join()
    update_thread.join()


