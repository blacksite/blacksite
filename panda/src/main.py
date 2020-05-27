from database import MongoDBConnect
from detector import Detector
from instance import Instance
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import pymongo
import logging
import sys

R_VALUE = 16
MAX_DETECTORS = 1000
DETECTOR_LENGTH = 32
INITIAL_DETECTOR_LIFESPAN = 3600
IMMATURE_DETECTOR_LIFESPAN = 604800
MATURE_DETECTOR_LIFESPAN = 2592000
MEMORY_DETECTOR_LIFESPAN = 31536000
CURRENT_DETECTORS = ""
EXECUTOR = ThreadPoolExecutor(10)
FUTURES = []
LOCK = threading.Lock()

DB = MongoDBConnect()


def generate_detector():
    # Create and return a detector
    # Format {"Value": value, "TYPE": "INITIAL", "LIFE": current time}

    value = ""

    for i in range(DETECTOR_LENGTH):
        temp = str(random.randint(0, 1))
        value += temp

    detector = Detector(value=value, type="INITIAL", life=time.time())

    return detector


def remove_detector(detector, from_database=True):
    # Remove detector from database and persistent memory
    LOCK.acquire()
    if from_database:
        DB.delete_detector(detector)
    try:
        del CURRENT_DETECTORS[detector.get_id()]
    except TypeError as error:
        logging.error(error)
    LOCK.release()
    add_new_detector()


def add_new_detector():
    # Add detector to detectors table and persistent memory

    temp_detector = generate_detector()
    LOCK.acquire()
    inserted_detector = DB.add_detector(temp_detector)
    temp_detector.set_id(inserted_detector.inserted_id)
    CURRENT_DETECTORS[temp_detector.get_id()] = temp_detector
    LOCK.release()


def generate_initial_detectors():
    # On startup, generate detectors to meet the desired number of detectors
    num_detectors = len(CURRENT_DETECTORS)
    while num_detectors < MAX_DETECTORS:
        add_new_detector()
        num_detectors = len(CURRENT_DETECTORS)
        print('{:.2f}%'.format(float(num_detectors)/MAX_DETECTORS*100))


def evaluate_detector_lifespans():
    # Continuously iterate through detectors to determine if lifespan has elapsed
    # If the lifespan has elapsed, delete the detector and create a new one
    # Add the new detector to the detectors table and to persistent list
    try:
        print('Detector lifespan evaluation started')
        while True:
            keys = []

            for k in CURRENT_DETECTORS.keys():
                keys.append(k)

            sum = 0
            for key in keys:
                detector = CURRENT_DETECTORS[key]
                lifetime = time.time() - detector.get_life()
                if detector.get_type() == 'INITIAL':
                    if lifetime > INITIAL_DETECTOR_LIFESPAN:
                        remove_detector(detector)
                        sum += 1
                elif detector.get_type() == 'IMMATURE':
                    if lifetime > IMMATURE_DETECTOR_LIFESPAN:
                        remove_detector(detector)
                        sum += 1
                elif detector.get_type() == 'MATURE':
                    if lifetime > MATURE_DETECTOR_LIFESPAN:
                        remove_detector(detector)
                        sum += 1
                elif detector.get_type() == 'MEMORY':
                    if lifetime > MEMORY_DETECTOR_LIFESPAN:
                        remove_detector(detector)
                        sum += 1

            if sum > 0:
                print('{:d} detectors were replaced'.format(sum))
    except RuntimeError as error:
        logging.error(error)


def update_persistent_detectors():
    # When a detector is updated from initial to immature / immature to mature / mature to memory
    # Update the persistent table in memory of detectors

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'update'}}]
        with DB.get_detectors_collection().watch(pipeline=pipeline, full_document='updateLookup') as stream:
            for update_change in stream:
                temp = update_change['fullDocument']
                detector = Detector(temp['_id'], temp['VALUE'], temp['TYPE'], temp['LIFE'])
                LOCK.acquire()
                CURRENT_DETECTORS[detector.get_id()] = detector
                LOCK.release()
                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_detectors_collection().watch(pipeline, resume_after=resume_token) as stream:
                for update_change in stream:
                    temp = update_change['fullDocument']
                    detector = Detector(temp['_id'], temp['VALUE'], temp['TYPE'], temp['LIFE'])
                    LOCK.acquire()
                    CURRENT_DETECTORS[detector.get_id()] = detector
                    LOCK.release()


def regenerate_detector():
    # When grizzly deletes a detector during training
    # This callback should create a new detector
    # Add the new detector to the detectors table
    # The Callback in grizzly should retrain the newly added detector

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'delete'}}]
        with DB.get_detectors_collection().watch(pipeline=pipeline) as stream:
            for delete_change in stream:
                deleted_detector = Detector(delete_change['documentKey']['_id'])
                remove_detector(deleted_detector, False)
                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_detectors_collection().watch(pipeline, resume_after=resume_token) as stream:
                for delete_change in stream:
                    deleted_detector = Detector(delete_change['documentKey']['_id'])
                    remove_detector(deleted_detector, False)


def classify_instance(instance):
    for dk, d in CURRENT_DETECTORS.items():
        if d.match(R_VALUE, instance.get_value()):
            LOCK.acquire()
            instance.set_type('SUSPICIOUS')
            instance.set_detector_id(d.get_id())
            DB.add_suspicious_instance(instance)
            LOCK.release()
    LOCK.acquire()
    DB.delete_new_instance(instance)
    LOCK.release()


def classify_initial_new_instances():
    # On startup, evaluate the new instances currently in the new_instance table
    instances = DB.get_all_new_instances()

    for ik, i in instances.items():
        FUTURES.append(EXECUTOR.submit(classify_instance, i))


def classify_new_instances():
    # Create a callback for when a new instance is add to the new_instance table by polar
    # Run through detectors, if a successful match is made, add instance to suspicious_instance table
    # format {_id: _id, VALUE : value, TYPE : type, DETECTOR_id : _id (of detector)}

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'insert'}}]
        with DB.get_new_instances_collection().watch(pipeline=pipeline) as stream:
            for insert_change in stream:
                temp = insert_change['fullDocument']
                inserted_instance = Instance(_id=temp['_id'], value=temp['VALUE'], type=temp['TYPE'])
                FUTURES.append(EXECUTOR.submit(classify_instance, inserted_instance))
                resume_token = stream.resume_token
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_new_instances_collection().watch(pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    temp = insert_change['fullDocument']
                    inserted_instance = Instance(_id=temp['_id'], value=temp['VALUE'], type=temp['TYPE'])
                    FUTURES.append(EXECUTOR.submit(classify_instance, inserted_instance))


if __name__ == "__main__":
    CURRENT_DETECTORS = DB.get_all_detectors()

    FUTURES.append(EXECUTOR.submit(generate_initial_detectors))
    FUTURES.append(EXECUTOR.submit(evaluate_detector_lifespans))
    FUTURES.append(EXECUTOR.submit(update_persistent_detectors))
    FUTURES.append(EXECUTOR.submit(regenerate_detector))
    FUTURES.append(EXECUTOR.submit(classify_initial_new_instances))
    FUTURES.append(EXECUTOR.submit(classify_new_instances))

    wait(FUTURES, return_when='ALL_COMPLETED')

