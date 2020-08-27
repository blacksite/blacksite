from common.database import MongoDBConnect
from common.detector import Detector
from common.instance import Instance
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pymongo
import logging

R_VALUE = 16
MAX_DETECTORS = 1000
DETECTOR_LENGTH = 32
INITIAL_DETECTOR_LIFESPAN = 60
IMMATURE_DETECTOR_LIFESPAN = 604800
MATURE_DETECTOR_LIFESPAN = 2592000
MEMORY_DETECTOR_LIFESPAN = 31536000
EXECUTOR = ThreadPoolExecutor(6)
FUTURES = []
LOCK = threading.Lock()

DB = MongoDBConnect()
CURRENT_DETECTORS = {}


def generate_detector(_id=None):
    # Create and return a detector
    # Format {"Value": value, "TYPE": "INITIAL", "LIFE": current time}

    value = random.getrandbits(32) & 0xffffffff
    # value = ""
    #
    # for i in range(DETECTOR_LENGTH):
    #     temp = str(random.randint(0, 1))
    #     value += temp
    #
    # value = value.encode()

    type = bin(0).encode()

    detector = Detector(value=value, type=type, life=time.time())

    if _id:
        detector.set_id(_id)

    return detector


def replace_detector(detector, add_to_database=False):
    detector = generate_detector(detector.get_id())
    # Remove detector from database and persistent memory
    LOCK.acquire()
    if add_to_database:
        DB.add_detector(detector)
    else:
        DB.update_detector(detector)
    try:
        CURRENT_DETECTORS[detector.get_id()] = detector
    except TypeError as error:
        logging.error(error)
    LOCK.release()


def add_new_detector():
    # Add detector to detectors table and persistent memory

    detector = generate_detector()
    LOCK.acquire()
    detector.set_id(DB.add_detector(detector))
    CURRENT_DETECTORS[detector.get_id()] = detector
    LOCK.release()


def generate_initial_detectors():
    # On startup, generate detectors to meet the desired number of detectors
    num_detectors = len(CURRENT_DETECTORS)
    detectors_needed = MAX_DETECTORS - num_detectors
    detectors_generated = 0
    while num_detectors < MAX_DETECTORS:
        add_new_detector()
        num_detectors = len(CURRENT_DETECTORS)
        detectors_generated += 1
        print('{:.2f}%'.format(float(detectors_generated)/detectors_needed*100))


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
                LOCK.acquire()
                detector = CURRENT_DETECTORS[key]
                LOCK.release()
                lifetime = time.time() - detector.get_life()
                if detector.get_type() == bin(0).encode():
                    if lifetime > INITIAL_DETECTOR_LIFESPAN:
                        replace_detector(detector)
                        sum += 1
                elif detector.get_type() == bin(1).encode():
                    if lifetime > IMMATURE_DETECTOR_LIFESPAN:
                        replace_detector(detector)
                        sum += 1
                elif detector.get_type() == bin(2).encode():
                    if lifetime > MATURE_DETECTOR_LIFESPAN:
                        replace_detector(detector)
                        sum += 1
                elif detector.get_type() == bin(3).encode():
                    if lifetime > MEMORY_DETECTOR_LIFESPAN:
                        replace_detector(detector)
                        sum += 1

            # if sum > 0:
            #    print('{:d} detectors were replaced'.format(sum))
    except RuntimeError as error:
        logging.error(error)
    except SystemError as error:
        logging.error(error)


def update_persistent_detectors():
    # When a detector is updated from initial to immature / immature to mature / mature to memory
    # Update the persistent table in memory of detectors

    try:
        resume_token = None
        pipeline = [{'$match': {'operationType': 'update'}}]
        with DB.get_detectors_collection().watch(pipeline) as stream:
            for update_change in stream:
                updated_id = update_change['documentKey']['_id']
                updated_fields = update_change['updateDescription']['updatedFields']
                if 'TYPE' in updated_fields:
                    if updated_fields['TYPE'] != bin(0).encode():
                        updated_detector = Detector(_id=updated_id, type=updated_fields['TYPE'])
                        LOCK.acquire()
                        CURRENT_DETECTORS[updated_detector.get_id()].set_type(updated_detector.get_type())
                        LOCK.release()
                        # print('Updated')
                resume_token = update_change['_id']
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_detectors_collection().watch(pipeline, resume_after=resume_token) as stream:
                for update_change in stream:
                    updated_id = update_change['documentKey']['_id']
                    updated_fields = update_change['updateDescription']['updatedFields']
                    if 'TYPE' in updated_fields:
                        if updated_fields['TYPE'] != bin(0).encode():
                            updated_detector = Detector(_id=updated_id, type=updated_fields['TYPE'])
                            LOCK.acquire()
                            CURRENT_DETECTORS[updated_detector.get_id()].set_type(updated_detector.get_type())
                            LOCK.release()
                            # print('Updated')


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
                # print('Regeneration fired')
                deleted_detector = Detector(_id=delete_change['documentKey']['_id'])
                replace_detector(deleted_detector, True)
                resume_token = delete_change['_id']
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_detectors_collection().watch(pipeline, resume_after=resume_token) as stream:
                for delete_change in stream:
                    deleted_detector = Detector(_id=delete_change['documentKey']['_id'])
                    replace_detector(deleted_detector, True)


def classify_instance(instance):
    # Classify a single instance using Current Detectors
    for dk, d in CURRENT_DETECTORS.items():
        if d.match(R_VALUE, instance.get_value()):
            LOCK.acquire()
            instance.set_type(bin(1).encode())
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
                insert_change['_id']
    except pymongo.errors.PyMongoError as error:
        if resume_token is None:
            logging.error('...' + str(error))
        else:
            with DB.get_new_instances_collection().watch(pipeline, resume_after=resume_token) as stream:
                for insert_change in stream:
                    temp = insert_change['fullDocument']
                    inserted_instance = Instance(_id=temp['_id'], value=temp['VALUE'], type=temp['TYPE'])
                    FUTURES.append(EXECUTOR.submit(classify_instance, inserted_instance))