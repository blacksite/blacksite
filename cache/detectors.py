import threading
import logging

LOCK = threading.Lock()
DETECTORS = {}


def add_detector(detector):
    global DETECTORS
    global LOCK

    LOCK.acquire()
    DETECTORS[detector.get_id()] = detector
    LOCK.release()


def remove_detector(detector):
    global DETECTORS
    global LOCK

    LOCK.acquire()
    try:
        del DETECTORS[detector.get_id()]
    except:
        logging.error("detector " + detector.get_id() + " could not be deleted")
    LOCK.release()


def get_detector(detector):
    global DETECTORS
    global LOCK

    try:
        return DETECTORS[detector.get_id()]
    except:
        logging.error("detector " + detector.get_id() + " could not be retrieved")
        return None


def size():
    global DETECTORS

    return len(DETECTORS)


def update_detector(detector):
    global DETECTORS
    global LOCK

    LOCK.acquire()
    try:
        DETECTORS[detector.get_id()] = detector
    except:
        logging.error("detector " + detector.get_id() + " could not be updated")
    LOCK.release()


def get_all_detectors():
    global DETECTORS

    return DETECTORS


def get_initial_detectors():
    global DETECTORS

    initial_detectors = {}

    for key, value in DETECTORS.items():
        if value.get_type() == 'INITIAL':
            initial_detectors[key] = value

    return initial_detectors


def set_detectors(detectors):
    global DETECTORS

    DETECTORS = detectors


def clear():
    global DETECTORS

    LOCK.acquire()
    DETECTORS = {}
    LOCK.release()