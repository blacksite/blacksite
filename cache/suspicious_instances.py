import threading
import logging

LOCK = threading.Lock()
SUSPICIOUS_INSTANCES = {}


def add_suspicious_instance(instance):
    global SUSPICIOUS_INSTANCES
    global LOCK

    LOCK.acquire()
    SUSPICIOUS_INSTANCES[instance.get_id()] = instance
    LOCK.release()


def size():
    global SUSPICIOUS_INSTANCES

    return len(SUSPICIOUS_INSTANCES)


def remove_suspicious_instance(instance):
    global SUSPICIOUS_INSTANCES
    global LOCK

    LOCK.acquire()
    try:
        del SUSPICIOUS_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be deleted")
    LOCK.release()


def get_suspicious_instance(instance):
    global SUSPICIOUS_INSTANCES

    try:
        return SUSPICIOUS_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be retrieved")
        return None


def get_all_suspicious_instances():
    global SUSPICIOUS_INSTANCES

    return SUSPICIOUS_INSTANCES.items()


def pop():
    global SUSPICIOUS_INSTANCES

    try:
        return SUSPICIOUS_INSTANCES.pop()
    except:
        logging.error("Could not pop")
        return None
