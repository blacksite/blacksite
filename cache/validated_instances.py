import threading
import logging

LOCK = threading.Lock()
VALIDATED_INSTANCES = {}


def add_validated_instance(instance):
    global VALIDATED_INSTANCES
    global LOCK

    LOCK.acquire()
    VALIDATED_INSTANCES[instance.get_id()] = instance
    LOCK.release()


def size():
    global VALIDATED_INSTANCES

    return len(VALIDATED_INSTANCES)


def remove_validated_instance(instance):
    global VALIDATED_INSTANCES
    global LOCK

    LOCK.acquire()
    try:
        del VALIDATED_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be deleted")
    LOCK.release()


def get_validated_instance(instance):
    global VALIDATED_INSTANCES

    try:
        return VALIDATED_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be retrieved")
        return None


def get_all_validated_instances():
    global VALIDATED_INSTANCES

    return VALIDATED_INSTANCES.items()


def pop():
    global VALIDATED_INSTANCES

    try:
        return VALIDATED_INSTANCES.pop()
    except:
        logging.error("Could not pop")
        return None