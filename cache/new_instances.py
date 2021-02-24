import threading
import logging

LOCK = threading.Lock()
NEW_INSTANCES = {}


def add_new_instance(instance):
    global NEW_INSTANCES
    global LOCK

    LOCK.acquire()
    NEW_INSTANCES[instance.get_id()] = instance
    LOCK.release()


def size():
    global NEW_INSTANCES

    return len(NEW_INSTANCES)


def remove_new_instance(instance):
    global NEW_INSTANCES
    global LOCK

    LOCK.acquire()
    try:
        del NEW_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be deleted")
    LOCK.release()


def get_new_instance(instance):
    global NEW_INSTANCES

    try:
        return NEW_INSTANCES[instance.get_id()]
    except:
        logging.error("Instance " + instance.get_id() + " could not be retrieved")
        return None


def get_all_new_instances():
    global NEW_INSTANCES

    return NEW_INSTANCES.items()


def pop():
    global NEW_INSTANCES

    try:
        return NEW_INSTANCES.pop()
    except:
        logging.error("Could not pop")
        return None
