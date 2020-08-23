from common.database import MongoDBConnect
from common.instance import Instance
import threading
import time

LOCK = threading.Lock()

DB = MongoDBConnect()


def add_new_instance(row):
    # Add detector to detectors table and persistent memory

    instance = Instance(type=bin(0).encode())
    for feature in row:
        instance.add_feature(feature)
    add_new_instance(instance)

    LOCK.acquire()
    DB.add_instance(instance)
    LOCK.release()

    time.sleep(10)




