from database import MongoDBConnect
from instance import Instance
import threading
from concurrent.futures import ThreadPoolExecutor
import csv
import time

EXECUTOR = ThreadPoolExecutor(6)
FUTURES = []
LOCK = threading.Lock()

DB = MongoDBConnect()


def add_new_instance(row):
    # Add detector to detectors table and persistent memory

    instance = Instance(type=bin(0).encode())
    for feature in row:
        instance.add_feature(feature)
    instance.calculate_crc32()
    add_new_instance(instance)

    LOCK.acquire()
    DB.add_instance(instance)
    LOCK.release()

    time.sleep(10)


if __name__ == "__main__":
    option = input('Read from csv file: y/n\n')

    while True:
        if option == 'y':
            filename = input('\nFilename to read from\n')
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    FUTURES.append(EXECUTOR.submit(add_new_instance, row))
            break
        elif option == 'n':
            # begin CICFlowmeter-V4.0
            print('\nStarting CICFlowmeter-V4.0')
            break
        else:
            print('Invalid input')
            option = input('Read from csv file: y/n\n')

    for f in FUTURES:
        print(f.result())

