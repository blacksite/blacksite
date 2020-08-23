import threading
from concurrent.futures import ThreadPoolExecutor, wait
from common.database import MongoDBConnect
import grizzly.grizzly
import panda.panda
import polar.polar
import csv
import sys

EXECUTOR = ThreadPoolExecutor(10)
FUTURES = []
LOCK = threading.Lock()
DB = MongoDBConnect()

if __name__ == "__main__":
    option = input('Read from csv file: y/n\n')

    while True:
        if option == 'y':
            filename = input('\nFilename to read from\n')
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    FUTURES.append(EXECUTOR.submit(polar.add_new_instance, row))
            break
        elif option == 'n':
            # begin CICFlowmeter-V4.0
            print('\nStarting CICFlowmeter-V4.0')
            break
        else:
            print('Invalid input')
            option = input('Read from csv file: y/n\n')

    panda.CURRENT_DETECTORS = DB.get_all_detectors()

    if len(panda.CURRENT_DETECTORS) < panda.MAX_DETECTORS:
        panda.generate_initial_detectors()
        print('Generating initial detectors')

    FUTURES.append(EXECUTOR.submit(panda.evaluate_detector_lifespans))
    print('Evaluating detector lifespans')

    FUTURES.append(EXECUTOR.submit(panda.update_persistent_detectors))
    print('Responding to detector updates')

    FUTURES.append(EXECUTOR.submit(panda.regenerate_detector))
    print('Regenerating deleted detectors')

    # FUTURES.append(EXECUTOR.submit(panda.classify_initial_new_instances))
    print('Classifying currently active new instances')

    #  FUTURES.append(EXECUTOR.submit(panda.classify_new_instances))
    print('Classifying new instances')

    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == '0':
            filename = input('\nEnter the Deep Neural Network filename to load from\n')
            # dnn.load_dnn(filename)
            break
        elif option == '1':
            file_prompt = input('\nWould you like to use a dataset from a csv file?: y/n\n')
            while True:
                if file_prompt == 'y':
                    filename = input('\nEnter the csv filename\n')
                    grizzly.train_dnn(filename)
                    break
                elif file_prompt == 'n':
                    grizzly.train_dnn()
                    break
                else:
                    print('\nInvalid input')
                    file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
            break
        else:
            print('\nInvalid input')
            option = input(
                '0: Load Deep Neural Network\n'
                '1: Train New Deep Neural Network\n'
            )

    # Start the callback to watch the dataset table
    # When a threshold is reached, a new DNN is trained
    # It replaces the current DNN

    option = input(
        '\n0: Continue\n'
        '1: Exit\n'
    )

    while True:
        if option == '0':
            break
        elif option == '1':
            save_prompt = input('\nSave Deep Neural Network: y/n\n')

            while True:
                if save_prompt == 'y':
                    filename = input('\nEnter the Deep Neural Network filename to save to\n')
                    grizzly.save_dnn(filename)
                    sys.exit(0)
                elif save_prompt == 'n':
                    sys.exit(0)
                else:
                    print('\nInvalid input')
                    save_prompt = input('Save Deep Neural Network: y/n\n')
        else:
            print('\nInvalid input')
            option = input(
                '0: Continue\n'
                '1: Exit\n'
            )

    FUTURES.append(EXECUTOR.submit(grizzly.retrain_dnn_callback))
    FUTURES.append(EXECUTOR.submit(grizzly.train_initial_detectors))
    FUTURES.append(EXECUTOR.submit(grizzly.retrain_detectors_callback))
    FUTURES.append(EXECUTOR.submit(grizzly.evaluate_initial_suspicious_instances))
    FUTURES.append(EXECUTOR.submit(grizzly.evaluate_suspicious_instances_callback))

    wait(FUTURES, return_when='ALL_COMPLETED')

    option = input(
        '\n0: Save Deep Neural Network\n'
        '1: Exit\n'
    )

    while True:
        if option == '0':
            filename = input('\nEnter the Deep Neural Network filename to save to\n')
            grizzly.save_dnn(filename)
            break
        elif option == 1:
            sys.exit(0)
        else:
            print('\nInvalid input')
            option = input(
                '0: Save Deep Neural Network\n'
                '1: Exit\n'
            )
