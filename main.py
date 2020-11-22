import threading
from concurrent.futures import ThreadPoolExecutor, wait
from components import grizzly, panda
from components import polar
import csv
import sys
from cache import dataset, detectors, new_instances, suspicious_instances, validated_instances
from os import path
import logging

EXECUTOR = ThreadPoolExecutor(10)
FUTURES = []
SAVE_FILE = None


def start_polar():
    ############################################
    ############################################
    ############################################
    ############################################
    # POLAR
    option = input('Read new instances from csv file: y/n\n')

    while True:
        if option == 'y':
            filename = input('\nFilename to read from\n')
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    polar.add_new_instance(row)
            break
        elif option == 'n':
            # begin CICFlowmeter-V4.0
            print('\nStarting CICFlowmeter-V4.0')
            break
        else:
            print('Invalid input')
            option = input('Read from csv file: y/n\n')


def start_panda():
    ############################################
    ############################################
    ############################################
    ############################################
    # PANDA
    global SAVE_FILE

    panda.set_common(detectors, new_instances, suspicious_instances, dataset, SAVE_FILE)
    panda.generate_initial_detectors()

    FUTURES.append(EXECUTOR.submit(panda.evaluate_detector_lifespans))
    print('Evaluating detector lifespans')

    FUTURES.append(EXECUTOR.submit(panda.monitor_new_instances))
    print('Monitoring new instances')


def start_grizzly():
    ############################################
    ############################################
    ############################################
    ############################################
    # GRIZZLY
    global SAVE_FILE

    grizzly.set_common(detectors, validated_instances, suspicious_instances, dataset)

    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == '0':
            filename = input('\nEnter the Deep Neural Network filename to load from\n')
            grizzly.load_dnn(filename)
            break
        elif option == '1':
            grizzly.train_dnn()
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

    save_prompt = input('Save Deep Neural Network: y/n\n')

    while True:
        if save_prompt == 'y':
            filename = input('Enter the Deep Neural Network filename to save to\n')
            grizzly.save_dnn(filename)
            sys.exit(0)
        elif save_prompt == 'n':
            sys.exit(0)
        else:
            print('\nInvalid input')
            save_prompt = input('Save Deep Neural Network: y/n\n')

    # FUTURES.append(EXECUTOR.submit(grizzly.monitor_dataset))
    # FUTURES.append(EXECUTOR.submit(grizzly.monitor_suspicious_instances))
    # FUTURES.append(EXECUTOR.submit(grizzly.monitor_detectors))


def start_dataset():
    file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
    while True:
        if file_prompt == 'y':
            # filename = input('\nEnter the csv filename\n'
            #                 '**If there are multiple files, separate with a comma (no spaces)**\n')
            filename = 'data/Day1.csv,data/Day2.csv'
            # filename = 'data/Sample.csv'
            dataset.read_from_file(filename)
            break
        elif file_prompt == 'n':
            # grizzly.train_dnn()
            break
        else:
            print('\nInvalid input')
            file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')


if __name__ == "__main__":
    # start_polar()
    # start_grizzly()
    # start_panda()


    option = input(
        '\nSave Deep Neural Network: y/n\n'
    )
    if option == 'y':
        filename = input('\nEnter the Deep Neural Network filename to save to\n')
        grizzly.save_dnn(filename)

    wait(FUTURES, return_when='ALL_COMPLETED')






