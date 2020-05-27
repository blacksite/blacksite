import deepneuralnetwork as dnn
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import sys

EXECUTOR = ThreadPoolExecutor(10)
FUTURES = []
LOCK = threading.Lock()

if __name__ == "__main__":

    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == '0':
            filename = input('Enter the Deep Neural Network filename to load from\n')
            # dnn.load_dnn(filename)
            break
        elif option == '1':
            file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
            while True:
                if file_prompt == 'y':
                    filename = input('Enter the csv filename\n')
                    dnn.DNN = dnn.train_dnn(filename)
                    break
                elif file_prompt == 'n':
                    dnn.DNN = dnn.train_dnn()
                    break
                else:
                    print('Invalid input')
                    file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
            break
        else:
            print('Invalid input')
            option = input(
                '0: Load Deep Neural Network\n'
                '1: Train New Deep Neural Network\n'
            )
    # Start the callback to watch the dataset table
    # When a threshold is reached, a new DNN is trained
    # It replaces the current DNN

    option = input(
        '0: Continue\n'
        '1: Exit\n'
    )

    while True:
        if option == '0':
            break
        elif option == '1':
            save_prompt = input('Save Deep Neural Network: y/n\n')

            while True:
                if save_prompt == 'y':
                    filename = input('Enter the Deep Neural Network filename to save to\n')
                    dnn.save_dnn(filename)
                    sys.exit(0)
                elif save_prompt == 'n':
                    sys.exit(0)
                else:
                    print('Invalid input')
                    save_prompt = input('Save Deep Neural Network: y/n\n')
        else:
            print('Invalid input')
            option = input(
                '0: Continue\n'
                '1: Exit\n'
            )

    FUTURES.append(EXECUTOR.submit(dnn.retrain_dnn_callback))
    FUTURES.append(EXECUTOR.submit(dnn.train_initial_detectors))
    FUTURES.append(EXECUTOR.submit(dnn.retrain_detectors_callback))
    FUTURES.append(EXECUTOR.submit(dnn.evaluate_initial_suspicious_instances))
    FUTURES.append(EXECUTOR.submit(dnn.evaluate_suspicious_instances_callback))

    wait(FUTURES, return_when='ALL_COMPLETED')

    option = input(
        '0: Save Deep Neural Network\n'
        '1: Exit\n'
    )

    while True:
        if option == '0':
            filename = input('Enter the Deep Neural Network filename to save to\n')
            dnn.save_dnn(filename)
            break
        elif option == 1:
            sys.exit(0)
        else:
            print('Invalid input')
            option = input(
                '0: Save Deep Neural Network\n'
                '1: Exit\n'
            )
