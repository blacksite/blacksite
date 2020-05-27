import deepneuralnetwork as dnn
import threading
from concurrent.futures import ThreadPoolExecutor, wait

EXECUTOR = ThreadPoolExecutor(10)
FUTURES = []
LOCK = threading.Lock()

if __name__ == "__main__":

    option = input(
        '0: Load Deep Neural Network\n'
        '1: Train New Deep Neural Network\n'
    )

    while True:
        if option == 0:
            while True:
                filename = input('Enter the Deep Neural Network filename\n')
                dnn.load_dnn(filename)
            break
        elif option == 1:
            dnn.train_dnn()
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

    FUTURES.append(EXECUTOR.submit(dnn.retrain_dnn_callback))
    FUTURES.append(EXECUTOR.submit(dnn.train_initial_detectors))
    FUTURES.append(EXECUTOR.submit(dnn.retrain_detectors_callback))
    FUTURES.append(EXECUTOR.submit(dnn.evaluate_initial_suspicious_instances))
    FUTURES.append(EXECUTOR.submit(dnn.evaluate_suspicious_instances_callback))

    wait(FUTURES, return_when='ALL_COMPLETED')

