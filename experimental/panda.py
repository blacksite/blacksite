from concurrent.futures import ThreadPoolExecutor, wait, as_completed
import threading
from models import NeuralNetworkNSA, StandardNSA
import numpy as np

MAX_DETECTORS = 1000
EXECUTOR = ThreadPoolExecutor(max_workers=500)
NUM_PARTITIONED_THREADS = 50
FUTURES = []
GRIZZLY = None
DETECTORS = None
NEW_INSTANCES = None
SUSPICIOUS_INSTANCES = None
DATASET = None
BEST_R_FOR_NSA = None
THREAD_TIMEOUT = 120
best_detectors = {}
best_accuracy = -1.0
best_r = -1
best_tp = 0.0
best_fp = 0.0
best_tn = 0.0
best_fn = 0.0
best_results = []
LOCK = threading.Lock()


def set_common(detectors, new_instances, suspicious_instances, dataset):
    global DETECTORS
    global NEW_INSTANCES
    global SUSPICIOUS_INSTANCES
    global DATASET
    global NUM_PARTITIONED_THREADS

    DETECTORS = detectors
    NEW_INSTANCES = new_instances
    SUSPICIOUS_INSTANCES = suspicious_instances
    DATASET = dataset


def evaluate_nsa(w, index):
    global DATASET

    model = StandardNSA()

    partitions_X, partitions_Y = DATASET.get_partitions()

    train_x, train_y = [], []

    for key, value in partitions_X.items():
        if key != index:
            train_x.extend(value)
            train_y.extend(partitions_Y[key])

    test_x, test_y = partitions_X[index], partitions_Y[index]

    model.fit(np.array(train_x), np.array(train_y), DATASET.MAX_FEATURES, DATASET.CLASSES)


def evaluate_dnn(w, grizzly, index, number_of_detectors_per_class=1000, std_dev=1):
    global DATASET

    model = NeuralNetworkNSA()
    model.fit(DATASET.get_number_of_features(), grizzly, DATASET.MAX_FEATURES, number_of_detectors_per_class=number_of_detectors_per_class, std_dev=std_dev)

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    results = model.predict(np.array(test_x, dtype='f4'), np.array(test_y, dtype='i4'), DATASET.CLASSES)

    # print("Finished dnn detector testing for partition " + str(index))
    w.write('{:^30.2f}'.format(results['Accuracy'] * 100.0))
    w.write('{:^30s}'.format(str(results['r-value'])))
    for c in DATASET.CLASSES:
        w.write(',{:^30.0f},{:^30.0f}'.format(results[c]['correct'], results[c]['incorrect']))
    w.write("\n")
    w.flush()