from concurrent.futures import ThreadPoolExecutor
import threading
from experimental.exp_models import NeuralNetworkNSA
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
model = NeuralNetworkNSA()


def set_common(dataset):

    global DATASET

    DATASET = dataset


def initialize_model():
    global model

    model = NeuralNetworkNSA()


def evaluate_dnn(w, grizzly, index, number_of_detectors_per_class, std_dev):
    global DATASET
    global model

    partitions_X, partitions_Y = DATASET.get_partitions()
    training_x, training_y = [], []
    test_x, test_y = partitions_X[index], partitions_Y[index]

    for x in range(DATASET.KFOLDS):
        if x != index:
            training_x.extend(partitions_X[x])
            training_y.extend(partitions_Y[x])


    model.fit(np.array(training_x, dtype='f4'), np.array(training_y, dtype='i4'), DATASET.CLASSES,
              DATASET.get_number_of_features(), grizzly, DATASET.MAX_FEATURES,
              number_of_detectors=number_of_detectors_per_class, std_dev=std_dev)

    results = model.predict(np.array(test_x, dtype='f4'), np.array(test_y, dtype='i4'), DATASET.CLASSES)

    # print("Finished dnn detector testing for partition " + str(index))
    w.write('{:^40.2f}'.format(results['Accuracy'] * 100.0))
    w.write(',{:^40s}'.format(str(results['r-value'])))
    for c in DATASET.CLASSES:
        w.write(',{:^40.0f},{:^40.0f}'.format(results[c]['correct'], results[c]['incorrect']))
    w.write("\n")
    w.flush()


def evaluate_ind_dnn(w, grizzly, index, key, number_of_detectors):
    global DATASET
    global model

    partitions_X, partitions_Y = DATASET.get_partitions()
    training_x, training_y = [], []
    test_x, test_y = partitions_X[key][index], partitions_Y[key][index]

    for x in range(DATASET.KFOLDS):
        if x != index:
            training_x.extend(partitions_X[key][x])
            training_y.extend(partitions_Y[key][x])


    model.fit(np.array(training_x, dtype='f4'), np.array(training_y, dtype='i4'), DATASET.CLASSES,
              DATASET.get_number_of_features(), grizzly, DATASET.MAX_FEATURES[key],
              number_of_detectors=number_of_detectors)

    results = model.predict(np.array(test_x, dtype='f4'), np.array(test_y, dtype='i4'), DATASET.CLASSES)

    # print("Finished dnn detector testing for partition " + str(index))
    w.write('{:^10s}'.format(str(results[0])))
    w.write(',{:^10.2f}'.format(results[1] * 100.0))
    for i in range(2, len(results)):
        w.write(',{:^10.0f}'.format(results[i]))
    w.write("\n")
    w.flush()
