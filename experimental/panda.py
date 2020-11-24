from common.detector import Detector
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
import threading
import logging
import copy
import time
from models.neural_network_nsa import NeuralNetworkNSA

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


def evaulate_model(grizzly, index):
    global DATASET

    model = NeuralNetworkNSA()
    model.fit(DATASET.get_number_of_features(), grizzly, DATASET.MAX_FEATURES)

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    model.predict(test_x, test_y, DATASET.CLASSES)


def generate_detector(seed):
    # Create and return a detector
    # Format {"Value": value, "TYPE": "INITIAL", "LIFE": current time}

    return Detector(num_features=DATASET.get_number_of_features(), seed=seed)


def experimental_train_dnn_detectors(grizzly, index):
    global DATASET
    global MAX_DETECTORS
    global EXECUTOR
    global FUTURES

    print("Starting dnn detector training for partition " + str(index))
    temp_detectors = {}

    FUTURES = []
    for key, value in DATASET.MAX_FEATURES.items():
        FUTURES.append(EXECUTOR.submit(experimental_generate_dnn_detector, grizzly, key, temp_detectors, value))

    wait(FUTURES, return_when='ALL_COMPLETED')
    FUTURES = []

    print("Finished dnn detector training for partition " + str(index))

    return temp_detectors


def experimental_generate_dnn_detector(grizzly, key, temp_detectors, value):
    global MAX_DETECTORS
    global LOCK
    global NUM_PARTITIONED_THREADS

    counts = {key: 0}

    executor = ThreadPoolExecutor(max_workers=NUM_PARTITIONED_THREADS)
    futures = []
    for i in range(NUM_PARTITIONED_THREADS):
        futures.append(executor.submit(experimental_generate_dnn_detector_multithreading, grizzly, key, counts,
                                       temp_detectors, value))
    wait(futures, return_when='ALL_COMPLETED')

    print('Finished generating detectors for ' + str(key))


def experimental_generate_dnn_detector_multithreading(grizzly, key, counts, temp_detectors, value):
    global MAX_DETECTORS
    global LOCK

    while counts[key] < MAX_DETECTORS:
        detector = generate_detector(value)

        classification = grizzly.classify(detector.get_mean_value())
        if classification != 0:
            LOCK.acquire()
            temp_detectors[detector.get_id()] = detector
            counts[key] += 1
            LOCK.release()


def experimental_test_dnn_detectors(w, index, temp_detectors):
    global DATASET
    global EXECUTOR
    global FUTURES
    global best_accuracy
    global best_r
    global best_results

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    print("Starting dnn detector testing for partition " + str(index))

    start = int(len(test_x[0]) / 4)
    end = len(test_x[0]) - int(len(test_x[0]) / 4)

    # Initialize detection matrix
    detection_matrix = []

    for i in range(len(temp_detectors)):
        temp = []
        for j in range(len(test_x)):
            temp.append(0)
        detection_matrix.append(temp)

    detector_keys = list(temp_detectors.keys())
    for i in range(len(detector_keys)):
        d = temp_detectors[detector_keys[i]]
        FUTURES.append(EXECUTOR.submit(build_detection_matrix, detection_matrix, i, d, test_x))

    wait(FUTURES, return_when='ALL_COMPLETED')
    FUTURES = []

    for r_value in range(start, end+1):
        FUTURES.append(EXECUTOR.submit(analyze_r_value, r_value, detection_matrix, temp_detectors, test_y))

    wait(FUTURES, return_when='ALL_COMPLETED')
    FUTURES = []

    print("Finished dnn detector testing for partition " + str(index))
    w.write('{:^30s}'.format(str(best_r)))
    w.write('{:^30.2f}'.format(best_results[0] * 100.0))
    for x in range(1, len(best_results)):
        w.write(',{:^30.0f}'.format(best_results[x]))
    w.write("\n")
    w.flush()


def build_detection_matrix(matrix, i, d, test_x):
    for j in range(len(test_x)):
        matrix[i][j] = d.compare_to(test_x[j])
    print("Finished " + str(i))


def experimental_train_nsa(index):
    global DATASET
    global EXECUTOR
    global FUTURES
    global best_accuracy
    global best_r
    global best_detectors

    partitions_X, partitions_Y = DATASET.get_partitions()
    validation_x, validation_y = partitions_X[(index + 1) % DATASET.KFOLDS], partitions_Y[(index + 1) % DATASET.KFOLDS]
    training_x, training_y = [], []
    for x in range(DATASET.KFOLDS):
        if x == index or x == (index + 1) % DATASET.KFOLDS:
            continue

        training_x.extend(partitions_X[index])
        training_y.extend(partitions_Y[index])

    best_accuracy = -1.0
    best_r = -1

    start = int(len(training_x[0])/4)
    end = len(training_x[0]) - int(len(training_x[0])/4)

    FUTURES = []
    print("Starting NSA training for partition " + str(index))
    for r_value in range(start, end+1):
        FUTURES.append(EXECUTOR.submit(experimental_generate_nsa_detectors, r_value, training_x, training_y,
                                       validation_x, validation_y))

    wait(FUTURES, return_when='ALL_COMPLETED')
    FUTURES = []

    print("Finished NSA training for partition " + str(index))
    return [best_detectors, best_r]


def experimental_generate_nsa_detectors(r_value, training_x, training_y, validation_x, validation_y):
    global LOCK
    global MAX_DETECTORS
    global best_detectors
    global DATASET
    global NUM_PARTITIONED_THREADS

    executor = ThreadPoolExecutor(max_workers=NUM_PARTITIONED_THREADS)
    futures = []

    temp_detectors = {}
    counts = {}
    for key, value in DATASET.MAX_FEATURES.items():
        counts[key] = 0
        for i in range(NUM_PARTITIONED_THREADS):
            futures.append(executor.submit(experimental_generate_nsa_detectors_multithreading, key, counts,
                                           temp_detectors, r_value, value, training_x, training_y))

    wait(futures, return_when='ALL_COMPLETED')

    analyze_r_value(r_value, temp_detectors, validation_x, validation_y)


def experimental_generate_nsa_detectors_multithreading(key, counts, temp_detectors, r_value, value, training_x,
                                                       training_y):
    global LOCK
    global THREAD_TIMEOUT

    start = time.time()

    while counts[key] < MAX_DETECTORS:
        current = time.time()
        if current - start > THREAD_TIMEOUT and counts[key] == 0:
            # print("Timeout nsa r-value " + str(r_value))
            break

        detector = generate_detector(value)
        match = False
        for j in range(len(training_x)):
            sample_class = list(training_y[j]).index(max(list(training_y[j])))
            if sample_class == 0 and detector.match(r_value, training_x[j]):
                match = True
                break

        if not match:
            LOCK.acquire()
            temp_detectors[detector.get_id()] = detector
            counts[key] += 1
            LOCK.release()


def experimental_test_nsa(w, index, trained_nsa):
    global DATASET
    global best_accuracy
    global best_r
    global best_results
    global EXECUTOR
    global FUTURES

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    best_accuracy = -1.0
    best_r = -1

    print("Starting NSA testing on partition " + str(index))

    temp_detectors = trained_nsa[0]
    r_value = trained_nsa[1]

    FUTURES = [EXECUTOR.submit(analyze_r_value, r_value, temp_detectors, test_x, test_y)]
    wait(FUTURES, return_when='ALL_COMPLETED')
    FUTURES = []

    print("Finished NSA testing for partition " + str(index))
    w.write('{:^30s}'.format(str(best_r)))
    w.write('{:^30.2f}'.format(best_results[0] * 100.0))
    for x in range(1, len(best_results)):
        w.write(',{:^30.0f}'.format(best_results[x]))
    w.write("\n")
    w.flush()


def analyze_r_value(r_value, detection_matrix, detectors, instances_y):
    global best_accuracy
    global best_detectors
    global best_r
    global best_tp
    global best_fp
    global best_tn
    global best_fn
    global best_results
    global LOCK
    global NUM_PARTITIONED_THREADS
    global DATASET

    temp = {}

    for i in range(len(DATASET.CLASSES)):
        temp[DATASET.CLASSES[i]] = [0, 0]

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    # for each sample, check if there is a detector over the r-value threshold
    for j in range(len(instances_y)):

        sample_class = list(instances_y[j]).index(max(list(instances_y[j])))
        detected = False

        for i in range(len(detection_matrix)):
            if detection_matrix[i][j] > r_value:
                detected = True
                break

        if detected:
            if sample_class == 0:
                temp[DATASET.CLASSES[sample_class]][1] += 1
                fp += 1
            else:
                temp[DATASET.CLASSES[sample_class]][0] += 1
                tp += 1
        else:
            if sample_class == 0:
                temp[DATASET.CLASSES[sample_class]][0] += 1
                tn += 1
            else:
                temp[DATASET.CLASSES[sample_class]][1] += 1
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # print("Accuracy: " + str(accuracy))

    if accuracy > best_accuracy:
        LOCK.acquire()
        best_accuracy = accuracy
        best_detectors = detectors
        best_r = r_value

        temp[0] = accuracy
        best_results = temp
        # print('Best Accuracy' + str(best_accuracy))
        LOCK.release()


def analyze_r_value(r_value, temp_detectors, instances_x, instances_y):
    global best_accuracy
    global best_detectors
    global best_r
    global best_tp
    global best_fp
    global best_tn
    global best_fn
    global best_results
    global LOCK
    global NUM_PARTITIONED_THREADS
    global DATASET

    if len(temp_detectors) == 0:
        return

    partition_x = {}
    partition_y = {}

    partition_size = int(len(instances_x) / NUM_PARTITIONED_THREADS)

    # partition dataset for multithreading
    for i in range(NUM_PARTITIONED_THREADS):

        if i < NUM_PARTITIONED_THREADS - 1:
            partition_x[i] = instances_x[partition_size * i: partition_size * (i+1)]
            partition_y[i] = instances_y[partition_size * i: partition_size * (i + 1)]
        else:
            partition_x[i] = instances_x[partition_size * i:]
            partition_y[i] = instances_y[partition_size * i:]

    # print('r_value: ' + str(r_value))

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    r_results = [0]

    for i in range(len(DATASET.CLASSES)):
        r_results.append(0)
        r_results.append(0)

    futures = []
    print("Starting analysis for r-value " + str(r_value))
    with ThreadPoolExecutor(max_workers=NUM_PARTITIONED_THREADS) as executor:
        for j in range(NUM_PARTITIONED_THREADS):
            futures.append(executor.submit(analyze_r_value_partitioned, r_value, temp_detectors, partition_x[j],
                                           partition_y[j]))

        for f in as_completed(futures):
            results = f.result()

            tp += results[0]
            fp += results[1]
            tn += results[2]
            fn += results[3]
            r = results[4]

            for i in range(len(r)):
                r_results[i+1] += r[i]

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # print("Accuracy: " + str(accuracy))

        if accuracy > best_accuracy:
            LOCK.acquire()
            best_accuracy = accuracy
            best_detectors = temp_detectors
            best_r = r_value

            r_results[0] = accuracy
            best_results = r_results
            # print('Best Accuracy' + str(best_accuracy))
            LOCK.release()

    print('Finished analysis for r-value ' + str(r_value))


def analyze_r_value_partitioned(r_value, temp_detectors, instances_x, instances_y):
    global DATASET

    temp = {}

    for i in range(len(DATASET.CLASSES)):
        temp[DATASET.CLASSES[i]] = [0, 0]

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    try:
        for j in range(len(instances_x)):
            matched = False
            for key, detector in temp_detectors.items():
                if detector.match(r_value, instances_x[j]):
                    matched = True
                    break

            sample_class = list(instances_y[j]).index(max(list(instances_y[j])))

            if matched:
                if sample_class == 0:
                    temp[DATASET.CLASSES[sample_class]][1] += 1
                    fp += 1
                else:
                    temp[DATASET.CLASSES[sample_class]][0] += 1
                    tp += 1
            else:
                if sample_class == 0:
                    temp[DATASET.CLASSES[sample_class]][0] += 1
                    tn += 1
                else:
                    temp[DATASET.CLASSES[sample_class]][1] += 1
                    fn += 1

        results = []

        for i in range(len(DATASET.CLASSES)):
            results.extend(temp[DATASET.CLASSES[i]])

        return tp, fp, tn, fn, results

    except Exception as e:
        logging.error(e)
