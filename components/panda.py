from common.detector import Detector
from common.instance import Instance
import random
import time
from concurrent.futures import ThreadPoolExecutor
R_VALUE = 16
MAX_DETECTORS = 1000
DETECTOR_LENGTH = 32
INITIAL_DETECTOR_LIFESPAN = 60
IMMATURE_DETECTOR_LIFESPAN = 604800
MATURE_DETECTOR_LIFESPAN = 2592000
MEMORY_DETECTOR_LIFESPAN = 31536000
EXECUTOR = ThreadPoolExecutor(6)
FUTURES = []
GRIZZLY = None
DETECTORS = None
NEW_INSTANCES = None
SUSPICIOUS_INSTANCES = None
DATASET = None
BEST_R_FOR_NSA = None


def set_common(detectors, new_instances, suspicious_instances, dataset):
    global LOCK
    global DETECTORS
    global NEW_INSTANCES
    global SUSPICIOUS_INSTANCES
    global DATASET

    DETECTORS = detectors
    NEW_INSTANCES = new_instances
    SUSPICIOUS_INSTANCES = suspicious_instances
    DATASET = dataset


def generate_detector():
    # Create and return a detector
    # Format {"Value": value, "TYPE": "INITIAL", "LIFE": current time}

    return Detector(DATASET.get_number_of_features())


def add_new_detector():
    # Add detector to detectors table and persistent memory
    global DETECTORS

    detector = generate_detector()
    DETECTORS.add_detector(detector)


def generate_initial_detectors():
    # On startup, generate detectors to meet the desired number of detectors
    global DETECTORS

    detectors_needed = MAX_DETECTORS - DETECTORS.size()
    detectors_generated = 0
    while DETECTORS.size() < MAX_DETECTORS:
        add_new_detector()
        detectors_generated += 1
        print('{:.2f}%'.format(float(detectors_generated)/detectors_needed*100))


def remove_detector(detector):
    global DETECTORS

    DETECTORS.remove_detector(detector)


def evaluate_detector_lifespans():
    # Continuously iterate through detectors to determine if lifespan has elapsed
    # If the lifespan has elapsed, delete the detector and create a new one
    # Add the new detector to the detectors table and to persistent list
    global DETECTORS

    print('Detector lifespan evaluation started')
    while True:

        sum = 0
        for key, detector in DETECTORS.get_all_detectors().items():
            lifetime = time.time() - detector.get_life()
            if detector.get_type() == 'INITIAL':
                if lifetime > INITIAL_DETECTOR_LIFESPAN:
                    remove_detector(detector)
                    add_new_detector()
                    sum += 1
            elif detector.get_type() == 'IMMATURE':
                if lifetime > IMMATURE_DETECTOR_LIFESPAN:
                    remove_detector(detector)
                    add_new_detector()
                    sum += 1
            elif detector.get_type() == 'MATURE':
                if lifetime > MATURE_DETECTOR_LIFESPAN:
                    remove_detector(detector)
                    add_new_detector()
                    sum += 1
            elif detector.get_type() == 'MEMORY':
                if lifetime > MEMORY_DETECTOR_LIFESPAN:
                    remove_detector(detector)
                    add_new_detector()
                    sum += 1

            # if sum > 0:
            #    print('{:d} detectors were replaced'.format(sum))


def classify_instance(instance):
    # Classify a single instance using Current Detectors
    global DETECTORS

    for key, d in DETECTORS.get_all_detectors().items():
        if d.match(R_VALUE, instance.get_value()):
            instance.add_detector_id(key)


def monitor_new_instances():
    global NEW_INSTANCES
    global SUSPICIOUS_INSTANCES

    while True:
        if NEW_INSTANCES.size() > 0:
            new_instance = NEW_INSTANCES.pop()

            classify_instance(new_instance)

            if len(new_instance.get_detector_ids()) > 0:
                SUSPICIOUS_INSTANCES.add_instance(new_instance)


def experimental_train_dnn_detectors(grizzly, index):

    print("Starting dnn detector training for partition " + str(index))
    temp_detectors = {}
    while len(temp_detectors) < MAX_DETECTORS:
        detector = generate_detector()
        classification = grizzly.classify(detector.get_mean_value())
        if classification[0] > 0.5:
            temp_detectors[detector.get_id()] = detector
    print("Finished dnn detector training for partition " + str(index))

    return temp_detectors


def experimental_test_dnn_detectors(w, index, temp_detectors):
    global DATASET

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    print("Starting dnn detector testing for partition " + str(index))
    best_accuracy = -1.0
    best_r = -1
    best_tp = 0.0
    best_fp = 0.0
    best_tn = 0.0
    best_fn = 0.0

    for r_value in range(1, len(test_x[0])):
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0
        for j in range(len(test_x)):
            for key, detector in temp_detectors.items():
                matched = False
                if detector.match(r_value, test_x[j]):
                    matched = True
                    break

                if matched:
                    if test_y[j] == 0:
                        fp += 1
                    else:
                        tp += 1
                else:
                    if test_y[j] == 0:
                        tn += 1
                    else:
                        fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_r = r_value
            best_tp = tp
            best_fp = fp
            best_tn = tn
            best_fn = fn
    print("Finished dnn detector training for partition " + str(index))
    w.write("DNN Trained Detectors\n")
    w.write("Optimized r-value: " + str(best_r) + "\n")
    w.write('{:^10.2f}'.format(best_accuracy))
    w.write('{:^10.2f}'.format(best_tp))
    w.write('{:^10.2f}'.format(best_fp))
    w.write('{:^10.2f}'.format(best_tn))
    w.write('{:^10.2f}'.format(best_fn))
    w.write("\n\n")


def experimental_train_nsa(index):
    global DATASET
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
    best_detectors = {}

    temp_detectors = {}
    print("Starting NSA training for partition " + str(index))
    for r_value in range(1, len(training_x[0])):
        while len(temp_detectors) < MAX_DETECTORS:
            detector = generate_detector()
            for j in range(len(training_x)):
                if training_y[j] == 0 and not detector.match(r_value, training_x[j]):
                    temp_detectors[detector.get_id()] = detector

        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        for j in range(len(validation_x)):
            for key, detector in temp_detectors.items():
                matched = False
                if detector.match(r_value, validation_x[j]):
                    matched = True
                    break

                if matched:
                    if validation_y[j] == 0:
                        fp += 1
                    else:
                        tp += 1
                else:
                    if validation_y[j] == 0:
                        tn += 1
                    else:
                        fn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_r = r_value
            best_detectors = temp_detectors.copy()

    print("Finished NSA training for partition " + str(index))
    return [best_detectors, best_r]


def experimental_test_nsa(w, index, trained_nsa):
    global DATASET

    partitions_X, partitions_Y = DATASET.get_partitions()
    test_x, test_y = partitions_X[index], partitions_Y[index]

    print("Starting NSA testing on partition " + str(index))
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    temp_detectors = trained_nsa[0]
    r_value = trained_nsa[1]

    for j in range(len(test_x)):
        for key, detector in temp_detectors.items():
            matched = False
            if detector.match(r_value, test_x[j]):
                matched = True
                break

            if matched:
                if test_y[j] == 0:
                    fp += 1
                else:
                    tp += 1
            else:
                if test_y[j] == 0:
                    tn += 1
                else:
                    fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("Finished NSA training for partition " + str(index))
    w.write("Standard NSA\n")
    w.write("Optimized r-value: " + str(r_value) + "\n")
    w.write('{:^10.2f}'.format(accuracy))
    w.write('{:^10.2f}'.format(tp))
    w.write('{:^10.2f}'.format(fp))
    w.write('{:^10.2f}'.format(tn))
    w.write('{:^10.2f}'.format(fn))
    w.write("\n\n")
