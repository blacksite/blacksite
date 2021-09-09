import math
import queue
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
import random

import numpy as np
from numba import cuda

from common.detector import Detector

number_of_threads = 25


class NeuralNetworkNSA:

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=number_of_threads)
        self.futures = []
        self.lock = threading.Lock()
        self.best_detectors = {}
        self.best_accuracy = -1.0
        self.best_r_value = -1
        self.best_tp = -1.0
        self.best_fp = 0.0
        self.best_tn = 0.0
        self.best_fn = 0.0
        self.best_results = []
        self.class_detectors = {}
        self.number_of_features = 0
        self.detectors = None
        self.detector_size = 0
        self.detectors_generated = 0
        self.number_of_detectors = 0
        self.generated_detectors = None
        self.queue = queue.Queue()
        self.detector_objects = []
        self.data_set = None
        self.validator = None

    def fit(self, x, y, data_set, validator, number_of_detectors):
        # print("Evaluating")
        self.data_set = data_set
        self.validator = validator
        print("Number of Detectors: " + str(number_of_detectors))
        # print("MAD: " +str(std_dev))
        self.detectors_generated = 0
        self.number_of_features = self.data_set.get_number_of_features()
        self.detector_size = self.number_of_features * 2

        # The number of detectors is equal to the product of the number of classes - 1 (to remove benign)
        # and the number of detectors per class
        self.number_of_detectors = number_of_detectors
        self.queue.maxsize = number_of_detectors

        # Create a 1D np array of zeros of size the number of
        # features times the number of detectors * 2 for two ranges
        self.detectors = np.zeros(self.detector_size * self.number_of_detectors)
        self.generated_detectors = []

        # print("Starting detector generation")
        # Call the generate_detectors method
        self.generate_detectors()
        # print('\nFinished detector generation')

        self.validate(x, y)

    def generate_detectors(self):

        start = time.time()
        threads = list()
        for i in range(10):
            self.generate_detectors_thread()
            # x = threading.Thread(target=self.generate_detectors_thread)
            # threads.append(x)
            # x.start()
        threading.Thread(target=self.transfer_detector_values).start()
        self.queue.join()

        for index, thread in enumerate(threads):
            thread.join()

        print()
        # print('\nDetectors generated')
        end = time.time()
        runtime = end - start
        print('Runtime: ' + str(runtime))

    def generate_detectors_thread(self):

        # Create added boolean
        while True:
            classes = self.data_set.get_classes()
            rand_index = random.randint(1, len(classes)-1)
            rand_samp_type = classes[rand_index]
            seed = self.data_set.get_min_max_features_by_type(rand_samp_type)
            detector = Detector(num_features=self.number_of_features,
                                seed=seed)

            classification = self.validator[rand_samp_type].classify(detector.get_mean_value())

            if classification == 1:
                if self.detectors_generated < self.number_of_detectors:
                    self.detector_objects.append(detector)
                    self.queue.put(detector)
                    self.detectors_generated += 1
                    sys.stdout.write("\rAdded detector: " + str(self.detectors_generated))
                    sys.stdout.flush()
                else:
                    break

    def transfer_detector_values(self):
        index = 0
        while True:
            detector = self.queue.get()
            if index < self.number_of_detectors:
                values = detector.get_np_values()
                detector_index = index * self.detector_size
                for i in range(self.detector_size):
                    self.detectors[detector_index + i] = values[i]
                index += 1
                self.queue.task_done()
            else:
                break

    def validate(self, x, y):
        # print('Validating')
        number_of_samples = x.shape[0]

        # If the the shapes fo x and y don't match, return
        if number_of_samples != y.shape[0]:
            print("Shape of x does not match shape of y")
            os._exit(0)

        # Transpose the x samples into a row-major 1D array of float32
        samples = x.ravel()
        samples.astype("f4")

        # Create the labels np array of size number of samples
        labels = y

        # For each label, convert the list into a float32 and add to labels
        number_of_detectors = len(self.detectors) / self.detector_size

        maxes = np.zeros(int(number_of_samples), dtype='i4')

        # print("Starting max calculation")
        calculate_maximums(self.detectors, samples, maxes, number_of_detectors, number_of_samples,
                           self.number_of_features)

        self.validate_r_values(maxes, number_of_samples, labels)

        # print("Detectors Validated")

    def validate_r_values(self, maxes, number_of_samples, labels):
        global number_of_threads

        self.futures = []

        results = {}
        for r_value in range(self.number_of_features):
            results[r_value] = [0, 0, 0, 0]

        for r_value in range(1, self.number_of_features):

            for sample_index in range(int(math.sqrt(number_of_threads))):
                # self.validate_r_value_kernel(maxes, results, r_value, sample_index, number_of_samples, labels, classes)
                self.futures.append(self.executor.submit(self.validate_r_value_kernel, maxes, results, r_value,
                                                         sample_index, number_of_samples, labels))

        wait(self.futures, return_when='ALL_COMPLETED')

        for r_value in range(1, self.number_of_features):
            # print(str(r_value))

            tp = results[r_value][0]
            fn = results[r_value][1]
            tn = results[r_value][2]
            fp = results[r_value][3]

            accuracy = float((tn + tp) / float(tn + tp + fn + fp))

            if accuracy > self.best_accuracy:
                self.best_tp = tp
                self.best_accuracy = accuracy
                self.best_r_value = r_value

    def validate_r_value_kernel(self, maxes, results, r_value, sample_index, number_of_samples, labels):

        for m in range(int(number_of_samples)):

            if sample_index < number_of_samples:
                classification = int(labels[sample_index])
                self.lock.acquire()
                if maxes[sample_index] > r_value:
                    # False Positive
                    if classification == 0:
                        results[r_value][3] += 1
                    # True Positive
                    else:
                        results[r_value][0] += 1
                else:
                    # True Negative
                    if classification == 0:
                        results[r_value][2] += 1
                    # False Negative
                    else:
                        results[r_value][1] += 1
                self.lock.release()

                sample_index += int(math.sqrt(number_of_threads))
            else:
                break

    def predict(self, x, y):
        # print('Predicting')
        number_of_samples = x.shape[0]

        # If the the shapes fo x and y don't match, return
        if number_of_samples != y.shape[0]:
            print("Shape of x does not match shape of y")
            os._exit(0)

        # Transpose the x samples into a row-major 1D array of float32
        samples = x.ravel()
        samples.astype("f4")

        labels = y

        number_of_detectors = len(self.detectors) / self.detector_size

        maxes = np.zeros(int(number_of_samples), dtype='i4')

        calculate_maximums(self.detectors, samples, maxes, number_of_detectors, number_of_samples,
                           self.number_of_features)

        self.best_accuracy = -1.0
        self.best_results = {}

        self.test_r_value(maxes, number_of_samples, labels)

        return self.best_results

    def test_r_value(self, maxes, number_of_samples, labels):
        global number_of_threads

        self.futures = []

        for sample_index in range(int(math.sqrt(number_of_threads))):
            self.futures.append(self.executor.submit(self.test_r_value_kernel, maxes, self.best_r_value, sample_index, number_of_samples, labels))

        wait(self.futures, return_when='ALL_COMPLETED')

    def test_r_value_kernel(self, maxes, r_value, sample_index, number_of_samples, labels):
        global number_of_threads

        for m in range(int(number_of_samples)):

            if sample_index < number_of_samples:
                encoded_label = int(labels[sample_index])  # list(labels[col]).index(max(list(labels[col])))
                original_class = self.data_set.get_classes()[encoded_label]

                self.lock.acquire()
                if original_class not in self.best_results:
                    self.best_results[original_class] = [0, 0]
                if maxes[sample_index] > r_value:
                    if encoded_label == 0:
                        # False Positive
                        self.best_results[original_class][0] += 1
                    else:
                        # True Positive
                        self.best_results[original_class][0] += 1
                else:
                    if encoded_label == 0:
                        # True Negative
                        self.best_results[original_class][1] += 1
                    else:
                        # False Negative
                        self.best_results[original_class][1] += 1
                self.lock.release()

                sample_index += int(math.sqrt(number_of_threads))
            else:
                break

    def teardown(self):
        del self.detectors
        del self.detector_objects
        self.executor.shutdown()


def calculate_maximums(detectors, samples, maxes, number_of_detectors, number_of_samples, number_of_features):
    #  print("Starting maximum calc in function")
    threads_per_block = (number_of_detectors, number_of_samples)
    blocks_per_grid = (1, 1)
    if number_of_detectors * number_of_samples > 512:
        threads_per_block = (32, 32)

    blocks_per_grid = (math.ceil(float(number_of_detectors / float(threads_per_block[0]))),
                       math.ceil(float(number_of_threads) / float(threads_per_block[1])))

    if blocks_per_grid[0] > 65535 or blocks_per_grid[1] > 65535:
        blocks_per_grid = (65535, 65535)

    calculate_maximums_kernel[blocks_per_grid, threads_per_block](detectors, samples, maxes, number_of_features)
    cuda.synchronize()
    # print("Submitted works")


@cuda.jit
def calculate_maximums_kernel(d_detectors, d_samples, d_maxes, number_of_features):
    detector_index = int(cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x)
    original_sample_index = int(cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y)

    size_of_detector = int(number_of_features * 2)
    size_of_sample = int(number_of_features)

    number_of_detectors = int(len(d_detectors) / size_of_detector)
    number_of_samples = int(len(d_samples) / size_of_sample)

    for n in range(number_of_detectors):

        if detector_index >= number_of_detectors:
            return

        sample_index = original_sample_index

        for m in range(number_of_samples):

            if sample_index < number_of_samples:

                feature_range_index = 0
                count = 0

                for i in range(number_of_features):
                    min_val = d_detectors[size_of_detector * detector_index + feature_range_index]
                    max_val = d_detectors[size_of_detector * detector_index + feature_range_index + 1]

                    samp = d_samples[size_of_sample * sample_index + i]

                    if min_val <= samp and max_val >= samp and max_val != 0.0:
                        count += 1

                    feature_range_index += 2

                cuda.atomic.max(d_maxes, sample_index, count)

                sample_index += int(cuda.gridDim.y * cuda.blockDim.y)
            else:
                break

        detector_index += int(cuda.gridDim.x * cuda.blockDim.x)


def calculate_maximums_loop(detectors, samples, maxes, number_of_features):
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=number_of_threads)
    futures = []
    for detector_index in range(int(math.sqrt(number_of_threads))):
        for sample_index in range(int(math.sqrt(number_of_threads))):
            # calculate_maximums_kernel_loop(detectors, samples, maxes, row, col, number_of_features, lock)
            futures.append(
                executor.submit(calculate_maximums_kernel_loop, detectors, samples, maxes, detector_index, sample_index,
                                number_of_features, lock))

    wait(futures, return_when='ALL_COMPLETED')


# @jit(nopython='True')
def calculate_maximums_kernel_loop(d_detectors, d_samples, d_maxes, detector_index, sample_index, number_of_features,
                                   lock):
    original_sample_index = sample_index
    size_of_detector = int(number_of_features * 2)
    size_of_sample = int(number_of_features)

    number_of_detectors = int(len(d_detectors) / size_of_detector)
    number_of_samples = int(len(d_samples) / size_of_sample)

    for n in range(number_of_detectors):

        if detector_index >= number_of_detectors:
            return

        for m in range(number_of_samples):

            if sample_index < number_of_samples:
                feature_range_index = 0
                count = 0

                for i in range(number_of_features):
                    min_val = d_detectors[size_of_detector * detector_index + feature_range_index]
                    max_val = d_detectors[size_of_detector * detector_index + feature_range_index + 1]

                    samp = d_samples[size_of_sample * sample_index + i]

                    if min_val <= samp and max_val >= samp and max_val != 0.0:
                        count += 1

                    feature_range_index += 2
                lock.acquire()
                if count > d_maxes[sample_index]:
                    d_maxes[sample_index] = count
                lock.release()

                sample_index += int(math.sqrt(number_of_threads))
            else:
                break

        sample_index = original_sample_index
        detector_index += int(math.sqrt(number_of_threads))


def progressBar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent / 100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
