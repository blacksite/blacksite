from numba import cuda, guvectorize, float32
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, wait
import threading
import sys
from common.detector import Detector


class NeuralNetworkNSA:

    def __init__(self):
        self.executor = None
        self.futures = []
        self.lock = threading.Lock()
        self.best_detectors = {}
        self.best_accuracy = -1.0
        self.best_r = -1
        self.best_tp = 0.0
        self.best_fp = 0.0
        self.best_tn = 0.0
        self.best_fn = 0.0
        self.best_results = []
        self.class_detectors = {}
        self.number_of_features = 0
        self.detectors = None
        self.detector_size = 0

    def fit(self, number_of_features, grizzly, seeds, number_of_detectors_per_class=1000):
        self.number_of_features = number_of_features
        self.detector_size = self.number_of_features * 2

        # The number of detectors is equal to the product of the number of classes - 1 (to remove benign)
        # and the number of detectors per class
        number_of_detectors = number_of_detectors_per_class * (len(seeds))

        # Create a 1D np array of zeros of size the number of
        # features times the number of detectors * 2 for two ranges
        self.detectors = np.zeros(self.number_of_features * number_of_detectors * 2)

        # Call the generate_detectors method
        self.generate_detectors(number_of_detectors, grizzly, seeds, number_of_detectors_per_class)
        print('Finished detector generation')

    def generate_detectors(self, number_of_detectors, grizzly, seeds, number_of_detectors_per_class):
        self.executor = ThreadPoolExecutor(max_workers=number_of_detectors)

        index = 0
        # for each key value pair in the max_features dictionary
        # run number of detectors per class number of threads to generate a single detector
        for key, value in seeds.items():
            for j in range(number_of_detectors_per_class):
                self.futures.append(self.executor.submit(self.generate_detectors_thread, grizzly, index, value))
                index += 1
        wait(self.futures, return_when='ALL_COMPLETED')

    # Generate a single detector thread
    # Continues to loop until a propective detector passes the grizzly check
    # If the detector passes, iterate through the detector values and add them to detectors np.array
    # The index is calculated by getting the index of this detector in the overall detectors
    # Then multiplying that index by size of each detector
    # Then adding the index of the current feature range value
    def generate_detectors_thread(self, grizzly, index, seed):

        # Create added boolean
        added = False
        while not added:
            detector = Detector(num_features=self.number_of_features, seed=seed)
            self.lock.acquire()
            classification = grizzly.classify(detector.get_mean_value())
            self.lock.release()

            if classification != 0:
                values = detector.get_np_values()
                detector_index = index * self.detector_size
                for i in range(self.detector_size):
                    self.detectors[detector_index + i] = values[i]
                added = True

    def predict(self, x, y, classes):
        number_of_samples = x.shape[0]
        sample_size = self.number_of_features
        labels = np.zeros(number_of_samples, dtype="f4")

        # If the the shapes fo x and y don't match, return
        if number_of_samples != y.shape[0]:
            print("Shape of x does not match shape of y")
            sys.exit(0)

        # Transpose the x samples into a row-major 1D array of float32
        samples = x.ravel()
        samples.astype("f4")

        # Create the labels np array of size number of samples
        labels = np.zeros(number_of_samples, dtype="f4")

        # For each label, convert the list into a float32 and add to labels
        for i in range(len(y)):
            labels[i] = list(y[i]).index(max(list(y[i])))

        self.test_dnn_detectors(samples, number_of_samples, labels, classes)

        return self.best_results

    def test_dnn_detectors(self, samples, number_of_samples, labels, classes):

        number_of_detectors = len(self.detectors) / self.detector_size

        maxes = np.zeros(int(number_of_samples), dtype='f')

        d_detectors = cuda.to_device(self.detectors)
        d_samples = cuda.to_device(samples)
        d_maxes = cuda.to_device(maxes)
        print("Starting max calculation")
        self.calculate_maximums(d_detectors, d_samples, d_maxes, number_of_detectors, number_of_samples)
        d_maxes.to_host()
        print("Finished max calculation")

        print("Starting r-value evaluation")

        for r_value in range(1, 75):
            self.futures.append(self.executor.submit(self.test_rv_value, maxes, r_value, number_of_samples, labels, classes))

        wait(self.futures, return_when='ALL_COMPLETED')

        print("Finished r-value evaluation")

    def calculate_maximums(self, detectors, samples, maxes, number_of_detectors, number_of_samples):

        threads_per_block = (number_of_detectors, number_of_samples)
        blocks_per_grid = (1, 1)
        if number_of_detectors * number_of_samples > 512:
            threads_per_block =(32,32)
            blocks_per_grid = (math.ceil(float(number_of_detectors / float(threads_per_block[0]))),
                                math.ceil(float(number_of_samples) / float(threads_per_block[1])))

        self.calculate_maximums_kernel[blocks_per_grid, threads_per_block](detectors, samples, maxes)

    @cuda.jit
    def calculate_maximums_kernel(self, d_detectors, d_samples, d_maxes):
        row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        index = col

        size_of_detector = 75 * 2
        size_of_sample = 75

        number_of_samples = len(d_samples) / size_of_sample

        if row < number_of_samples and col < number_of_samples:

            detectorIndex = 0
            count = 0.0

            for i in range(75):
                min = d_detectors[size_of_detector * row + detectorIndex]
                max = d_detectors[size_of_detector * row + detectorIndex + 1]

                samp = d_samples[size_of_sample * col + i]

                if min <= samp and max >= samp:
                    count += 1

                detectorIndex += 2

            cuda.atomic.max(d_maxes, index, count)

    def test_rv_value(self, maxes, r_value, number_of_samples, labels, classes):
        results = {"Accuracy":-1}

        for i in range(len(classes)):
            results[classes[i]] = [0, 0]

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for col in range(int(number_of_samples)):
            match = False

            if maxes[col] > r_value:
                match = True
                break

            sample_class = int(labels[col])# list(labels[col]).index(max(list(labels[col])))
            if match:
                if sample_class == 0:
                    results[classes[sample_class]][1] += 1
                    fp += 1
                else:
                    tp += 1
                    results[classes[sample_class]][0] += 1
            else:
                if labels[col] == 0:
                    tn += 1
                    results[classes[sample_class]][0] += 1
                else:
                    fn += 1
                    results[classes[sample_class]][1] += 1

        accuracy = float(tp + tn) / (tp + tn + fp + fn)

        self.lock.acquire()
        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy
            self.best_r = r_value
            results["Accuracy"] = accuracy
            self.best_results = results
        self.lock.release()
