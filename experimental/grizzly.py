from keras.models import Sequential
from keras.layers import Dense
import keras.metrics as km
import numpy as np
from tensorflow import keras
from os import path
import threading


# Disable GPU otimization
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Class variables
DNN_LOCK = threading.Lock()
DNN_TRAINING_THRESHOLD = 30
FILENAME = 'deepneuralnetwork.blk'
ACCURACY_THRESHOLD = 0.8
BATCH_SIZE = 80
BEST_ACCURACY = 0.0
DETECTORS = None
VALIDATED_INSTANCES = None
SUSPICIOUS_INSTANCES = None
DATASET = None
DNN = None
CURRENT_DATASET_SIZE = None


def set_common(detectors, validated_instances, suspicious_instances, dataset):
    global DETECTORS
    global VALIDATED_INSTANCES
    global SUSPICIOUS_INSTANCES
    global DATASET
    global CURRENT_DATASET_SIZE

    DETECTORS = detectors
    VALIDATED_INSTANCES = validated_instances
    SUSPICIOUS_INSTANCES = suspicious_instances
    DATASET = dataset
    CURRENT_DATASET_SIZE = DATASET.size()


def define_model_multiclass():
    global DATASET

    # create and fit the DNN network
    model = Sequential()
    model.add(Dense(70, input_dim=DATASET.get_number_of_features(), activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(DATASET.NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=["accuracy"])
    return model


def define_model():
    # create and fit the DNN network
    model = Sequential()
    model.add(Dense(70, input_dim=DATASET.get_number_of_features(), activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=["accuracy", km.TruePositives(), km.FalsePositives(), km.TrueNegatives(),
                           km.FalseNegatives()])
    return model


# Used for experimental purposes
# Provides functionality for saving results
def experimental_train_dnn(w, index=None, fit=True):
    global NUM_BENIGN_INSTANCES
    global NUM_MALICIOUS_INSTANCES
    global RAW_PARTITION_SIZES
    global BEST_ACCURACY
    global DNN
    global DATASET


    partitions_X, partitions_Y = DATASET.get_partitions()

    print("Starting dnn training for partition " +str(index))

    # for batch training
    # batches = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # for b in batches:
    #     estimator = KerasClassifier(build_fn=define_model, epochs=100, batch_size=b, verbose=2)
    #     kfold = KFold(n_splits=10, shuffle=True)
    #     results = cross_val_score(estimator, np.array(training_instances), np.array(training_labels), cv=kfold)
    #     print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #
    #     w.write("{:s}".format(str(b)))
    #     for a in results:
    #         w.write(',{:s}'.format(str(a)))
    #     w.write('\n')
    #     w.flush()

    # for individual deep neural network
    # for i in range(DATASET.KFOLDS):
    #     # create training and testing x and y datasets from the kFolds
    #     training_x, training_y = [], []
    #     test_x, test_y = partitions_X[i], partitions_Y[i]
    #
    #     for x in range(DATASET.KFOLDS):
    #         if x == i:
    #             continue
    #
    #         training_x.extend(partitions_X[i])
    #         training_y.extend(partitions_Y[i])
    #
    #     # begin training the model
    #     model = define_model()
    #     model.fit(np.array(training_x), np.array(training_y), BATCH_SIZE, epochs=100, verbose=2)
    #     results = model.evaluate(np.array(test_x), np.array(test_y))
    #     w.write('{:^10.2f}'.format(results[0]))
    #     w.write(',{:^10.2f}'.format(results[1] * 100.0))
    #     for x in range(2, len(results)):
    #         w.write(',{:^10.0f}'.format(results[x]))
    #     w.write('\n')
    #     w.flush()
    #
    #     # Determine if current model has better accuracy than previous
    #     # If so, set DNN equal to current model
    #     if results[1] > BEST_ACCURACY:
    #         BEST_ACCURACY = results[1]
    #         DNN = model

    # create training and testing x and y datasets from the kFolds
    training_x, training_y = [], []
    test_x, test_y = partitions_X[index], partitions_Y[index]

    for x in range(DATASET.KFOLDS):
        if x == index:
            continue

        training_x.extend(partitions_X[index])
        training_y.extend(partitions_Y[index])

    # begin training the model
    model = define_model_multiclass()
    if fit:
        model.fit(np.array(training_x), np.array(training_y), BATCH_SIZE, epochs=100, verbose=2)
        prediction = model.predict(np.array(test_x), batch_size=BATCH_SIZE)
        results = calculate_results(prediction, test_y)
    else:
        results = DNN.evaluate(np.array(test_x), np.array(test_y))
    w.write('{:^30.2f}'.format(results[0] * 100.0))
    for x in range(1, len(results)):
        w.write(',{:^30.0f}'.format(results[x]))
    w.write('\n')
    w.flush()

    DNN = model

    print("Finished dnn training " + str(index))


def calculate_results(p, y):
    global DATASET
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    temp = {}

    for i in range(len(DATASET.CLASSES)):
        temp[DATASET.CLASSES[i]] = [0, 0]

    for i in range(len(p)):
        prediction = list(p[i]).index(max(list(p[i])))
        actual = list(y[i]).index(max(list(y[i])))

        if actual == 0:
            if prediction == actual:
                temp[DATASET.CLASSES[actual]][0] += 1
                tn += 1
            else:
                temp[DATASET.CLASSES[actual]][1] += 1
                fp += 1
        else:
            if prediction == actual or prediction != 0:
                temp[DATASET.CLASSES[actual]][0] += 1
                tp += 1
            else:
                temp[DATASET.CLASSES[actual]][1] += 1
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    results = [accuracy]

    for i in range(len(DATASET.CLASSES)):
        results.extend(temp[DATASET.CLASSES[i]])

    return results


def save_dnn(filename):
    global DNN
    DNN.save("../model/" + filename + ".dnn")


def load_dnn(filename):
    global DNN

    while True:
        if not path.exists(filename):
            filename = input("The entered file does not exist. Please re-enter a file name\n")
        else:
            break

    DNN_LOCK.acquire()
    DNN = keras.models.load_model(filename)
    DNN_LOCK.release()


def classify(value):
    global DNN
    global DNN_LOCK

    if DNN:

            DNN_LOCK.acquire()
            classification = DNN.predict(np.array([value]))
            DNN_LOCK.release()

            return list(classification[0]).index(max(list(classification[0])))

    else:
        print("No DNN available")
        exit(-1)