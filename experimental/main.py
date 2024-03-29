from experimental import grizzly, panda
import sys
from cache import dataset, detectors, new_instances, suspicious_instances, validated_instances
from os import path
import os
import logging
import time

SAVE_FILE = None


def start_dataset():
    file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
    while True:
        if file_prompt == 'y':
            filename = input('\nEnter the csv filename\n'
                            '**If there are multiple files, separate with a comma (no spaces)**\n')
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


def start_experiment():
    start = time.time()

    save_file = input("Please enter the file name to save the results\n")

    while True:
        if path.exists("results/" + save_file + ".csv"):
            option = input("That file exists, would you like to overwrite it: y/n\n")
            if option == "y":
                break
            else:
                save_file = input("Please enter the file name to save the results\n")
        else:
            break

    result_directory = '../results/' + save_file
    try:
        os.mkdir(result_directory)
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")

    filename = '../data/Day1.csv,../data/Day2.csv,../data/Day3.csv,../data/Day4.csv,../data/Day5.csv,' \
               '../data/Day8.csv,../data/Day9.csv,../data/Day10.csv'
    # filename = '../data/Day2.csv,../data/Day3.csv,../data/Day4.csv,../data/Day5.csv'
    filename = '../data'
    # filename = '../data/test.csv'
    w_dataset = open(result_directory + "/dataset.csv", "w")
    dataset.read_from_mqtt_file(w_dataset, filename)

    w_dnn = open(result_directory + "/dnn.csv", "w")
    w_dnn.write("{:^40s}".format("Accuracy"))
    for i in range(len(dataset.CLASSES)):
        w_dnn.write(
            ",{:^40s},{:^40s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
    w_dnn.write("\n")
    w_dnn.flush()
    grizzly.set_common(detectors, validated_instances, suspicious_instances, dataset)
    panda.set_common(detectors, new_instances, suspicious_instances, dataset)

    std_devs = [1]
    num_detectors = [10000, 25000, 50000, 100000]
    writers = {}

    # for s in std_devs:
    #     for n in num_detectors:
    #         writers[s+n] = open(result_directory + "/dnn-detectors-" + str(s) + "-" + str(n) + ".csv", "w")
    #
    #         writers[s+n].write("{:^40s}".format("Accuracy"))
    #         writers[s + n].write(",{:^40s}".format("r-value"))
    #         for c in dataset.CLASSES:
    #             writers[s+n].write(
    #                     ",{:^40s},{:^40s}".format(c + "-correct", c + "-not-correct"))
    #         writers[s+n].write("\n")
    #         writers[s+n].flush()
    for key, value in dataset.PARTITION_X.items():
        w_dnn = open(result_directory + "/" + key + "-dnn.csv", "w")
        w_dnn.write("{:^40s}".format("Accuracy"))
        for i in range(len(dataset.CLASSES)):
            w_dnn.write(
                ",{:^40s},{:^40s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
        w_dnn.write("\n")
        w_dnn.flush()
        if key != 'Benign':
            for n in num_detectors:
                writers[key + ' ' + n] = open(result_directory + "/dnn-detectors-" + key + "-" + str(n) + ".csv", "w")

                writers[key + ' ' + n].write("{:^40s}".format("Accuracy"))
                writers[key + ' ' + n].write(",{:^40s}".format("r-value"))
                for c in dataset.CLASSES:
                    writers[key + ' ' + n].write(
                        ",{:^40s},{:^40s}".format(c + "-correct", c + "-not-correct"))
                writers[key + ' ' + n].write("\n")
                writers[key + ' ' + n].flush()

        for i in range(dataset.KFOLDS):

            try:
                grizzly.experimental_train_dnn(w_dnn, i)
                grizzly.evaluate_ind_dnn(w_dnn, i, key)
                panda.initialize_model()

                # for s in std_devs:
                #     for n in num_detectors:
                #         panda.evaluate_dnn(writers[s+n], grizzly, i, n, s)

                for n in num_detectors:
                    panda.evaluate_dnn(writers[n], grizzly, i, n, None)
                            # panda.evaluate_nsa(w_detectors_nsa, i)

            except Exception as e:
                logging.error(e)
                # grizzly.save_dnn(save_file)
                sys.exit(-1)

    end = time.time() - start

    print('Total runtime: ' + str(end) + 's')


def start_ind_experiment():
    start = time.time()

    save_file = input("Please enter the file name to save the results\n")

    while True:
        if path.exists("results/" + save_file + ".csv"):
            option = input("That file exists, would you like to overwrite it: y/n\n")
            if option == "y":
                break
            else:
                save_file = input("Please enter the file name to save the results\n")
        else:
            break

    result_directory = '../results/' + save_file
    try:
        os.mkdir(result_directory)
    except OSError:
        print("Creation of the directory failed")
    else:
        print("Successfully created the directory")

    filename = '../data/Day1.csv,../data/Day2.csv,../data/Day3.csv,../data/Day4.csv,../data/Day5.csv,' \
               '../data/Day8.csv,../data/Day9.csv,../data/Day10.csv'
    # filename = '../data/Day2.csv,../data/Day3.csv,../data/Day4.csv,../data/Day5.csv'
    # filename = '../data/Day8.csv,../data/Day9.csv'
    filename = '../data'
    # filename = '../data/test.csv'
    w_dataset = open(result_directory + "/dataset.csv", "w")
    dataset.read_from_mqtt_file(w_dataset, filename)

    grizzly.set_common(detectors, validated_instances, suspicious_instances, dataset)
    panda.set_common(detectors, new_instances, suspicious_instances, dataset)

    num_detectors = [100, 250, 500, 1000]

    for key, value in dataset.PARTITION_X.items():
        w_dnn = open(result_directory + "/" + key + "-dnn.csv", "w")
        w_dnn.write("{:^10s}".format("Accuracy"))
        w_dnn.write(",{:^10s},{:^10s},{:^10s},{:^10s}".format("TP", "FP", "TN", "FN"))
        w_dnn.write("\n")
        w_dnn.flush()

        writers = {}
        for n in num_detectors:
            writers[key + ' ' + str(n)] = open(result_directory + "/dnn-detectors-" + key + "-" + str(n) + ".csv", "w")
            writers[key + ' ' + str(n)].write(",{:^10s}".format("r-value"))
            writers[key + ' ' + str(n)].write("{:^10s}".format("Accuracy"))
            writers[key + ' ' + str(n)].write(",{:^10s},{:^10s},{:^10s},{:^10s}".format("TP", "FP", "TN", "FN"))
            writers[key + ' ' + str(n)].write("\n")
            writers[key + ' ' + str(n)].flush()

        for i in range(dataset.KFOLDS):

            try:
                grizzly.evaluate_ind_dnn(w_dnn, i, key)
                panda.initialize_model()

                for n in num_detectors:
                    panda.evaluate_ind_dnn(writers[key + ' ' + str(n)], grizzly, i, key, n)

            except Exception as e:
                logging.error(e)
                sys.exit(-1)

    end = time.time() - start

    print('Total runtime: ' + str(end) + 's')


if __name__ == "__main__":
    start_ind_experiment()
