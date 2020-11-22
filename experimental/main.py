from experimental import grizzly, panda
import sys
from cache import dataset, detectors, new_instances, suspicious_instances, validated_instances
from os import path
import os
import logging

SAVE_FILE = None


def start_dataset():
    file_prompt = input('Would you like to use a dataset from a csv file?: y/n\n')
    while True:
        if file_prompt == 'y':
            # filename = input('\nEnter the csv filename\n'
            #                 '**If there are multiple files, separate with a comma (no spaces)**\n')
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

    # filename = '../data/Day1.csv,../data/Day2.csv'
    filename = '../data/test.csv'
    # filename = '../data/Day1.csv'
    w_dataset = open(result_directory + "/dataset.csv", "w")
    dataset.read_from_file(w_dataset, filename)

    w_dnn = open(result_directory + "/dnn.csv", "w")
    w_dnn.write("{:^30s}".format("Accuracy"))
    for i in range(len(dataset.CLASSES)):
        w_dnn.write(
            ",{:^30s},{:^30s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
    w_dnn.write("\n")
    w_dnn.flush()
    grizzly.set_common(detectors, validated_instances, suspicious_instances, dataset)

    # Writer for DNN detectors
    w_detectors_dnn = open(result_directory + "/dnn-detectors.csv", "w")
    w_detectors_dnn.write("{:^30s}, {:^30s}".format("r-value", "Accuracy"))
    for i in range(len(dataset.CLASSES)):
        w_detectors_dnn.write(
            ",{:^30s},{:^30s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
    w_detectors_dnn.write("\n")
    w_detectors_dnn.flush()

    # Writer for NSA detectors
    w_detectors_nsa = open(result_directory + "/nsa-detectors.csv", "w")
    w_detectors_nsa.write("{:^30s}, {:^30s}".format("r-value", "Accuracy"))
    for i in range(len(dataset.CLASSES)):
        w_detectors_nsa.write(
            ",{:^30s},{:^30s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
    w_detectors_nsa.write("\n")
    w_detectors_nsa.flush()
    panda.set_common(detectors, new_instances, suspicious_instances, dataset)
    for i in range(dataset.KFOLDS):

        try:
            # if i == 0:
            #     grizzly.load_dnn("../model/" + save_file + ".dnn")
            #     grizzly.experimental_train_dnn(w_dnn, i, False)
            # else:
            grizzly.experimental_train_dnn(w_dnn, i)

            trained_dnn_detectors = panda.experimental_train_dnn_detectors(grizzly, i)
            panda.experimental_test_dnn_detectors(w_detectors_dnn, i, trained_dnn_detectors)
            trained_nsa = panda.experimental_train_nsa(i)
            panda.experimental_test_nsa(w_detectors_nsa, i, trained_nsa)
        except Exception as e:
            logging.error(e)
            grizzly.save_dnn(save_file)
            sys.exit(-1)


if __name__ == "__main__":
    start_experiment()
