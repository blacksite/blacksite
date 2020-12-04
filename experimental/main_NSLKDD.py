from experimental import grizzly, panda
import sys
from cache import dataset_NSLKDD as dataset, detectors, new_instances, suspicious_instances, validated_instances
from os import path
import os
import logging

SAVE_FILE = None


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

    w_dataset = open(result_directory + "/dataset.csv", "w")
    dataset.read_from_file(w_dataset)

    w_dnn = open(result_directory + "/dnn.csv", "w")
    w_dnn.write("{:^30s}".format("Accuracy"))
    for i in range(len(dataset.CLASSES)):
        w_dnn.write(
            ",{:^30s},{:^30s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
    w_dnn.write("\n")
    w_dnn.flush()
    grizzly.set_common(detectors, validated_instances, suspicious_instances, dataset)
    panda.set_common(detectors, new_instances, suspicious_instances, dataset)

    std_devs = [1, 2, 3]
    num_detectors = [1000, 2500, 5000, 10000]
    writers = {}

    # for s in std_devs:
    for n in num_detectors:
        writers[n] = open(result_directory + "/dnn-detectors-" + str(n) + ".csv", "w")

        writers[n].write("{:^30s}, {:^30s}".format("r-value", "Accuracy"))
        for i in range(len(dataset.CLASSES)):
            writers[n].write(
                    ",{:^30s},{:^30s}".format(dataset.CLASSES[i] + "-correct", dataset.CLASSES[i] + "-not-correct"))
        writers[n].write("\n")
        writers[n].flush()

    try:
        # if i == 0:
        #     grizzly.load_dnn("../model/" + save_file + ".dnn")
        #     grizzly.experimental_train_dnn(w_dnn, i, False)
        # else:
        grizzly.experimental_train_dnn(w_dnn, 0)

        # for s in std_devs:
        for n in num_detectors:
            panda.evaluate_dnn(writers[n], grizzly, 0, n)
            # panda.evaluate_nsa(w_detectors_nsa, i)

    except Exception as e:
        logging.error(e)
        grizzly.save_dnn(save_file)
        sys.exit(-1)


if __name__ == "__main__":
    start_experiment()
