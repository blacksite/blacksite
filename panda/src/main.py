from database import MongoDBConnect
import random
import time
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize

MAX_DETECTORS = 1000
DETECTOR_LENGTH = 32
INITIAL_DETECTOR_LIFESPAN = 3600
IMMATURE_DETECTOR_LIFESPAN = 604800
MATURE_DETECTOR_LIFESPAN = 2592000
MEMORY_DETECTOR_LIFESPAN = 31536000
CURRENT_DETECTORS = ""

DB = MongoDBConnect()

def generate_detector():
    # Variable to store the
    # string
    value = ""

    # Loop to find the string
    # of desired length
    for i in range(DETECTOR_LENGTH):
        # randint function to generate
        # 0, 1 randomly and converting
        # the result into str
        temp = str(random.randint(0, 1))

        # Concatenatin the random 0, 1
        # to the final result
        value += temp

    detector = {"VALUE": value, "TYPE": "INITIAL", 'LIFE': time.time()}

    return detector


def remove_detector(detector):
    # print(detector['TYPE'] + ' detector removed')
    DB.delete_detector(detector)
    del CURRENT_DETECTORS[detector['_id']]
    add_detector()


def add_detector():
    temp_detector = generate_detector()
    d = DB.add_detector(temp_detector)
    temp_detector['_id'] = d.inserted_id
    CURRENT_DETECTORS[d.inserted_id] = temp_detector
    # print('Detector added')


def generate_initial_detectors():
    num_detectors = len(CURRENT_DETECTORS)
    while (num_detectors < MAX_DETECTORS):
        add_detector()
        num_detectors = len(CURRENT_DETECTORS)
        print('{:.2f}%'.format(float(num_detectors)/MAX_DETECTORS*100))


def evaluate_lifespans():
    num_detectors = len(CURRENT_DETECTORS)
    while (num_detectors < MAX_DETECTORS):
        add_detector()
        num_detectors = len(CURRENT_DETECTORS)

    keys = []

    for k in CURRENT_DETECTORS.keys():
        keys.append(k)

    sum = 0
    for key in keys:
        value = CURRENT_DETECTORS[key]
        lifetime = time.time() - value['LIFE']
        if value['TYPE'] == 'INITIAL':
            if lifetime > INITIAL_DETECTOR_LIFESPAN:
                remove_detector(value)
                sum += 1
        elif value['TYPE'] == 'IMMATURE':
            if lifetime > IMMATURE_DETECTOR_LIFESPAN:
                remove_detector(value)
                sum += 1
        elif value['TYPE'] == 'MATURE':
            if lifetime > MATURE_DETECTOR_LIFESPAN:
                remove_detector(value)
                sum += 1
        elif value['TYPE'] == 'MEMORY':
            if lifetime > MEMORY_DETECTOR_LIFESPAN:
                remove_detector(value)
                sum += 1

    if (sum > 0):
        print('{:d} detectors were replaced'.format(sum))


if __name__ == "__main__":
    CURRENT_DETECTORS = DB.get_all()

    print('Generating initial detectors')
    generate_initial_detectors()
    print('Finished generating initial detectors\n')

    print('Detector lifespan evaluation started')
    while True:
        evaluate_lifespans()


