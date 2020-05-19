from database import MongoDBConnect

DB = MongoDBConnect()
DNN = ''
CURRENT_DETECTORS = ''


def get_detectors():
    print('Getting detectors')


def train_detectors():
    print('Training detectors')


def train_dnn():
    print('Training DNN')


def test_dnn():
    print('Testing DNN')


def print_dnn_results():
    # Print DNN results
    print(CURRENT_DETECTORS)


def save_dnn():
    print('Saving DNN')


def read_dnn():
    print('Reading DNN')


def delete_dnn():
    print('Deleting DNN')


if __name__ == "__main__":
    option = input('0: Read from file\n1: Create & train new Deep Neural Network\n')

    while True:
        if option == 0:
            filename = input('Enter filename')
            read_dnn()
            break
        elif option == 1:
            train_dnn()
            test = input('0: Test DNN\n1: Do not test DNN\n')
            while True:
                if test == 0:
                    test_dnn()
                elif test == 1:
                    break
                else:
                    print('Invalid input')
                    test = input('0: Test DNN\n1: Do not test DNN\n')
            break
        else:
            print('Invalid input')
            option = input('0: Read from file\n1: Create & train new Deep Neural Network\n')

    train_detectors()


