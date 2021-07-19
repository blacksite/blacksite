import bson
import random
import time
import uuid
import numpy as np


class Detector:

    def __init__(self, num_features=None, _id=None, value=None, type=None, life=None, seed=None, std_dev=1):
        self._id = _id
        self.VALUE = value
        self.TYPE = type
        self.LIFE = life
        self.NUM_FEATURES = num_features
        self.SEED = seed
        self.single_value = []
        self.generate_median_mad(4)

    def generate_random(self):
        value = []

        for i in range(self.NUM_FEATURES):
            max_val = self.SEED[i][1]
            if max_val != 0.0:
                num1 = random.uniform(0, 1)
                num2 = random.uniform(0, 1)

                if num1 > num2:
                    temp = num1
                    num1 = num2
                    num2 = temp

                value.append([num1, num2])
            else:
                value.append([0.0, 0.0])

        self.VALUE = value
        self.LIFE = time.time()
        self.set_initial()
        self._id = uuid.uuid4()

    def generate_median_mad(self, std_dev):
        value = []

        for i in range(self.NUM_FEATURES):
            min_val = max(0.0, self.SEED[i][0] - (self.SEED[i][1] * std_dev))
            max_val = min(1.0, self.SEED[i][0] + (self.SEED[i][1] * std_dev))
            if max_val != 0.0:
                num1 = random.uniform(min_val, max_val)
                num2 = random.uniform(min_val, max_val)

                if num1 < num2:
                    value.append([num1, num2])
                else:
                    value.append([num2, num1])
            else:
                value.append([0.0, 0.0])

        self.VALUE = value
        self.LIFE = time.time()
        self.set_initial()
        self._id = uuid.uuid4()

    def generate_min_max(self):
        value = []

        for i in range(self.NUM_FEATURES):
            min_val = self.SEED[i][0]
            max_val = self.SEED[i][1]
            if max_val != 0.0:
                num1 = random.uniform(min_val, max_val)
                num2 = random.uniform(min_val, max_val)

                if num1 > num2:
                    temp = num1
                    num1 = num2
                    num2 = temp

                value.append([num1, num2])
            else:
                value.append([0.0, 0.0])

        self.VALUE = value
        self.LIFE = time.time()
        self.set_initial()
        self._id = uuid.uuid4()

    def generate_min_max_single(self, std_dev=3):
        value = []
        temp = []

        for i in range(self.NUM_FEATURES):
            min_val = self.SEED[i][0]
            max_val = self.SEED[i][1]
            if max_val != 0.0:
                num1 = random.uniform(min_val, max_val)

                value.append(num1)
            else:
                value.append(0.0)

            min_val = max(0.0, num1 - (self.SEED[i][2] * std_dev))
            max_val = min(1.0, num1 + (self.SEED[i][2] * std_dev))

            temp.append([min_val, max_val])

        self.VALUE = temp
        self.single_value = value
        self.LIFE = time.time()
        self.set_initial()
        self._id = uuid.uuid4()

    def get_id(self):
        return self._id

    def set_id(self, _id):
        self._id = _id

    def get_value(self):
        return self.VALUE

    def set_value(self, value):
        self.VALUE = value

    def get_mean_value(self):
        mean_values = []

        for i in range(len(self.VALUE)):
            num1 = self.VALUE[i][0]
            num2 = self.VALUE[i][1]

            mean = (num1+num2)/2

            mean_values.append(mean)

        return mean_values

    def get_max_values(self):
        maxes = []

        for x in self.VALUE:
            maxes.append(x[1])
        return maxes

    def get_min_values(self):
        mins = []

        for x in self.VALUE:
            mins.append(x[0])
        return mins

    def get_single_value(self):
        return self.single_value

    def get_type(self):
        return self.TYPE

    def set_type(self, type):
        self.TYPE = type

    def get_life(self):
        return self.LIFE

    def set_life(self, life):
        self.LIFE = life

    def match(self, r, value):
        if len(self.get_value()) != len(value):
            return -1

        matches = 0
        for i in range(len(self.get_value())):
            if self.get_value()[i][0] <= value[i] <= self.get_value()[i][1]:
                matches += 1

                if matches >= r:
                    return True

        return False

    def get_database_values(self):
        values = {"VALUE": self.get_value(), "TYPE": bson.Binary(self.get_type()), "LIFE": self.get_life()}

        if self.get_id():
            values['_id'] = self.get_id()

        return values

    def compare_to(self, value):
        if len(self.get_value()) != len(value):
            return -1

        matches = 0
        for i in range(len(self.get_value())):
            if self.get_value()[i][0] <= value[i] <= self.get_value()[i][1]:
                matches += 1

        return matches

    def set_initial(self):
        self.TYPE = 'INITIAL'

    def set_immature(self):
        self.TYPE = 'IMMATURE'

    def set_mature(self):
        self.TYPE = 'MATURE'

    def set_memory(self):
        self.TYPE = 'MEMORY'

    def get_np_values(self):
        temp = np.array(self.VALUE, dtype='f4')

        return temp.ravel()
