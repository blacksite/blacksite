import bson
import random
import time
import uuid


class Detector:

    def __init__(self, num_features=None, _id=None, value=None, type=None, life=None, seed=None):
        self._id = _id
        self.VALUE = value
        self.TYPE = type
        self.LIFE = life
        self.NUM_FEATURES = num_features
        self.SEED = seed
        self.generate()

    def generate(self):
        value = []

        for i in range(self.NUM_FEATURES):
            num1 = random.uniform(0, self.SEED[i])
            num2 = random.uniform(0, self.SEED[i])

            if num1 < num2:
                value.append((num1,num2))
            else:
                value.append((num2,num1))

        self.VALUE = value
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