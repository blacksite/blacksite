
class Detector:

    def __init__(self, _id=None, value=None, type=None, life=None):
        self._id = _id
        self.VALUE = value
        self.TYPE = type
        self.LIFE = life

    def get_id(self):
        return self._id

    def set_id(self, _id):
        self._id = _id

    def get_value(self):
        return self.VALUE

    def set_value(self, value):
        self.VALUE = value

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
            if self.get_value()[i] == value[i]:
                matches += 1

        if matches >= r:
            return 1
        else:
            return 0