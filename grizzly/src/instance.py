
class Instance:

    def __init__(self, _id=None, value=None, type=None, detector_id=None):
        self._id = _id
        self.VALUE = value
        self.TYPE = type
        self.DETECTOR_id = detector_id
        self.FEATURES = {}

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

    def get_detector_id(self):
        return self.DETECTOR_id

    def set_detector_id(self, detector_id):
        self.DETECTOR_id = detector_id

    def add_feature(self, key, value):
        self.FEATURES[key] = value

    def get_features(self):
        return self.FEATURES