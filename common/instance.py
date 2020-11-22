
class Instance:

    def __init__(self, _id=None, value=None, classification=None):
        self._id = _id
        self.VALUE = value
        self.DETECTOR_IDS = []
        self.FEATURES = {}
        self.CLASSIFICATION = classification

    def get_id(self):
        return self._id

    def set_id(self, _id):
        self._id = _id

    def get_value(self):
        return self.VALUE

    def set_value(self, value):
        self.VALUE = value

    def add_detector_id(self, detector_id):
        self.DETECTOR_IDS.append(detector_id)

    def get_detector_ids(self):
        return self.DETECTOR_IDS

    def add_feature(self, key, value):
        self.FEATURES[key] = value

    def get_features(self):
        return self.FEATURES

    def get_classification(self):
        return self.CLASSIFICATION

    def set_classification(self, classification):
        self.CLASSIFICATION = classification

    def get_database_values(self):
        values = {"VALUE": self.get_value(),
                     "DETECTOR_id": self.get_detector_id()}
        values.update(self.get_features())

        return values
