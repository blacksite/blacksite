#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
import numpy as np
from instance import Instance
from detector import Detector

class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        self.db = self.client["blacksite"]

    def get_all_dataset(self):

        return np.array(list(self.db['dataset'].find()))

    def get_all_detectors(self):
        collection = self.db['detectors']
        temp = list(collection.find())
        detectors = {}
        for d in temp:
            detector = Detector(d['_id'], d['VALUE'], d['TYPE'], d['LIFE'])
            detectors[detector.get_id()] = detector

        return detectors

    def get_all_suspicious_instances(self):
        temp = list(self.db['suspicious_instances'].find())
        instances = {}
        for i in temp:
            instance = Instance(i['_id'], i['VALUE'], i['TYPE'], i['DETECTOR_id'])
            for key, value in i:
                if key != '_id' and key != 'VALUE' and key != 'TYPE' and key != 'DETECTOR_id':
                    instance.add_feature(key, value)
            instances[instance.get_id()] = instance

        return instances

    def remove_detector(self, detector):
        col = self.db['detectors']
        myquery = {"_id": detector.get_id()}
        col.delete_one(myquery)

    def get_one(self, _id, col):
        sample = np.array(list(col.find({"_id" : _id})))
        return sample

    def get_single_detector(self, _id):
        collection = self.db['detectors']
        myquery = {"_id": _id}
        temp = collection.find(myquery)
        detector = Detector(temp['_id'], temp['VALUE'], temp['TYPE'], temp['LIFE'])

        return detector

    def update_detector_type(self, detector):
        collection = self.db['detectors']
        myquery = {"_id": detector.get_id()}
        newvalues = {"TYPE": detector.get_type()}
        collection.update_one(myquery, newvalues)

        # print("Database update successful")

    def remove_suspicious_instance(self, instance):
        col = self.db['suspicious_instances']
        myquery = {"_id": instance.get_id()}
        col.delete_one(myquery)

    def add_confirmation_instance(self, instance):
        col = self.db['confirmation_instances']
        newvalues = {"VALUE": instance.get_value(), "TYPE": instance.get_type(),
                     "DETECTOR_id": instance.get_detector_id()}
        newvalues.update(instance.get_features())
        col.insert_one(instance)

    def get_dataset_collection(self):
        return self.db['dataset']

    def get_detectors_collection(self):
        return self.db['detectors']

    def get_suspicious_instance_collection(self):
        return self.db['suspicious_instances']

    def get_confirmation_instance_collection(self):
        return self.db['confirmation_instances']