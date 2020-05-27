#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
from detector import Detector
from instance import Instance
import numpy as np


class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        self.db = self.client["blacksite"]

    def get_all_detectors(self):
        collection = self.db['detectors']
        temp = list(collection.find())
        detectors = {}
        for d in temp:
            detector = Detector(d['_id'], d['VALUE'], d['TYPE'], d['LIFE'])
            detectors[detector.get_id()] = detector

        return detectors

    def get_all_new_instances(self):
        collection = self.db['new_instances']
        temp = list(collection.find())
        instances = {}
        for i in temp:
            instance = Instance(i['_id'], i['VALUE'], i['TYPE'], i['DETECTOR_id'])
            for key, value in i:
                if key != '_id' and key != 'VALUE' and key != 'TYPE' and key != 'DETECTOR_id':
                    instance.add_feature(key, value)
            instances[instance.get_id()] = instance

        return instances

    def get_one(self, _id, collection):
        sample = np.array(list(collection.find({"_id" : _id})))
        return sample

    def update_detector_type(self, detector):
        collection = self.db['detectors']
        myquery = {"_id": detector.get_id()}
        newvalues = {"TYPE": detector.get_type()}
        collection.update_one(myquery, newvalues)

        # print("Database update successful")

    def add_detector(self, detector):
        collection = self.db['detectors']
        newvalues = detector.get_database_values()
        d = collection.insert_one(newvalues)
        return d

    def delete_detector(self, detector):
        collection = self.db['detectors']
        myquery = {"_id" : detector.get_id()}
        d = collection.delete_one(myquery)
        return d

    def delete_new_instance(self, instance):
        collection = self.db['new_instances']
        myquery = {"_id": instance.get_id()}
        i = collection.delete_one(myquery)
        return i

    def add_suspicious_instance(self, instance):
        collection = self.db['suspicious_instances']
        newvalues = instance.get_database_values()
        collection.insert_one(newvalues)
        return d

    def get_detectors_collection(self):
        return self.db['detectors']

    def get_new_instances_collection(self):
        return self.db['new_instances']

    def get_suspicious_instances_collection(self):
        return self.db['suspicious_instances']