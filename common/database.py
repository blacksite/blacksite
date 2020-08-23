#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
from common.detector import Detector
from common.instance import Instance
import numpy as np


class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        self.db = self.client["blacksite"]

    def get_all_dataset(self):
        return np.array(list(self.db['dataset'].find()))

    def add_new_instance(self, instance):
        collection = self.db['new_instances']
        new_values = instance.get_database_values()
        collection.insert_one(new_values)

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

    def delete_new_instance(self, instance):
        collection = self.db['new_instances']
        myquery = {"_id": instance.get_id()}
        i = collection.delete_one(myquery)
        return i

    def get_one(self, _id, collection):
        sample = np.array(list(collection.find({"_id" : _id})))
        return sample

    def get_all_detectors(self):
        collection = self.db['detectors']
        temp = list(collection.find())
        detectors = {}
        for d in temp:
            detector = Detector(d['_id'], d['VALUE'], d['TYPE'], d['LIFE'])
            detectors[detector.get_id()] = detector

        return detectors

    def update_detector(self, detector):
        collection = self.db['detectors']
        myquery = {"_id": detector.get_id()}
        newvalues = {"$set" : detector.get_database_values()}
        collection.update_one(myquery, newvalues)

    def update_detector_type(self, detector):
        collection = self.db['detectors']
        myquery = {"_id": detector.get_id()}
        newvalues = {"$set" : {'TYPE': detector.get_type()}}
        collection.update_one(myquery, newvalues)

    def add_detector(self, detector):
        collection = self.db['detectors']
        newvalues = detector.get_database_values()
        _id = collection.insert_one(newvalues).inserted_id
        return _id

    def delete_detector(self, detector):
        collection = self.db['detectors']
        myquery = {"_id" : detector.get_id()}
        d = collection.delete_one(myquery)
        return d

    def add_suspicious_instance(self, instance):
        collection = self.db['suspicious_instances']
        newvalues = instance.get_database_values()
        collection.insert_one(newvalues)

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

    def remove_suspicious_instance(self, instance):
        col = self.db['suspicious_instances']
        myquery = {"_id": instance.get_id()}
        col.delete_one(myquery)

    def add_confirmation_instance(self, instance):
        col = self.db['confirmation_instances']
        newvalues = instance.get_database_values()
        col.insert_one(newvalues)

    def get_dataset_collection(self):
        return self.db['dataset']

    def get_detectors_collection(self):
        return self.db['detectors']

    def get_new_instances_collection(self):
        return self.db['new_instances']

    def get_suspicious_instances_collection(self):
        return self.db['suspicious_instances']

    def get_confirmation_instance_collection(self):
        return self.db['confirmation_instances']