#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
import numpy as np

class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        self.db = self.client["blacksite"]

    def get_all_dataset(self):

        return np.array(list(self.db['dataset'].find()))

    def get_all_detectors(self):
        temp = list(self.db['detectors'].find())
        detectors = {}
        for d in temp:
            temp = {'_id': d['_id'], "VALUE": d['VALUE'], "TYPE": d['TYPE'], 'LIFE': d['LIFE']}
            detectors[d["_id"]] = temp

        return detectors

    def get_all_suspicious_instances(self):
        temp = list(self.db['suspicious_instances'].find())
        detectors = {}
        for i in temp:
            temp = {'_id': i['_id'], "VALUE": i['VALUE'], "DETECTOR_id": i['DETECTOR_id']}
            detectors[i["_id"]] = temp

        return detectors

    def remove_detector(self, detector):
        col = self.db['detectors']
        myquery = {"_id": detector["_id"]}
        col.delete_one(myquery)

    def get_one(self, _id, col):
        sample = np.array(list(col.find({"_id" : _id})))
        return sample

    def update_detector(self, detector):
        col = self.db['detectors']
        myquery = {"_id": detector["_id"]}
        newvalues = {"$set": {"TYPE": detector["TYPE"]}}
        col.update_one(myquery, newvalues)

        # print("Database update successful")

    def remove_suspicious_instance(self, instance):
        col = self.db['suspicious_instances']
        myquery = {"_id": instance["_id"]}
        col.delete_one(myquery)

    def add_confirmation_instance(self, instance):
        col = self.db['confirmation_instances']
        col.insert_one(instance)

    def get_dataset_collection(self):
        return self.db['dataset']

    def get_detectors_collection(self):
        return self.db['detectors']

    def get_suspicious_instance_collection(self):
        return self.db['suspicious_instances']

    def get_confirmation_instance_collection(self):
        return self.db['confirmation_instances']