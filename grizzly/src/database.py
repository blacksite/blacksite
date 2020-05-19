#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
from pprint import pprint
import numpy as np
import math


class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        self.db = self.client["blacksite"]

    def get_all(self, col):
        temp = list(col.find())
        detectors = {}
        for d in temp:
            temp = {'_id': d['_id'], "VALUE": d['VALUE'], "TYPE": d['TYPE'], 'LIFE': d['LIFE']}
            detectors[d["_id"]] = temp

        return detectors

    def get_one(self, _id, col):
        sample = np.array(list(col.find({"_id" : _id})))
        return sample

    def update_detector(self, detector):
        col = self.db['detectors']
        myquery = {"_id": detector["_id"]}
        newvalues = {"$set": {"TYPE": detector["TYPE"]}}
        col.update_one(myquery, newvalues)

        # print("Database update successful")

    def remove_new_instance(self, instance):
        col = self.db['new_instance']
        myquery = {"_id": instance["_id"]}
        col.delete_one(myquery)

    def add_suspicious_instance(self, instance):
        col = self.db['suspicious_instance']
        col.insert_one(instance)

    def get_dataset_collection(self):
        return self.db['dataset']

    def get_detectors_collection(self):
        return self.db['detectors']

    def get_suspicious_instance_collection(self):
        return self.db['suspicious_instance']

    def get_new_instance_collection(self):
        return self.db['new_instance']