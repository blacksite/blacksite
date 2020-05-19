#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
import numpy as np


class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/COMP895-Brown?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
        self.db = client["blacksite"]
        self.col = self.db["detectors"]

    def get_all(self):
        temp = list(self.col.find())
        detectors = {}
        for d in temp:
            temp = {'_id': d['_id'], "VALUE": d['VALUE'], "TYPE": d['TYPE'], 'LIFE': d['LIFE']}
            detectors[d["_id"]] = temp

        return detectors

    def get_one(self, _id):
        sample = np.array(list(self.col.find({"_id" : _id})))
        return sample

    def update_detector(self, detector):

        myquery = {"_id": detector["_id"]}
        newvalues = {"$set": {"TYPE": detector["TYPE"]}}
        self.col.update_one(myquery, newvalues)

        print("Database update successful")

    def add_detector(self, detector):
        d = self.col.insert_one(detector)
        return d

    def delete_detector(self, detector):
        d = self.col.delete_one(detector)
        return d