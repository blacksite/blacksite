#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
from pprint import pprint
import numpy as np
import math

class MongoDBConnect:



    def __init__(self):
        # connect to MongoDB
        client = MongoClient("mongodb+srv://comp895va:dnjn895@hbcu-scoreboard-sjn7a.mongodb.net")
        self.db = client["scorecard"]
        self.col = self.db["academicyear20182019"]

    def get_all(self):
        print("Retrieving all institutions")
        samples = list(self.col.find())
        print("All institutions successfully retrieved")
        return np.array(samples[:-1])

    def getOne(self, UNITID):
        sample = np.array(list(self.col.find({"UNITID" : UNITID})))
        return sample

    def get_HBCUs(self):
        samples = list(self.col.find({"HBCU": "1"}))
        return np.array(samples)

    def getHBCUNames(self):
        samples = []
        for x in self.col.find({"HBCU": "1"}):
            samples.append(list(x.values())[4])

        return np.array(samples)

    def get_variable_code(self):
        sample = self.col.find_one()

        variables_from_db = list(self.db["InstitutionalVariables"].find())
        variables_dict = {}
        for x in variables_from_db:
            variables_dict[str(x["VARIABLE NAME"])] = str(x["NAME OF DATA ELEMENT"])

        variable_codes = list(sample.keys())[24:]
        variables = []

        for code in variable_codes:

            if code in variables_dict:
                variables.append(variables_dict[code])
            else:
                variables.append(code)

        #print(variables)
        return np.array(variables)

    def update_scores(self, scores):

        for x in scores:
            myquery = {"_id": x["_id"]}
            newvalues = {"$set": {"SCORE": x["SCORE"]}}
            self.col.update_one(myquery, newvalues)

        print("Database update successful")

    def get_all_from_collection(self, collection_name):
        print("Retrieving all institutions for " + collection_name)
        temp_col = self.db[collection_name]
        samples = list(temp_col.find())
        print("All institutions successfully retrieved from " + collection_name)
        return np.array(samples[:-1])

    def update_scores_for_collection(self, collection_name, scores):
        temp_col = self.db[collection_name]

        for x in scores:
            myquery = {"_id": x["_id"]}
            newvalues = {"$set": {"SCORE": x["SCORE"]}}
            temp_col.update_one(myquery, newvalues)

        print("Database update successful for " + collection_name)