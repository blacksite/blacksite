#Connect to MongDB
#Instantiate object

from pymongo import MongoClient
from instance import Instance

class MongoDBConnect:

    def __init__(self):
        # connect to MongoDB
        self.client = MongoClient("mongodb+srv://root:root@cluster0-mns8t.mongodb.net/test")
        # self.client = MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
        self.db = self.client["blacksite"]

    def add_instance(self, instance):
        collection = self.db['new_instances']
        newvalues = instance.get_database_values()
        collection.insert_one(newvalues).inserted_id

    def get_new_instances_collection(self):
        return self.db['new_instances']