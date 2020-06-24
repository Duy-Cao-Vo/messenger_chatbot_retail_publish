import pymongo
from pymongo import MongoClient
import json


cluster = MongoClient("mongodb+srv://duyvo:1234@cluster0-0eopm.mongodb.net/test?retryWrites=true&w=majority")
db = cluster.get_database('messenger_chatbot')

auto_marketing = {"started": {"content": "",
                              "quick_replies": [""],
                              "buttons": []},
                  "not_read": {"content": "",
                               "quick_replies": [""],
                               "buttons": []},
                  "not_clicked": {"content": "",
                                 "quick_replies": [""],
                                 "buttons": []},
                  "haved_clicked": {"content": "",
                                    "quick_replies": [""],
                                    "buttons": []}
                   }

collection4 = db.auto_marketing
collection4.insert_one(auto_marketing)