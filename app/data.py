from os import getenv

from certifi import where
from dotenv import load_dotenv
from MonsterLab import Monster
from pandas import DataFrame
from pymongo import MongoClient


class Database:
    load_dotenv()
    database = MongoClient(getenv('DB_URL'), tlsCAFile=where())["Bandersnatch"]

    def __init__(self, collection: str = 'MonsterTable'):
        """ MongoDB: Instantiates the Database class with the collection method for further calls """
        self.collection = self.database[collection]

    def seed(self, amount: int = 10):
        """ MongoDB: Randomly generates [amount] of monsters to insert into Database Collection """
        monster_list = [Monster().to_dict() for _ in range(amount)]
        self.collection.insert_many(monster_list)

    def custom_add(self, monster: dict):
        """ MongoDB: Adds a User's Custom Monster to the DataBase """
        self.collection.insert_one(monster)

    def reset(self):
        """ MongoDB: Empties the Database Collection of all documents """
        return self.collection.delete_many({})

    def remove(self, deletions: int):
        """ MongoDB: Remove a finite amount of rows """
        for i in range(deletions):
            self.collection.delete_one({})

    def count(self) -> int:
        """ MongoDB: Counts the amount of documents in the Database Collection """
        return self.collection.count_documents({})

    def dataframe(self) -> DataFrame:
        """ MongoDB: Converts the Database Collection into a pandas DataFrame """
        item_details = self.collection.find()
        items_df = DataFrame(item_details)
        return items_df

    def html_table(self) -> str:
        """ MongoDB: Converts the Database Dataframe into an HTML Table """
        item_details = self.collection.find()
        items_df = DataFrame(item_details)
        items_html = items_df.to_html()
        return items_html


# Instantiation / Test Drivers
if __name__ == '__main__':

    # Initially Populate DB with 1000 Monsters (if less than)
    db = Database()
    if db.count() < 1000:
        difference = 1000 - db.count()
        db.seed(amount=difference)
    db.reset()  # Uncomment to reset table and test above!
