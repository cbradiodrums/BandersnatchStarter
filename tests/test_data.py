# import pytest

from app.data import Database
from MonsterLab import Monster
from random import randint

# Instantiate Database
db = Database()


def test_Database_Instantiated(db: any = None):
    """ Ensure the Database Class is instantiated """
    if db is None:
        db = Database()
    assert db is not None


def test_Database_Count(db: any = None):
    """ Ensure the Database count method works """
    if db is None:
        db = Database()
    assert type(db.count()) == int


def test_Database_Seed_Remove_Reset(db: any = None, empty: any = None):
    """ Ensure the Database can seed, remove, and reset correctly """
    if db is None:
        db = Database()

    # Random Adds and Deletes
    tests = 10  # the amount of random tests to conduct
    db_size = db.count()
    for test in range(tests):
        amount = randint(1, 100)
        db.seed(amount=amount)
        assert db.count() == amount + db_size
        db.remove(deletions=amount)
        assert db.count() == db_size

    # Test Emptying the DataBase
    empty = True  # Change to None or Comment Out to skip
    if empty:
        db.reset()
        assert db.count() == 0


def test_Database_custom_monster(db: any = None):
    """ Ensure that a custom monster can be added """
    if db is None:
        db = Database()

    tests = 10
    for test in range(tests):
        monster = Monster().to_dict()
        db.custom_add(monster)
        assert db.collection.find_one(monster)
        db.collection.delete_one(monster)
