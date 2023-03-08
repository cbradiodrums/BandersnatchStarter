from base64 import b64decode
import os

from Fortuna import random_int, random_float
from MonsterLab import Monster
from flask import Flask, render_template, request
from pandas import DataFrame

from app.data import Database
from app.graph import chart
from app.machine import Machine

SPRINT = 1
APP = Flask(__name__)


@APP.route("/")
def home():
    return render_template(
        "home.html",
        sprint=f"Sprint {SPRINT}",
        monster=Monster().to_dict(),
        password=b64decode(b"VGFuZ2VyaW5lIERyZWFt"),
    )


@APP.route("/data", methods=["GET", "POST"])
def data():
    if SPRINT < 1:
        return render_template("data.html")
    db = Database()

    # If the USER submitted data -- Custom Block
    if request.method == 'POST':

        # If random monsters are to be Deleted
        if 'delete_rows' in request.form.getlist('data_manipulation[]'):

            # If the user actually selects a value for deletions
            if request.form.get('delete_rows'):

                if 'ALL' in request.form.get('delete_rows'):
                    db.reset()

                else:
                    deletions = int(request.form.get('delete_rows'))
                    db.remove(deletions=deletions)

        # If random Monsters will be added
        elif 'add_rows' in request.form.getlist('data_manipulation[]'):

            # If the user actually selects a value for additions
            if request.form.get('add_rows'):

                if 'RESET' in request.form.get('add_rows'):
                    restore = 1000 - db.count()
                    if restore > 0:
                        db.seed(amount=restore)
                    elif restore < 0:
                        db.remove(deletions=abs(restore))

                else:
                    additions = int(request.form.get('add_rows'))
                    db.seed(amount=additions)

        # If User added a custom monster
        elif 'custom' in request.form.getlist('data_manipulation[]'):

            default = Monster().to_dict()
            if request.form.get('dice_amount') and request.form.get('dice_type'):
                damage = f"{request.form.get('dice_amount')}" \
                             f"{request.form.get('dice_type')}" \
                             f"{request.form.get('mod')}"
            else:
                damage = default['Damage']

            usr_monster = {
              "Name": request.form.get('name') if request.form.get('name').strip(' ') else default['Name'],
              "Type": request.form.get('type') if request.form.get('type').strip(' ') else default['Type'],
              "Level": request.form.get('level') if request.form.get('level') else default['Level'],
              "Rarity": request.form.get('rarity') if request.form.get('rarity') else default['Rarity'],
              "Damage": damage,
              "Health": request.form.get('health') if request.form.get('health') else default['Health'],
              "Energy": request.form.get('energy') if request.form.get('energy') else default['Energy'],
              "Sanity": request.form.get('sanity') if request.form.get('sanity') else default['Sanity'],
              "Timestamp": default["Timestamp"],
            }
            db.custom_add(monster=usr_monster)

    return render_template(
        "data.html",
        count=db.count(),
        table=db.html_table(),
    )


@APP.route("/view", methods=["GET", "POST"])
def view():
    if SPRINT < 2:
        return render_template("view.html")
    db = Database()
    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    x_axis = request.values.get("x_axis") or options[1]
    y_axis = request.values.get("y_axis") or options[2]
    target = request.values.get("target") or options[4]
    graph = chart(
        df=db.dataframe(),
        x=x_axis,
        y=y_axis,
        target=target,
    ).to_json()
    return render_template(
        "view.html",
        options=options,
        x_axis=x_axis,
        y_axis=y_axis,
        target=target,
        count=db.count(),
        graph=graph,
    )


@APP.route("/model", methods=["GET", "POST"])
def model():
    if SPRINT < 3:
        return render_template("model.html")
    db = Database()
    options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
    filepath = os.path.join("app", "model.joblib")
    if not os.path.exists(filepath):
        df = db.dataframe()
        machine = Machine(df[options])
        machine.save(filepath)
    else:
        machine = Machine.open(filepath)
    stats = [round(random_float(1, 250), 2) for _ in range(3)]
    level = request.values.get("level", type=int) or random_int(1, 20)
    health = request.values.get("health", type=float) or stats.pop()
    energy = request.values.get("energy", type=float) or stats.pop()
    sanity = request.values.get("sanity", type=float) or stats.pop()
    prediction, confidence = machine(DataFrame(
        [dict(zip(options, (level, health, energy, sanity)))]
    ))
    info = machine.info()
    return render_template(
        "model.html",
        info=info,
        level=level,
        health=health,
        energy=energy,
        sanity=sanity,
        prediction=prediction,
        confidence=f"{confidence:.2%}",
    )


if __name__ == '__main__':
    APP.run()
