from base64 import b64decode
import os

from Fortuna import random_int, random_float
from MonsterLab import Monster
from flask import Flask, render_template, request, current_app
from pandas import DataFrame

from app.data import Database
from app.graph import chart, corr_heatmap, bar_chart, damage_calc
from app.machine import Machine

SPRINT = 3
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

    if not db.dataframe().empty:

        # Altair Chart
        options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
        x_axis = request.values.get("x_axis") or options[1]
        y_axis = request.values.get("y_axis") or options[2]
        target = request.values.get("target") or options[4]
        graph = chart(
            df=db.dataframe().drop('_id', axis=1),
            x=x_axis,
            y=y_axis,
            target=target,
        ).to_json()

        # SNS Heatmap -- Drop Unique Cols
        heat_cols = options + ['Type', 'Damage']
        drop_cols = [i for i in db.dataframe().columns if i not in heat_cols]
        df_hm1, df_hm2 = db.dataframe().drop(drop_cols, axis=1), db.dataframe().drop(drop_cols, axis=1)
        heatmap, heatmap2 = corr_heatmap(df_hm1, ordinal=True), corr_heatmap(df_hm2)

        # Bar Chart -- Damage
        df_dmg = DataFrame({'Dice Roll + Modifier': db.dataframe()['Damage'],
                            'Average Damage': db.dataframe()['Damage'].apply(damage_calc)
                            })
        bar = bar_chart(x=df_dmg['Dice Roll + Modifier'], y=df_dmg['Average Damage'], df=df_dmg)

        return render_template(
            "view.html",
            options=options,
            x_axis=x_axis,
            y_axis=y_axis,
            target=target,
            count=db.count(),
            graph=graph,
            heatmap=heatmap,
            heatmap2=heatmap2,
            bar=bar
        )
    else:
        return render_template("view.html", warning='Please add some Monsters to the Database')


@APP.route("/model", methods=["GET", "POST"])
def model(model_params=None, tmp_error=None):
    if SPRINT < 3:
        return render_template("model.html")
    print(f'model_params: {model_params}')

    # Ensure the database is filled to train new models
    db = Database()
    if db.dataframe().empty:
        return render_template("model.html", warning='Please add some Monsters to the Database to train new models')

    # Determine Which Models Exist (if any)
    model_path = f'{current_app.root_path}\\models'
    dt_model = Machine.open(f'{model_path}\\dt_model.joblib') \
        if os.path.exists(f'{model_path}\\dt_model.joblib') else None
    rfc_model = Machine.open(f'{model_path}\\rfc_model.joblib') \
        if os.path.exists(f'{model_path}\\rfc_model.joblib') else None
    knn_model = Machine.open(f'{model_path}\\knn_model.joblib') \
        if os.path.exists(f'{model_path}\\knn_model.joblib') else None

    # Determine if a Temporary Model is waiting on Parameter Submission
    tmp_model = [s for s in os.listdir(model_path) if 'new' in s]
    tmp_model = tmp_model[0].split('.')[0] if tmp_model else None
    print(f'tmp_model: {tmp_model}')

    # If USER submitted form data
    if request.method == 'POST' or tmp_model:
        print(f"request.form.getlist('instantiate_models[]'): {request.form.getlist('instantiate_models[]')}")
        print(f"request.form.get('model_params'): {request.form.get('model_params')}")
        print(f"request.form.getlist('model_params[]'): {request.form.getlist('model_params[]')}")
        print(f"request.form.getlist('del_model[]'): {request.form.getlist('del_model[]')}")
        print(f"request.form.getlist('parameter_reset[]'): {request.form.getlist('parameter_reset[]')}")

        # If USER Instantiated New Model, No Temporary Model Pending
        if request.form.getlist('instantiate_models[]'):
            tmp_model = request.form.getlist('instantiate_models[]')[0]  # Extract str from list
            Machine.save(tmp_model=tmp_model, filepath=model_path)
            model_params = Machine.display_params(tmp_model=tmp_model)
            print(f'display model_params: {model_params}')

            return render_template("model.html", dt_model=dt_model, rfc_model=rfc_model,
                                   knn_model=knn_model, tmp_model=tmp_model, model_params=model_params)

        # If USER Submitted New Model Parameters
        if request.form.get('model_params'):
            # Retrieve the Model Parameters Form and Parse into Dictionary
            usr_parameters = request.form.getlist('model_params[]')
            model_params = Machine.parse_params(usr_parameters)
            print(f'parsed model_params: {model_params}')

            # Instantiate and Fit a New Model Using the Parameters and Monster Database
            options = ["Level", "Health", "Energy", "Sanity", "Rarity"]
            new_model = Machine(tmp_model=tmp_model, df=db.dataframe()[options],
                                target='Rarity', model_params=model_params)

            # Save the Model (Joblib Dictionary) and Delete the Temporary Model
            Machine.save(model=new_model, tmp_model=tmp_model,
                         filepath=model_path, df=db.dataframe()[options])
            os.remove(f'{model_path}\\{tmp_model}.joblib')
            tmp_model = None

        # USER wishes to Delete a Model!
        if request.form.getlist('del_model[]'):
            model_name = request.form.getlist('del_model[]')[0]
            os.remove(f'{model_path}\\{model_name}.joblib')

        # If USER Navigated away from ii. Model Parameter form submission and is hung up...
        if request.form.getlist('parameter_reset[]'):
            os.remove(f'{model_path}\\{tmp_model}.joblib')
            tmp_model = None

    # If Models were created in POST, open their Joblib Dictionaries:
    dt_model = Machine.open(f'{model_path}\\dt_model.joblib') \
        if os.path.exists(f'{model_path}\\dt_model.joblib') else None
    rfc_model = Machine.open(f'{model_path}\\rfc_model.joblib') \
        if os.path.exists(f'{model_path}\\rfc_model.joblib') else None
    knn_model = Machine.open(f'{model_path}\\knn_model.joblib') \
        if os.path.exists(f'{model_path}\\knn_model.joblib') else None

    return render_template("model.html", dt_model=dt_model, rfc_model=rfc_model,
                           knn_model=knn_model, tmp_model=tmp_model)


if __name__ == '__main__':
    APP.run()
