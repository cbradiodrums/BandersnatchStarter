from pandas import DataFrame
import joblib
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Machine:

    @staticmethod
    def save(tmp_model: str = None, model: any = None, filepath: str = None, df: DataFrame = None):
        """ Saves a (Temporary) New Model or Trained ML Model to Disk """

        # Temporary Model Save File
        if tmp_model and not model:
            joblib.dump({"info": {"Created": f"{datetime.datetime.now().strftime('%Y/%m/%d(%H:%M:%S)')}"}},
                        f'{filepath}\\{tmp_model}.joblib')

        # New Model with Parameters
        if model:
            tmp = joblib.load(f'{filepath}\\{tmp_model}.joblib')
            model_name = f"{tmp_model.split('_')[1]}_model"
            joblib.dump({"model": model,
                         "info":
                             {
                                 "Created": tmp['info']['Created'],
                                 "Trained": f"{datetime.datetime.now().strftime('%Y/%m/%d(%H:%M:%S)')}",
                                 "Type": f"{model.model}",
                                 "Training Set": f"{df.columns if not df.empty else ''} "
                                                 f"|| Rows x Columns: {df.shape if not df.empty else ''}"}
                         },
                        f'{filepath}\\{model_name}.joblib')

    @staticmethod
    def display_params(tmp_model: str = None, return_defaults: bool = None):
        """ Display Parameters dependent on classifier for USER submission """

        # List of Common Classifier Parameters
        dt_params = {"random_state": [[_ for _ in range(1, 43)], 42],
                     "max_depth": [[_ for _ in range(1, 101)], 5],
                     "min_samples_split": [[_ for _ in range(1, 101)], 10],
                     "min_samples_leaf": [[_ for _ in range(1, 101)], 5],
                     "criterion": [["entropy", "gini"], "gini"]}
        rfc_params = {"random_state": [[_ for _ in range(1, 43)], 42],
                      "n_estimators": [[_ for _ in range(1, 1001)], 100],
                      "min_samples_split": [[_ for _ in range(1, 101)], 10],
                      "min_samples_leaf": [[_ for _ in range(1, 101)], 5],
                      "criterion": [["entropy", "gini"], "gini"],
                      "max_depth": [[_ for _ in range(1, 101)], 5]}
        knn_params = {"n_neighbors": [[_ for _ in range(1, 101)], 5],
                      "weights": [['uniform', 'distance'], 'uniform'],
                      "algorithm": [['auto', 'brute', 'kd_tree', 'ball_tree'], 'auto'],
                      "p": [[_ for _ in range(1, 11)], 2]}
        parameter_choices = {"new_dt": dt_params, "new_rfc": rfc_params, "new_knn": knn_params}

        if not return_defaults:
            return parameter_choices[tmp_model]

        # If the USER navigated away before submitting defaults with temporary ML joblib file
        else:

            default_parameters = {}
            for key, value in parameter_choices[tmp_model]:
                default_parameters.update({key: value[1]})

            return default_parameters

    @staticmethod
    def parse_params(usr_parameters: list = None):
        """ IN: USER Forms as List, OUT: Model Parameters as Dictionary """

        model_params = {}
        for param_pair in usr_parameters:
            key, value = param_pair.split('-')[0], param_pair.split('-')[1]
            try:
                value = int(value)
            except ValueError:
                pass  # print(f'{value} cannot be of type: integer') # VERIFY!!
            model_params.update({key: value})

        return model_params

    def __init__(self, tmp_model: str, df: DataFrame = None, target: str = None, model_params: dict = None):
        """ Initializes a model based on parameters, target, and feature matrix"""

        # 1) Instantiate a Model dependent on type and submitted USER parameters
        if tmp_model == 'new_dt':

            self.model = DecisionTreeClassifier(**model_params)

        elif tmp_model == 'new_rfc':

            self.model = RandomForestClassifier(**model_params)

        elif tmp_model == 'new_knn':

            self.model = KNeighborsClassifier(**model_params)

        # 2) Fit and train the model parsed from above
        features = df.drop(columns=[target])
        self.model.fit(features, df[target])

    def __call__(self, feature_basis: DataFrame):
        prediction = self.model.predict(feature_basis)[0]
        confidence = [c for c in self.model.predict_proba(feature_basis)[0]][int(prediction.split(" ")[1])]
        return prediction, confidence

    @staticmethod
    def open(filepath: str):
        return joblib.load(filepath)

    def info(self: dict):
        return self['info']
