from pandas import DataFrame
import joblib
from sklearn.ensemble import RandomForestClassifier
import datetime


class Machine:

    def __init__(self, df: DataFrame):
        target = df["Rarity"]
        features = df.drop(columns=["Rarity"])
        self.model = RandomForestClassifier()
        self.model.fit(features, target)

    def __call__(self, feature_basis: DataFrame):
        prediction = self.model.predict(feature_basis)[0]
        confidence = [c for c in self.model.predict_proba(feature_basis)[0]][int(prediction.split(" ")[1])]
        return prediction, confidence

    def save(self, filepath: str, name: str = 'RFC'):
        joblib.dump({"model": self, "info": f"{name}-{datetime.datetime.now().strftime('%Y/%m/%d(%H:%M:%S)')}"},
                    filepath)

    @staticmethod
    def open(filepath: str):
        return joblib.load(filepath)

    def info(self: dict):
        return self['info']
