# imports
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance = Pipeline([
          ('distance_pipe', DistanceTransformer()),
          ('scaler', MinMaxScaler())
          ])
        time = Pipeline([
              ('time_pipe', TimeFeaturesEncoder('pickup_datetime')),
              ('scaler', OneHotEncoder())
        ])
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        preprocessing = ColumnTransformer([
        ('distance', distance, dist_cols),
        ('time', time, time_cols )
        ])
        pipeline = Pipeline([
        ('prepro', preprocessing),
        ('model', Lasso())
        ])

        return pipeline

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        return pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.set_pipeline()
        pipeline_trained = self.run()
        y_pred = pipeline_trained.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    # get data

    df = get_data()
    # clean data
    df_clean = clean_data(df)

    # set X and y
    X = df_clean.drop("fare_amount", axis=1)
    y = df_clean.fare_amount
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    trainer = Trainer(X_train,y_train)
    pipe = trainer.set_pipeline()
    pipe_trained = trainer.run()
    # evaluate
    result = trainer.evaluate(X_test,y_test)
    print('TODO')
