__author__ = 'Blaise'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


class customRF:
    """ A custom class to define a random forest model """
    def __init__(self, nb_learners):
        self.model = RandomForestClassifier(n_estimators=nb_learners, class_weight=None, max_depth=4)
    def train(self, df_train):
        print df_train.columns[1:len(df_train.columns) -2]
        print("Model training...")
        self.model.fit(df_train[df_train.columns[1:len(df_train.columns)-2]], df_train['Class'])

        print("End of training")


    def test(self, df_test):
            predictions = self.model.predict(df_test[df_test.columns[1:len(df_test.columns)-2]])

            score = f1_score(df_test['Class'], predictions)
            print("Model F1 score: " + str(score))
            print("Model Accuracy score: " + str(accuracy_score(df_test['Class'], predictions)))
            #print(list(predictions))
            return predictions
