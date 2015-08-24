__author__ = 'Blaise'

from forest import customRF
from generator import get_rules
from IO.read import read_from_csv
from utils import split_train_test
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from selector.selector import randomSelector, ruleScore1

import numpy as np
from evaluator.evaluator import buildCoverageMatrix, evaluate
import pandas as pd

if __name__ == "__main__":
    nb_learners = 10
    rf = customRF(nb_learners)
    df = read_from_csv()
    df_train, df_test = split_train_test(df,75)
    print len(df_train),len(df_test)
    rf.train(df_train)
    rf.test(df_test)

    estimators = rf.model.estimators_
    rules = []
    for estimator in estimators:
        rules.extend([ rule for rule in get_rules(estimator.tree_, df.columns)])


    print(len(rules))

    k = 10

    subsetrules =  randomSelector(rules,k)

    print(subsetrules)

    CMRandomTest = buildCoverageMatrix(df_test, subsetrules )

    print("Random selector")
    evaluate(CMRandomTest, df_test, len(rules), nb_learners, k=k)

    CMRuleScore1Train = buildCoverageMatrix(df_train, rules )

    ruleScore1(df_train, CMRuleScore1Train, rules,k)

    CMRuleScore1Test = buildCoverageMatrix(df_test, rules )

    print("Rule score 1")
    evaluate(CMRuleScore1Test, df_test, len(rules), nb_learners, k=k)