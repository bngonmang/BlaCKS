__author__ = 'Blaise'

from forest import customRF
from generator import get_rules
from IO.read import read_from_csv
from utils import split_train_test
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from selector.selector import randomSelector

import numpy as np
from evaluator.evaluator import buildCoverageMatrix, evaluate
import pandas as pd

if __name__ == "__main__":
    nb_learners = 100
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

        #tree.export_graphviz(estimator,out_file='tree.dot')

    # expr = '( ( (dtsurv - dtdebeff) <= 5 ) and ( (typsin < 1) or (typsin>1) ))'
    # expr = parse(expr)
    # print expr
    #
    # expr = '((attribute_22 <= 1.0) and (attribute_21 <= 2) and (attribute_2 > 3))'
    # expr = parse(expr)
    # print expr
    #print(df_train

    #CM = buildCoverageMatrix(df_test, rules)

    #evaluate(CM, df_test,nb_learner=5)

    subsetrules =  randomSelector(rules,100)

    print(subsetrules)

    CM2 = buildCoverageMatrix(df_test, subsetrules )

    evaluate(CM2, df_test, len(rules), nb_learners)

