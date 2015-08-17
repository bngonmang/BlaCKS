from __future__ import division
__author__ = 'Blaise'

import numpy as np
from interpreter.interpreter import evaluate_rule_on_df, tree_to_string, parse
from sklearn.metrics import  f1_score

def buildCoverageMatrix(df, rules):
    result = np.zeros((len(df.index), len(rules)))
    print len(rules)
    for i in range(len(rules)):
        for  idx, row in df.iterrows():
            result[idx,i] = evaluate_rule_on_df(rules[i],row)


    return result

def evaluate(coverageMatrix, df, total_rules_number, nb_learner=10, score_function = f1_score, k=100):
    print( ((k/total_rules_number)*nb_learner)/2 )
    predictions = [ int( (np.count_nonzero(row) >= ((k/total_rules_number)*nb_learner)/2 ) )  for row in coverageMatrix ]
    #print (predictions)
    #print(df['Class'])
    print(score_function(df['Class'], predictions))