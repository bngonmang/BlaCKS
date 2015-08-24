from __future__ import division
__author__ = 'Blaise'
import random
import numpy as np
from operator import itemgetter

def ruleScore1(df, coverageMatrix, rules, k=10):
    scores = {}

    for j in range(len(coverageMatrix[0])):
        #print coverageMatrix[:,j]
        #print df['Class']
        cov = np.count_nonzero(coverageMatrix[:,j])
        cc = np.count_nonzero(coverageMatrix[:,j] * df['Class'].values )
        ic = cov - cc
        scores[j] = ((cc-ic)/(cc+ic)) + (cc/(ic+4))
        #print scores[j]

    l = scores.items()
    l.sort(key=itemgetter(1),reverse=True)
    #print l

    return [rules[r] for (r,v) in l][:k]

def randomSelector(rules, k=10):
    return random.sample(rules,k)


