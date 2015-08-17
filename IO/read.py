__author__ = 'Blaise'
import pandas as pd

def read_from_csv(csv_path='/home/rpc/Desktop/vmshared/sonar.csv', encoding='utf-8', nrows=None, dtype= None ):
        df = pd.read_csv(csv_path, encoding=encoding, nrows=nrows )
        return df
