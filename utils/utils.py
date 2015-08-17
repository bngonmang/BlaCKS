__author__ = 'Blaise'

def split_train_test(df, percentage_train=50):
        """ Generates an index with first the training indices. """
        return (
            df.loc[(df.index.values % 100) < percentage_train].reset_index().copy(),
            df.loc[~((df.index.values % 100) < percentage_train)].reset_index().copy(),
        )