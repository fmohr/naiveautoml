import sklearn.pipeline
from sklearn.base import clone


class Pipeline(sklearn.pipeline.Pipeline):

    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def clone(self):
        return clone(self)
