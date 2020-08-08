""" This file contains all scoring methods for use throughout the project. Some are merely wrappers around SciKitLearn,
    others are custom scoring mechanisms for semi-supervised graph learning. The purpose of the class is to return one
    single score object containing all scores and str() and repr() methods for prettier printing"""


class Scores:
    def __init__(self, score_config):
        self.score_config = score_config