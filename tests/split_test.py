""" This file is for all split unittests for the project """

import unittest
import numpy as np

from utils.holdout import Holdout
from read.preprocessing import GenericDataset, PrimaryLabelset
from utils.logger import log
from utils.helper import parse_arguments


class TestSplitMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.json_data, self.device = parse_arguments()
        dataset: GenericDataset = None
        if self.json_data["dataset"] == "primary_labelset":
            self.dataset = PrimaryLabelset(self.json_data).dataset.to(self.device)
        else:
            raise NotImplementedError(f"{self.json_data['dataset']} is not a dataset")  # Add to logger when implemented

    def test_type(self):
        self.assertEqual(type(Holdout(self.json_data, self.dataset).split()), dict)

    def test_intersection(self):
        self.assertEqual(np.unique(Holdout(self.json_data, self.dataset).split().values()), 1)

    def test_balanced(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_all(self):
        pass


if __name__ == '__main__':
    unittest.main()
