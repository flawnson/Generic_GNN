""" This file is for all split unittests for the project """

import unittest
import numpy as np

from utils.holdout import Holdout
from read.preprocessing import GenericDataset, PrimaryLabelset
from utils.logger import log


class TestSplitMethods(unittest.TestCase):
    def setUp(self) -> None:
        dataset: GenericDataset = None
        if json_data["dataset"] == "primary_labelset":
            dataset = PrimaryLabelset(json_data).dataset.to(device)
        else:
            raise NotImplementedError(f"{json_data['dataset']} is not a dataset")  # Add to logger when implemented
        pass

    def test_intersection(self):
        self.assertEqual(Holdout().split(), 'FOO')

    def test_balanced(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
