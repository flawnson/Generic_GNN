""" This file is for all split unittests for the project """

import unittest

from utils.holdout import Holdout
from read.preprocessing import GenericDataset, PrimaryLabelset
from utils.logger import log


class TestSplitMethods(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_intersection(self):
        self.assertEqual(Holdout().stratified_split(), 'FOO')

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
