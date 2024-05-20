# Unit test suite
import unittest

from models.unsupervised.edit_distance import TreeEditDistance


class TestTreeEditDistance(unittest.TestCase):

    def setUp(self):
        self.ted_calculator = TreeEditDistance()

    def test_empty_strings(self):
        str1 = ""
        str2 = ""
        expected_distance = 0
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_one_empty_string(self):
        str1 = ""
        str2 = "nonempty"
        expected_distance = len(str2)
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

        str1 = "nonempty"
        str2 = ""
        expected_distance = len(str1)
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_identical_strings(self):
        str1 = "identical"
        str2 = "identical"
        expected_distance = 0
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_different_strings(self):
        str1 = "kitten"
        str2 = "sitting"
        expected_distance = 3
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_another_case(self):
        str1 = "flaw"
        str2 = "lawn"
        expected_distance = 2
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_case_with_substitution(self):
        str1 = "abc"
        str2 = "yabd"
        expected_distance = 2
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)

    def test_longer_strings(self):
        str1 = "intention"
        str2 = "execution"
        expected_distance = 5
        self.assertEqual(self.ted_calculator.compute(
            str1, str2), expected_distance)


if __name__ == '__main__':
    unittest.main()
