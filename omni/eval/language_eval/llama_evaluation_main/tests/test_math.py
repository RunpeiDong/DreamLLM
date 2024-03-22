#!/usr/bin/env python3

import unittest

from datasets import load_dataset
from llama_evaluation.utils import random_choice_data


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.dataset = load_dataset("competition_math", split="test")

    def test_random_id(self):
        prefix_idx = [3, 12, 13, 15, 16, 44, 47, 52, 57, 71]
        idx = random_choice_data(self.dataset)
        self.assertEqual(idx[:10], prefix_idx)


if __name__ == '__main__':
    unittest.main()
