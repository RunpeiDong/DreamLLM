#!/usr/bin/env python3

import unittest

from llama_evaluation.utils import ensure_model_info_exist, get_max_eval_count


class TestDatabase(unittest.TestCase):

    def test_max_evalcount(self):
        kwargs = {
            "model_name": "llama",
            "temperature": 0.1,
            "topp": 0.95,
            "topk": -1,
        }
        model_id = ensure_model_info_exist(**kwargs)
        eval_cnt = get_max_eval_count(model_id)
        self.assertEqual(eval_cnt, 1)

        eval_cnt = get_max_eval_count(-1)
        self.assertEqual(eval_cnt, 0)


if __name__ == '__main__':
    unittest.main()
