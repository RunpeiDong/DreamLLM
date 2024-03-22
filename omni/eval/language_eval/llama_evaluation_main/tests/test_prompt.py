#!/usr/bin/env python3

import unittest

from llama_evaluation.prompts import FewshotPrompt, Prompt, load_bbh_prompt, BASE_TEMPLATE


class TestPrompt(unittest.TestCase):

    def setUp(self):
        self.input_str = "This is a test input string."

    def test_base_prompt(self):
        prefix1, val1 = "This is {}.", "a test"
        prompt = Prompt(template=prefix1)
        self.assertEqual(prompt.render(val1), prefix1.format(val1))
        self.assertIsInstance(prompt.render(val1), str)

        prompt = Prompt(template=None)
        self.assertEqual(prompt.render(val1), val1)
        self.assertIsInstance(prompt.render(val1), str)

    def test_few_shot(self):
        fewshot_prompt = FewshotPrompt("", template="")
        prompt = Prompt()
        self.assertEqual(fewshot_prompt.render(self.input_str), prompt.render(self.input_str))

        fewshot_prompt = FewshotPrompt("", template=None)
        self.assertEqual(fewshot_prompt.render(self.input_str), prompt.render(self.input_str))

    def test_bbh_prompt(self):
        subtask = [
            'boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa',
            'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton',
            'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects',
            'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting',
            'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names',
            'salient_translation_error_detection', 'snarks', 'sports_understanding',
            'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting',
        ]
        for task in subtask:
            load_bbh_prompt(task, template=BASE_TEMPLATE)


if __name__ == '__main__':
    unittest.main()
