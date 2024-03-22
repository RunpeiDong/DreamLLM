#!/usr/bin/env python3

from typing import List, Tuple

from llama_evaluation_main.llama_evaluation.prompts.template import BASE_TEMPLATE, BASE_FEWSHOT_TEMPLATE, fewshot_prompt


class Prompt:
    """Base class for all prompts. Directly return prompt."""

    def __init__(self, prefix: str = None, template: str = None):
        self.prefix = prefix
        self.template = template

    def render(self, *prompt: str):
        if self.template:
            prompt = self.template.format(*prompt)
        else:
            prompt = "".join(prompt)

        if self.prefix:
            prompt = self.prefix + prompt
        return prompt


class FewshotPrompt(Prompt):
    """Prompt with template for completion tasks."""

    def __init__(
        self, few_shot_examples: List[Tuple[str, str]],
        prefix: str = None, template: str = BASE_TEMPLATE,
        fewshot_template: str = BASE_FEWSHOT_TEMPLATE,
    ):
        self.fewshot_template = fewshot_template
        self.fewshot_prefix = fewshot_prompt(self.fewshot_template, few_shot_examples) if few_shot_examples else ""
        prefix = self.fewshot_prefix if prefix is None else prefix + self.fewshot_prefix
        super().__init__(prefix, template)
