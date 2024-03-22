#!/usr/bin/env python3

from llama_evaluation_main.llama_evaluation.prompts.template import BASE_FEWSHOT_TEMPLATE, SFT_FEWSHOT_TEMPLATE, fewshot_prompt

__all__ = ["GSM8K_PROMPT", "SFT_GSM8K_PROMPT"]


GSM8K_EXAMPLES = [
# 1-shot
(
r"""Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?""",
r"""Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The final answer is 72."""
#Final Answer: The final answer is 72. I hope it is correct."""
),
# 2-shot
(
r"""Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?""",
r"""Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The final answer is 10."""
#Final Answer: The final answer is 10. I hope it is correct."""
),
# 3-shot
(
r"""James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?""",
r"""He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year. The final answer is 624."""
#Final Answer: The final answer is 624. I hope it is correct."""
),
# 4-shot
(
r"""Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?""",
r"""Half of the number of Randy's mango trees is 60/2 = <<60/2=30>>30 trees. So Randy has 30 - 5 = <<30-5=25>>25 coconut trees. Therefore, Randy has 60 + 25 = <<60+25=85>>85 treeson his farm. The final answer is 85."""
#Final Answer: The final answer is 85. I hope it is correct."""
),
]

GSM8K_PROMPT = fewshot_prompt(BASE_FEWSHOT_TEMPLATE, GSM8K_EXAMPLES)
SFT_GSM8K_PROMPT = fewshot_prompt(SFT_FEWSHOT_TEMPLATE, GSM8K_EXAMPLES)
