#!/usr/bin/env python3

import unittest

from llama_evaluation.utils import extract_markdown


class TestCodegen(unittest.TestCase):

    def test_extract_md(self):

        text = """
This is some text with Python code blocks:
```python
def f():
    pass
```
blabla
```
print("H")
```
"""
        result = extract_markdown(text, python_only=False)
        self.assertEqual(result, ['def f():\n    pass', 'print("H")'])
        result = extract_markdown(text, python_only=True)
        self.assertEqual(result, ['def f():\n    pass'])


if __name__ == '__main__':
    unittest.main()
