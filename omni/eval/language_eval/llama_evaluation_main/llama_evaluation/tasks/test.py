#!/usr/bin/env python3

import requests

url = "http://192.18.14.32:8080/generate"


headers = {
    "Content-Type": "application/json"
}
data = {
    "inputs": "def print_helloworld():\n",
    "parameters": {
        "max_new_tokens": 512,
        "details": True,
    }
}

response = requests.post(url, json=data, headers=headers)
print(response.text)

