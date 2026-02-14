import requests
import os
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
key = os.getenv("AZURE_OPENAI_API_KEY") #2024-02-15-preview


url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-15-preview"

headers = {
    "Content-Type": "application/json",
    "api-key": key
}

data = {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.text)
