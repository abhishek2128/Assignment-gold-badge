import requests
print("start ollama model ")
print( "prompt--what is the capital of India  under 50 word ?")
url = "http://localhost:11434/api/generate"

payload = {
    "model": "phi3",   # change to your installed model
    "prompt": "what is the capital of India  under 50 word ?",
    "stream": False ,  # for live streming True
    "options": {
        "num_predict": 50
    }
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()
    print("Response:\n")
    print(data["response"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
