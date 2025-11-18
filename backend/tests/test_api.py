import requests

URL = "http://127.0.0.1:8000/predict"

data = {"features": [5.1, 3.5, 1.4, 0.2]}

response = requests.post(URL, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
