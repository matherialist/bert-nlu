import requests


URL = "http://127.0.0.1:5000/predict"

json = {"utterance": "make me a reservation in south carolina"}

r = requests.post(url=URL, json=json)
response = r.json()
print(response)
