import requests


URL = "http://127.0.0.1:5000/predict"

json = {"utterance": "switch off the light please"}

if __name__ == '__main__':
    r = requests.post(url=URL, json=json)
    response = r.json()
    print('Intent: ', response['intent']['name'])
    print('Confidence: ', response['intent']['confidence'])
    print('Slots:')
    for slot in response['slots']:
        print(' ', slot['slot'], '=', slot['value'])
