import requests
import argparse

URL = "http://127.0.0.1:5000/predict"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter utterance.')
    parser.add_argument('utterance', type=str)
    args = parser.parse_args()
    json = {"utterance": args.utterance}
    r = requests.post(url=URL, json=json)
    response = r.json()
    print('Intent: ', response['intent']['name'])
    print('Confidence: ', response['intent']['confidence'])
    if len(response['slots']) > 0:
        print('Slots:')
        for slot in response['slots']:
            print(' ', slot['slot'], '=', slot['value'])
    else:
        print('Slots:\n', ' None')
