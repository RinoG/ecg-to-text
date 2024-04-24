import requests, uuid, json


with open('microsoft_cloud_translate_api_key.json', 'r') as file:
    api_info = json.load(file)

# Add your key and endpoint
key = api_info['key']
location = api_info['location']
endpoint = "https://api.cognitive.microsofttranslator.com"


path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'to': 'en'
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# You can pass more than one object in body.
body = [{
    'text': "sinusrhythmus verdacht auf p-sinistrocardiale ueberdrehter linkstyp linksanteriorer hemiblock rechtsschenkelblock bifaszikulÄrer block qrs(t) abnorm    anteriorer infarkt     alter unbest.    inferiorer infarkt     wahrscheinlich alt 4.46                 ,"
}]

request = requests.post(constructed_url, params=params, headers=headers, json=body)
response = request.json()

print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))