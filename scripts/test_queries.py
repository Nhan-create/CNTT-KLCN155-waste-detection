import urllib3
urllib3.disable_warnings()
import requests

headers = {'User-Agent': 'WasteDetectionResearchBot/1.0 (academic research; contact@university.edu)'}

queries = [
    ('pin_1', 'corroded battery filetype:bitmap'),
    ('pin_2', 'alkaline battery leak corrosion filetype:bitmap'),
    ('can', 'crushed can litter filetype:bitmap'),
    ('cardboard', 'dumped cardboard street filetype:bitmap'),
    ('shoe', 'discarded shoe beach filetype:bitmap'),
    ('plastic', 'plastic bottle litter grass filetype:bitmap'),
    ('paper', 'newspaper litter street filetype:bitmap'),
    ('food', 'rotting apple compost filetype:bitmap'),
    ('gutter', 'street gutter litter filetype:bitmap'),
    ('river', 'riverbank litter debris filetype:bitmap'),
    ('grass', 'litter on grass park filetype:bitmap'),
    ('trail', 'litter forest trail filetype:bitmap'),
    ('mixed_pile', 'fly tipping rubbish road filetype:bitmap'),
    ('floating', 'floating plastic canal water filetype:bitmap')
]

for label, q in queries:
    url = f'https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srnamespace=6&srlimit=2&srsearch={requests.utils.quote(q)}'
    r = requests.get(url, headers=headers, verify=False, timeout=10)
    items = r.json().get('query', {}).get('search', [])
    print(f'{label}:', items[0]['title'] if items else 'NONE')
