import json

with open('reports/qa_acceptance/manifest_hinh_anh_rac.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for i, f in enumerate(data['files'], 1):
    w = f['pil_decode']['info']['width']
    h = f['pil_decode']['info']['height']
    fmt = f['pil_decode']['info']['format']
    print(f"{i:02d}: {f['filename']} | {w}x{h} | {fmt} | {f['size_bytes']} bytes")
