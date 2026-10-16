import re
from pathlib import Path

html_path = (Path(__file__).resolve().parents[2] / "artifacts" / "scratch")
text = html_path.read_text(encoding="utf-8")

files = re.findall(r'"filename":"([^"]+)"', text)
print(f"Total files found in html: {len(files)}")
for fn in files[:40]:
    print(" ", fn)

yaml_files = [fn for fn in files if fn.endswith(('.yaml', '.yml', '.txt', '.json', '.zip'))]
print("\nSpecial non-image/non-annotation files:")
for fn in yaml_files:
    if not (fn.startswith("syn_") and fn.endswith(".txt")):
        print(" ", fn)
