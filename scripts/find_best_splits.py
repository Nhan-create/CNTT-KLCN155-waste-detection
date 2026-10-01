import random
from collections import Counter
import pandas as pd

df = pd.read_csv('data/audit/real_detection_source_manifest.csv')
app = df[df['review_status'] == 'APPROVED']

imgs = []
for _, r in app.iterrows():
    lbl_p = 'data/detection/' + r['relative_label_path'].replace('\\', '/')
    classes = set()
    with open(lbl_p, 'r') as f:
        for line in f:
            classes.add(int(line.split()[0]))
    imgs.append({
        'fn': r['filename'],
        'grp': r['group_id'],
        'cls': sorted(list(classes)),
        'boxes': r['num_boxes']
    })

grps = [x['grp'] for x in imgs]
grp2img = {x['grp']: x for x in imgs}

random.seed(42)
best_combo = None
best_score = -9999

for _ in range(100000):
    shuffled = grps.copy()
    random.shuffle(shuffled)
    val_grps = set(shuffled[:5])
    test_grps = set(shuffled[5:10])
    train_grps = set(shuffled[10:])

    def get_classes(g_set):
        s = set()
        for g in g_set:
            s.update(grp2img[g]['cls'])
        return s

    c_train = get_classes(train_grps)
    c_val = get_classes(val_grps)
    c_test = get_classes(test_grps)

    common = {1, 2, 4, 5, 6, 7, 9}
    if not common.issubset(c_train) or not common.issubset(c_val) or not common.issubset(c_test):
        continue

    b_train = sum(grp2img[g]['boxes'] for g in train_grps)
    b_val = sum(grp2img[g]['boxes'] for g in val_grps)
    b_test = sum(grp2img[g]['boxes'] for g in test_grps)

    # Put the massive 90-box image taco_1107 (grp 1367) in train
    if 'real_grp_1367' not in train_grps:
        continue

    # Battery (grp 1354, 1387): 1 in train, 1 in test (val gets 0)
    # Shoes (grp 1359, 1388): 1 in train, 1 in val (test gets 0)
    if not (0 in c_train and 0 in c_test and 8 in c_train and 8 in c_val):
        continue

    score = - abs(b_val - 45) - abs(b_test - 45)
    if score > best_score:
        best_score = score
        best_combo = (train_grps, val_grps, test_grps, b_train, b_val, b_test)

if best_combo:
    train_grps, val_grps, test_grps, b_tr, b_va, b_te = best_combo
    print(f"Found best combo (score {best_score}):")
    print(f"Train ({len(train_grps)} groups, {b_tr} boxes):")
    for g in sorted(train_grps):
        print(f"  {grp2img[g]['fn']} ({g}): boxes={grp2img[g]['boxes']}, cls={grp2img[g]['cls']}")
    print(f"Val ({len(val_grps)} groups, {b_va} boxes):")
    for g in sorted(val_grps):
        print(f"  {grp2img[g]['fn']} ({g}): boxes={grp2img[g]['boxes']}, cls={grp2img[g]['cls']}")
    print(f"Test ({len(test_grps)} groups, {b_te} boxes):")
    for g in sorted(test_grps):
        print(f"  {grp2img[g]['fn']} ({g}): boxes={grp2img[g]['boxes']}, cls={grp2img[g]['cls']}")
else:
    print("No valid combination found.")
