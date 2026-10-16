import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

data = json.load(open('reports/qa_acceptance/suspicious_gt_audit_table.json', encoding='utf-8'))
print(f"{'Severity':9s} | {'Filename':25s} | {'Object ID':12s} | {'Class':10s} | {'Area %':7s} | {'Issue Category'}")
print("-" * 105)
for it in data['suspicious_items']:
    print(f"{it['severity']:9s} | {it['filename'][:25]:25s} | {it['object_id']:12s} | {it['class_name']:10s} | {it['area_pct']:6.2f}% | {it['issue_category']}")
