import os

path = input("Dán đường dẫn thư mục: ").strip().strip("'\"")

for root, dirs, files in os.walk(path):
    print(root)
    for f in files:
        print("  ", f)