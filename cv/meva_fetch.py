"""T2. Поиск и скачивание видео MEVA (открытый S3, без регистрации и без AWS CLI).

  pip install boto3
  python cv/meva_fetch.py find school.G421                 # найти файлы по подстроке в имени
  python cv/meva_fetch.py get <ключ из find> data/meva/    # скачать одно видео
  python cv/meva_fetch.py annotations                      # клонировать репозиторий разметки (нужен git)

Видео и разметка кладутся в data/, эта папка не попадает в git.
"""
import os
import subprocess
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "mevadata-public-01"
ANN_REPO = "https://gitlab.kitware.com/meva/meva-data-repo.git"

s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))


def find(substr):
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if substr in k and k.endswith((".avi", ".mp4")):
                print(f"{obj['Size'] / 1e6:8.1f} MB  {k}")
                n += 1
    print("найдено:", n)


def get(key, dest):
    if dest.endswith(("/", "\\")) or os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
        dest = os.path.join(dest, os.path.basename(key))
    else:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    s3.download_file(BUCKET, key, dest)
    print("скачано:", dest)


def annotations():
    subprocess.run(["git", "clone", "--depth", "1", ANN_REPO, "data/meva-data-repo"], check=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "find":
        find(sys.argv[2])
    elif cmd == "get":
        get(sys.argv[2], sys.argv[3])
    elif cmd == "annotations":
        annotations()
    else:
        print(__doc__)