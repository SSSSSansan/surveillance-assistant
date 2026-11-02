"""T2. Поиск и скачивание видео MEVA (открытый S3, без регистрации).

Нужен AWS CLI: https://aws.amazon.com/cli/  (аккаунт не нужен, ключ --no-sign-request).

  python cv/meva_fetch.py find school.G421          # найти файлы по подстроке в имени
  python cv/meva_fetch.py get <ключ из find> data/meva/   # скачать одно видео
  python cv/meva_fetch.py annotations               # клонировать репозиторий разметки

Видео и разметка кладутся в data/, эта папка не попадает в git.
"""
import subprocess
import sys

BUCKET = "s3://mevadata-public-01/"
ANN_REPO = "https://gitlab.kitware.com/meva/meva-data-repo.git"


def find(substr):
    out = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", "--recursive", BUCKET],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if substr in line and line.endswith((".avi", ".mp4")):
            print(line)


def get(key, dest):
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", BUCKET + key, dest], check=True)


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