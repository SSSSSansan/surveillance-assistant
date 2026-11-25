"""T20, шаг 1. Выбрать второе видео MEVA: скачать кандидатов, прогнать ВАШ cv/sample_snapshots.py
(те же срезы каждые 30 с, тот же YOLO11s, conf 0.3) и выбрать клип, где больше всего людей.

    python cv/rank_meva.py --out data/candidates --gpu 1
    python cv/rank_meva.py --out data/candidates --keys g421_0307_1720 g421_0307_1735

Что делает: для каждого кандидата скачивает видео (boto3 из cv/meva_fetch.py; если boto3 нет -
обычная ссылка, без AWS CLI), запускает sample_snapshots.py -> <out>/<ключ>/snapshots.json,
печатает таблицу: среднее и максимум людей на срез, медиану высоты рамки.

Кандидаты: школьные камеры G421 (холл, как на этапе 2) и G420 (вход). Ссылки взяты из чужого
публичного репозитория, где эти клипы уже использовали; видео этапа 2 (2018-03-15 15:00, G421)
сюда НЕ входит. Чтобы найти другие клипы: python cv/meva_fetch.py find school.G421

--gpu N ставит CUDA_VISIBLE_DEVICES (на общем сервере не занимай чужую карту).
Каждый клип ~100-150 МБ; перед запуском проверь место на диске.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = "drops-123-r13/"
CANDIDATES = {
    "g421_0307_1650": "2018-03-07/16/2018-03-07.16-50-00.16-55-00.school.G421.r13.avi",
    "g421_0307_1720": "2018-03-07/17/2018-03-07.17-20-00.17-25-00.school.G421.r13.avi",
    "g421_0307_1735": "2018-03-07/17/2018-03-07.17-35-00.17-40-00.school.G421.r13.avi",
    "g420_0309_1010": "2018-03-09/10/2018-03-09.10-10-01.10-15-01.school.G420.r13.avi",
    "g420_0307_1720": "2018-03-07/17/2018-03-07.17-20-01.17-25-01.school.G420.r13.avi",
}


def download(key, out_dir):
    rel = CANDIDATES[key]
    dst = Path(out_dir) / rel.split("/")[-1]  # имя файла как в MEVA: из него sample_snapshots берёт video_id и камеру
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"  {key}: уже скачано")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(HERE))
        import meva_fetch  # ваш скрипт; импорт создаёт boto3-клиент
        meva_fetch.get(PREFIX + rel, str(dst))
    except ImportError:
        url = f"https://mevadata-public-01.s3.amazonaws.com/{PREFIX}{rel}"
        print(f"  {key}: boto3 нет, качаю по ссылке {url}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {key}: не скачалось ({e}). Скачай вручную и положи как {dst}")
            return None
    except Exception as e:  # noqa: BLE001
        print(f"  ! {key}: не скачалось ({e}). Скачай вручную и положи как {dst}")
        return None
    return dst


def stats(snap_path):
    d = json.loads(Path(snap_path).read_text(encoding="utf-8"))
    counts = [s["people_count"] for s in d["snapshots"]]
    hs = sorted(b[3] - b[1] for s in d["snapshots"] for b in s.get("boxes", []))
    return {"video_id": d["video_id"], "snapshots": len(counts), "people_total": sum(counts),
            "mean": sum(counts) / max(1, len(counts)), "max": max(counts, default=0),
            "median_h": hs[len(hs) // 2] if hs else 0, "min_h": hs[0] if hs else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="data/candidates")
    ap.add_argument("--keys", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--gpu", help="номер видеокарты (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in CANDIDATES.items():
            print(f"{k:<16} {PREFIX}{v}")
        return

    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    table = []
    for k in a.keys:
        if k not in CANDIDATES:
            print(f"неизвестный ключ {k}, см. --list")
            continue
        video = download(k, Path(a.out) / "video")
        if video is None:
            continue
        sd = Path(a.out) / k
        if not (sd / "snapshots.json").exists():
            r = subprocess.run([sys.executable, str(HERE / "sample_snapshots.py"), str(video),
                                "--out-dir", str(sd), "--weights", a.weights], env=env)
            if r.returncode != 0:
                print(f"  ! {k}: sample_snapshots.py упал, пропуск")
                continue
        table.append({"key": k, "dir": str(sd), **stats(sd / "snapshots.json")})

    table.sort(key=lambda r: -r["people_total"])
    print("\nкандидат          срезов  людей всего  среднее  макс  медиана h, px  мин h, px")
    for r in table:
        print(f"{r['key']:<16}  {r['snapshots']:5d}  {r['people_total']:10d}  {r['mean']:7.1f}  "
              f"{r['max']:4d}  {r['median_h']:12.0f}  {r['min_h']:9.0f}")
    if table:
        best = table[0]
        print(f"\nВЫБОР: {best['key']}  (snapshots.json: {best['dir']}/snapshots.json)")
        print("ориентир этапа 2: 159 людей на 11 срезах, медиана высоты 263 px")
        if best["people_total"] < 120:
            print("  ! меньше ~120 людей: для T20 нужно ~150 кропов; можно взять второй клип и склеить эталон из двух.")
        if best["median_h"] < 100:
            print("  ! люди мелкие (<100 px): цвет будет определяться хуже, запиши это в отчёт.")
        Path(a.out, "ranking.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print("дальше: python -m cv.crop_people --snapshots <dir>/snapshots.json --out <dir>/crops")


if __name__ == "__main__":
    main()
