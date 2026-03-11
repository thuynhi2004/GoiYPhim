"""
Script tải dữ liệu MovieLens 100K về thư mục data/ml-100k/
Chạy: python download_data.py
"""
import urllib.request
import zipfile
import os
import sys


def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    os.makedirs("data", exist_ok=True)
    dest = os.path.join("data", "ml-100k.zip")

    def _progress(count, block_size, total_size):
        pct = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\r  Tiến độ tải: {pct}%  ")
        sys.stdout.flush()

    print("Đang tải dữ liệu MovieLens 100K (~5 MB)...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print("\nGiải nén...")
    with zipfile.ZipFile(dest, "r") as z:
        z.extractall("data")
    os.remove(dest)
    print("✓ Hoàn tất! Dữ liệu lưu tại: data/ml-100k/")


if __name__ == "__main__":
    if os.path.exists(os.path.join("data", "ml-100k", "u.data")):
        print("✓ Dữ liệu đã tồn tại tại data/ml-100k/ — không cần tải lại.")
    else:
        download_movielens()
