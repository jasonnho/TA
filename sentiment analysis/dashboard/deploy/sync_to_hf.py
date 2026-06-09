"""Sync dashboard dari repo skripsi -> clone deploy `hf_space_repo`.

Menyalin app.py + data (yang dipakai dashboard) + assets ke folder hf_space_repo,
supaya tidak perlu edit manual di dua tempat. app.py sengaja dibuat IDENTIK di
dua repo (auto-deteksi lokasi data), jadi cukup di-copy utuh.

Alur kerja:
  1. edit + tes di repo skripsi (sentiment analysis/dashboard/)
  2. jalankan skrip ini
  3. cd <hf_space_repo> ; git add . ; git commit -m "..." ; git push

Lokasi hf_space_repo default = sibling dari TA_Notebook (mis. D:/Kuliah/TA/hf_space_repo).
Override: env HF_SPACE_REPO atau argumen --hf-repo <path>.

Catatan: Dockerfile, requirements.txt, README HF TIDAK disalin (jarang berubah &
sudah ada di hf_space_repo). Skrip ini hanya menyalin yang berubah saat dev:
app.py, data, assets. File yang DIHAPUS di sumber tidak otomatis dihapus di tujuan.

Jalankan:
  D:/miniconda3/envs/ta_nlp/python.exe "sentiment analysis/dashboard/deploy/sync_to_hf.py"
"""
import os
import sys
import shutil
import filecmp
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))   # .../dashboard/deploy
DASH_DIR = os.path.dirname(HERE)                     # .../dashboard
SA_DIR = os.path.dirname(DASH_DIR)                   # .../sentiment analysis
PROJECT = os.path.dirname(SA_DIR)                    # TA_Notebook
DEFAULT_HF = os.path.join(os.path.dirname(PROJECT), "hf_space_repo")

# File data yang dibaca dashboard (suffix default, tanpa --mtl-only)
DATA_FILES = [
    "sentiment_labeled.csv",
    "tfidf_keywords.json",
    "aspect_summary.json",
    "insights_summary.json",
    "opinion_words.json",
    "phrase_words.json",
]


def _copy_if_changed(src, dst, label):
    """Salin hanya jika isi beda (hindari 'modified' palsu di git akibat mtime)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
        return False
    shutil.copy2(src, dst)
    print(f"  {label}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-repo", default=os.environ.get("HF_SPACE_REPO", DEFAULT_HF),
                    help="path ke clone hf_space_repo")
    args = ap.parse_args()
    hf = os.path.abspath(args.hf_repo)

    if not os.path.isdir(os.path.join(hf, ".git")):
        sys.exit(f"ERROR: '{hf}' bukan git repo.\n"
                 f"Set lokasi via --hf-repo <path> atau env HF_SPACE_REPO.")

    print(f"Sync -> {hf}\n")
    n = 0

    # 1. app.py (identik di dua repo)
    n += _copy_if_changed(os.path.join(DASH_DIR, "app.py"),
                          os.path.join(hf, "app.py"), "app.py")

    # 2. data
    for fn in DATA_FILES:
        src = os.path.join(SA_DIR, "data", fn)
        if os.path.exists(src):
            n += _copy_if_changed(src, os.path.join(hf, "data", fn), f"data/{fn}")
        else:
            print(f"  (lewati, tidak ada: {fn})")

    # 3. assets (style.css + images/) — merge per-file, tidak menghapus file lama
    src_assets = os.path.join(DASH_DIR, "assets")
    for root, _, files in os.walk(src_assets):
        for f in files:
            s = os.path.join(root, f)
            rel = os.path.relpath(s, src_assets)
            n += _copy_if_changed(s, os.path.join(hf, "assets", rel), f"assets/{rel}")

    print(f"\nSelesai ({n} file berubah). Lanjut:" if n else "\nSelesai (tidak ada perubahan).")
    if n:
        print(f"  cd \"{hf}\"\n  git add . && git commit -m \"update dashboard\" && git push")


if __name__ == "__main__":
    main()
