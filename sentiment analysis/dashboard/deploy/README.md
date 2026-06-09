# Deploy config — Dashboard ke Hugging Face Spaces

Folder ini berisi **berkas konfigurasi deploy** untuk men-deploy dashboard
(`../app.py`) ke Hugging Face Spaces. Disimpan di repo skripsi sebagai dokumentasi
supaya setup deploy ikut terversi.

> **Source dashboard** ada di repo ini (`sentiment analysis/dashboard/` + data di
> `sentiment analysis/data/` + gambar di `assets/`). File di sini hanya tambahan
> khusus deploy — **bukan duplikat** data/gambar.

## Isi

| File | Fungsi |
|------|--------|
| `sync_to_hf.py` | **Skrip sync**: copy `app.py` + data + assets (yang berubah) dari repo skripsi → `hf_space_repo` |
| `Dockerfile` | Build container + jalankan `gunicorn app:server` di port 7860 |
| `requirements.txt` | Dependency **ramping** khusus dashboard (tanpa torch) |
| `.dockerignore` | Exclude `__pycache__`, dll dari build |
| `hf_space_README.md` | README dengan header khusus HF Space (`sdk: docker`, `app_port: 7860`) |
| `CARA_UPDATE.md` | Panduan update gambar/data/kode + cara push ke HF |

## Repo deploy yang sebenarnya

Yang benar-benar di-push ke HF adalah **repo git terpisah**:

```
D:\Kuliah\TA\hf_space_repo\          # remote → huggingface.co/spaces/jaassoonn1/dashboard-sentimen-desa-wisata
```

HF Space wajib jadi git repo tersendiri (remote-nya ke huggingface.co) dan gambar
harus lewat Git LFS — makanya tidak digabung ke repo skripsi ini.

## `app.py` identik di dua repo

`app.py` di `hf_space_repo` **sama persis** dengan `../app.py` di repo ini. Caranya:
`app.py` meng-auto-deteksi lokasi data (`./data` saat deploy, `../data` di repo
skripsi), dan sudah punya `server = app.server` + run block ramah-deploy. Jadi
menyalinnya cukup copy utuh (ditangani `sync_to_hf.py`) — tanpa edit manual.

## Update rutin

```powershell
# edit + tes di repo skripsi, lalu:
D:\miniconda3\envs\ta_nlp\python.exe "sentiment analysis\dashboard\deploy\sync_to_hf.py"
cd D:\Kuliah\TA\hf_space_repo ; git add . ; git commit -m "..." ; git push
```

Detail lengkap: `CARA_UPDATE.md`.

## Cara deploy ulang dari nol (kalau `hf_space_repo` hilang)

```powershell
git clone https://huggingface.co/spaces/jaassoonn1/dashboard-sentimen-desa-wisata hf_space_repo
cd hf_space_repo
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png"
# salin file deploy dari folder ini (Dockerfile, requirements.txt, .dockerignore,
# hf_space_README.md → README.md), lalu isi app.py + data + assets via sync:
D:\miniconda3\envs\ta_nlp\python.exe "<TA_Notebook>\sentiment analysis\dashboard\deploy\sync_to_hf.py"
git add . ; git commit -m "deploy" ; git push
```
