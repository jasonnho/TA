# Cara Update & Deploy Dashboard

Dashboard di-deploy ke **Hugging Face Spaces**:
`https://jaassoonn1-dashboard-sentimen-desa-wisata.hf.space`

Ada **dua repo**:
- **Source** (pengembangan & tes lokal): repo skripsi `TA_Notebook`, folder
  `sentiment analysis/dashboard/`.
- **Deploy** (di-push ke HF): clone terpisah `hf_space_repo`
  (default `D:\Kuliah\TA\hf_space_repo`), remote-nya ke huggingface.co.

> `app.py` **identik** di dua repo — ia meng-auto-deteksi lokasi data (`./data` saat
> deploy, `../data` di repo skripsi). Jadi tidak perlu edit manual saat menyalin.

## Alur update (utama — pakai skrip sync)

```powershell
# 1. Edit & TES di repo skripsi
cd D:\Kuliah\TA\TA_Notebook
D:\miniconda3\envs\ta_nlp\python.exe "sentiment analysis\dashboard\app.py"   # http://localhost:8050

# 2. Sync ke clone deploy (copy app.py + data + assets yang berubah saja)
D:\miniconda3\envs\ta_nlp\python.exe "sentiment analysis\dashboard\deploy\sync_to_hf.py"

# 3. Push ke HF
cd D:\Kuliah\TA\hf_space_repo
git add . ; git commit -m "deskripsi perubahan" ; git push

# 4. (arsip) commit repo skripsi ke GitHub
```

Setelah `git push` (langkah 3), HF otomatis rebuild (~2–4 menit). Saat login git:
**username** = username HF, **password** = **access token Write** (bukan password akun).

> Lokasi `hf_space_repo` default = sibling dari `TA_Notebook`. Kalau beda, jalankan
> sync dengan `--hf-repo <path>` atau set env `HF_SPACE_REPO`.

## Skenario

### Nambah / ganti gambar desa
1. Taruh file `.jpg/.jpeg/.png/.webp` di **repo skripsi**:
   `sentiment analysis/dashboard/assets/images/<nama desa>/`.
   ⚠️ Nama folder = **nama desa huruf kecil**, persis kolom `nama desa wisata` di
   `sentiment_labeled.csv` (mis. `kampung blekok`). Salah nama → gambar tak muncul
   (app men-scan folder otomatis saat start).
2. Jalankan `sync_to_hf.py` → push (langkah 2–3 di atas). Gambar otomatis masuk
   **Git LFS** (sudah diatur di `.gitattributes` hf_space_repo).

### Ubah data (hasil regenerate)
Regenerate JSON via `scripts/build_opinion_words.py` / `scripts/build_phrase_words.py`
di repo skripsi → output masuk `sentiment analysis/data/` → `sync_to_hf.py` ikut
menyalinnya → push.

### Ubah kode tampilan (`app.py`)
Edit `sentiment analysis/dashboard/app.py` di repo skripsi, tes lokal, lalu
`sync_to_hf.py` + push. **Tidak perlu** edit `app.py` di hf_space_repo — sync
menyalinnya utuh (file-nya identik).

## Catatan

- File biner (gambar) **wajib** lewat Git LFS di HF. Sudah diatur di `.gitattributes`
  hf_space_repo, jadi cukup `git add` biasa.
- `sync_to_hf.py` hanya menyalin file yang **isinya berubah**, dan **tidak menghapus**
  file yang dihapus di sumber (kalau perlu hapus, lakukan manual di hf_space_repo).
- `Dockerfile`, `requirements.txt`, dan README HF jarang berubah → tidak ikut di-sync;
  ubah langsung di hf_space_repo bila perlu (salinan dokumentasinya ada di folder
  `deploy/` repo skripsi).
- Free tier HF bisa "tidur" setelah lama tanpa pengunjung; kunjungan berikutnya
  membangunkannya otomatis (cold start beberapa detik).
