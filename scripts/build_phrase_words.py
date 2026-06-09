"""Build phrase-based word/bigram data for the dashboard word cloud & bigram chart.

Masalah lama: word cloud & bigram mengambil SELURUH teks review yang overall-nya
Negatif, lalu hitung frekuensi kata → kata positif "bagus" bocor ke cloud negatif
(karena review "pemandangan bagus tapi tiket mahal" overall-nya Negatif).

Solusi (phrase-based, hybrid MTL + mdhugol):
  1. Pecah tiap review (teks MENTAH, masih punya koma + "tapi/namun") jadi klausa.
  2. Tentukan sentimen TIAP klausa:
       - klausa menyebut aspek yg di-tag model → pakai sentimen per-aspek
         (kolom aspect_sentiments_str = MTL bila ada span, else overall) ← presisi
       - selainnya → klasifikasi klausa pakai mdhugol  ← coverage ~100%
  3. Kumpulkan kata (di-stem, sama seperti cleaned_review_stemmed) per
     (desa, sentimen) → unigram + bigram counts → simpan JSON.

Output: sentiment analysis/data/phrase_words.json (+ _mtl_only.json).
Dashboard (app.py) tinggal baca JSON ini — tidak menjalankan model.

Catatan: skrip ini mencerminkan cell baru di SA_AutoLabel.ipynb (logika sama),
dipakai untuk regenerate data tanpa harus menjalankan ulang seluruh notebook.

Jalankan:  D:/miniconda3/envs/ta_nlp/python.exe scripts/build_phrase_words.py
"""
import os
import re
import sys
import json
from collections import Counter, defaultdict

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # skrip ada di scripts/, data di root proyek
DATA_DIR = os.path.join(PROJECT_ROOT, "sentiment analysis", "data")
LEX_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "colloquial-indonesian-lexicon.csv")

TOP_WORDS = 120   # disimpan per (desa, sentimen) — cukup untuk merge multi-desa
TOP_BIGRAMS = 60

# Pengaman polaritas: routing per-klausa sudah benar untuk mayoritas, tapi review
# "run-on" tanpa pemisah (mis. "pemandangan bagus tiketnya mahal" tanpa koma/"tapi")
# masih bisa membocorkan kata. Filter ini membuang kata yang SANGAT dominan di satu
# polaritas dari polaritas lain. RATE dinormalisasi per kelas (korpus ~86% positif,
# jadi semua kata umum condong positif kalau pakai count mentah). Kata muncul di
# sentimen s hanya jika rate-nya di s >= FRAC * rate puncak antar-sentimen.
DOMINANCE_FRAC = 0.25
MIN_TOTAL = 10            # kata terlalu jarang tidak difilter (low impact)
# Bigram lebih spesifik: keluhan asli ('tiket mahal') hampir tak muncul di positif,
# sedangkan frasa positif yang bocor ('desa bersih', dari 'desa kurang bersih' yg
# negasinya kebuang stopword) muncul JAUH lebih sering di positif. Jadi pakai rasio
# count mentah: tampilkan bigram di sentimen s hanya jika count-nya >= peak/BG_RATIO.
BG_RATIO = 2.5

# ── Cleaning setup (mirror Clean_Reviews.ipynb cell 9 + SA_AutoLabel cell 6) ──
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_stemmer = StemmerFactory().create_stemmer()
_sastrawi_sw = set(StopWordRemoverFactory().get_stop_words())
EXTRA_STOPWORDS = {
    "google", "maps", "map", "review", "reviews", "bintang", "star", "stars",
    "ya", "yah", "dong", "deh", "sih", "nih", "tuh", "nah", "mah", "loh", "lah",
    "kok", "kan", "pun", "kah", "toh",
    "udah", "udh", "aja", "doang", "doank", "seh", "bgt", "gtu", "gitu", "gini",
}
STOPWORDS = _sastrawi_sw | EXTRA_STOPWORDS

_slang_df = pd.read_csv(LEX_PATH, usecols=["slang", "formal"]).dropna().drop_duplicates(subset="slang")
SLANG_DICT = dict(zip(_slang_df["slang"].str.lower().str.strip(),
                      _slang_df["formal"].str.lower().str.strip()))

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D]+",
    flags=re.UNICODE,
)


def normalize_slang(text):
    return " ".join(SLANG_DICT.get(w, w) for w in text.split())


def clean_text(text):
    """Sama dengan Clean_Reviews.clean_text — TANPA stemming."""
    text = str(text).lower()
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_slang(text)
    text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text


_stem_cache = {}


def _stem_word(w):
    s = _stem_cache.get(w)
    if s is None:
        s = _stemmer.stem(w)
        _stem_cache[w] = s
    return s


def display_words(clause_norm):
    """Kata yang ditampilkan: di-clean + di-stem, sama bentuk dgn cleaned_review_stemmed."""
    cleaned = clean_text(clause_norm)
    out = []
    for w in cleaned.split():
        s = _stem_word(w)
        if len(s) > 2 and s.isalpha():
            out.append(s)
    return out


# ── Aspect keywords (mirror SA_AutoLabel cell 24) ──
ASPECT_KEYWORDS = {
    "Pemandangan & Alam": ["pemandangan", "alam", "sawah", "gunung", "sungai", "mangrove",
                            "pantai", "air terjun", "pohon", "hutan", "danau", "laut",
                            "bukit", "taman", "kebun", "persawahan", "terasering",
                            "sunrise", "sunset", "view", "landscape"],
    "Kebersihan": ["bersih", "kotor", "sampah", "toilet", "kamar mandi", "wc",
                    "kebersihan", "rapi", "jorok", "kumuh", "terawat"],
    "Pelayanan": ["pelayanan", "ramah", "warga", "masyarakat", "guide",
                   "pengelola", "staff", "petugas", "penjaga", "pelayan", "layanan", "sambutan"],
    "Fasilitas": ["fasilitas", "parkir", "jembatan", "gazebo", "wifi", "spot foto", "kolam",
                   "wahana", "mushola", "playground", "ayunan", "gardu pandang", "homestay",
                   "penginapan", "villa", "cottage", "outbound"],
    "Harga & Tiket": ["harga", "tiket", "murah", "mahal", "biaya", "gratis",
                       "bayar", "tarif", "retribusi", "terjangkau", "worth"],
    "Kuliner": ["makanan", "minuman", "kopi", "warung", "restoran", "kuliner", "menu", "cafe",
                 "makan", "minum", "jajanan", "oleh-oleh", "souvenir", "snack", "kedai", "rumah makan"],
    "Suasana": ["suasana", "tenang", "sejuk", "nyaman", "asri", "adem",
                 "damai", "segar", "teduh", "dingin", "udara", "santai", "rileks"],
    "Akses & Lokasi": ["lokasi", "akses", "jarak", "jauh", "dekat", "strategis",
                        "jalan", "rute", "transportasi", "kendaraan", "motor", "mobil", "bus", "ojek"],
    "Budaya & Tradisi": ["budaya", "adat", "tradisi", "rumah adat", "upacara", "kesenian",
                          "tari", "musik", "gamelan", "batik", "kerajinan", "sejarah", "heritage"],
}


def clause_category(clause, amap):
    """Kategori aspek untuk klausa. Utamakan kategori yang ADA di peta sentimen
    review (amap) supaya kita pakai sentimen aspek yang relevan; kalau tidak ada
    yang cocok di amap, kembalikan match pertama (→ amap.get None → fallback mdhugol)."""
    t = clause.lower()
    first = None
    for cat, kws in ASPECT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            if cat in amap:
                return cat
            if first is None:
                first = cat
    return first


# ── Clause segmentation ──
_CONJ = ("tapi", "namun", "tetapi", "cuma", "cuman", "sayang", "sedangkan",
         "walaupun", "walau", "meskipun", "meski", "kecuali", "hanya", "selain")
_SPLIT_RE = re.compile(r"[.,;!?\n]+|\b(?:" + "|".join(_CONJ) + r")\b")


def segment_clauses(text):
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
    return [p for p in parts if p]


# ── mdhugol (sentence-level) ──
HF_LABEL_MAP = {"LABEL_0": "Positif", "LABEL_1": "Netral", "LABEL_2": "Negatif"}


def load_mdhugol():
    import torch
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    name = "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    device = 0 if torch.cuda.is_available() else -1
    print(f"  mdhugol device: {'cuda' if device == 0 else 'cpu'}")
    return pipeline("sentiment-analysis", model=model, tokenizer=tok, device=device)


def build(csv_path, out_path, use_mdhugol_fallback=True):
    name = os.path.basename(csv_path)
    print(f"\n{'='*64}\nBuild phrase data: {name}  (mdhugol fallback={use_mdhugol_fallback})\n{'='*64}")
    df = pd.read_csv(csv_path)

    aspects_list = df["aspects_str"].fillna("").str.split("|")
    asent_list = df["aspect_sentiments_str"].fillna("").str.split("|")
    amaps = [dict(zip(a, s)) for a, s in zip(aspects_list, asent_list)]

    # Pass 1: segmentasi + tentukan sentimen via peta aspek; kumpulkan klausa
    # yang butuh mdhugol untuk diklasifikasi sekaligus (batch).
    records = []  # (desa, clause, sent_or_None)
    need_mdhugol = []
    n_reviews = 0
    for desa, review, amap in zip(df["nama desa wisata"], df["review"].fillna(""), amaps):
        n_reviews += 1
        text = normalize_slang(str(review).lower())
        for clause in segment_clauses(text):
            cat = clause_category(clause, amap)
            sent = amap.get(cat) if cat else None
            if sent not in ("Positif", "Negatif", "Netral"):
                sent = None
            records.append([desa, clause, sent])
            if sent is None and use_mdhugol_fallback:
                need_mdhugol.append(clause)

    n_aspect = sum(1 for r in records if r[2] is not None)
    print(f"  reviews: {n_reviews:,} | klausa: {len(records):,} | "
          f"via aspek-MTL: {n_aspect:,} ({n_aspect/max(len(records),1)*100:.1f}%)")

    # Pass 2: mdhugol pada klausa unik yang belum bersentimen
    sent_by_clause = {}
    if use_mdhugol_fallback and need_mdhugol:
        uniq = [c for c in dict.fromkeys(need_mdhugol) if c.strip()]
        print(f"  mdhugol pada {len(uniq):,} klausa unik ...")
        clf = load_mdhugol()
        preds = clf(uniq, batch_size=64, truncation=True, max_length=128)
        sent_by_clause = {c: HF_LABEL_MAP[p["label"]] for c, p in zip(uniq, preds)}
        del clf
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Pass 3: agregasi count per (desa, sentimen) + bucket "Semua"
    words_ctr = defaultdict(lambda: defaultdict(Counter))   # desa -> sent -> Counter
    bg_ctr = defaultdict(lambda: defaultdict(Counter))
    n_labeled = 0
    for desa, clause, sent in records:
        if sent is None:
            sent = sent_by_clause.get(clause)
            if sent is None:
                continue
        words = display_words(clause)
        if not words:
            continue
        n_labeled += 1
        for d in (desa, "Semua"):
            words_ctr[d][sent].update(words)
            bg_ctr[d][sent].update(f"{a} {b}" for a, b in zip(words, words[1:]))

    print(f"  klausa berkontribusi (punya sentimen + kata): {n_labeled:,} "
          f"({n_labeled/max(len(records),1)*100:.1f}%)")

    # Pengaman polaritas (dari bucket 'Semua', diterapkan ke semua desa)
    suppressed = _polarity_suppression(words_ctr["Semua"])
    bg_suppressed = _bigram_suppression(bg_ctr["Semua"])

    # Serialize top-N (setelah filter polaritas)
    out = {}
    for desa in words_ctr:
        out[desa] = {}
        for sent in ("Positif", "Negatif", "Netral"):
            sup = suppressed[sent]
            bsup = bg_suppressed[sent]
            w = [(k, v) for k, v in words_ctr[desa][sent].most_common() if k not in sup][:TOP_WORDS]
            b = [(k, v) for k, v in bg_ctr[desa][sent].most_common()
                 if k not in bsup and not (set(k.split()) & sup)][:TOP_BIGRAMS]
            if w or b:
                out[desa][sent] = {
                    "words": [[k, int(v)] for k, v in w],
                    "bigrams": [[k, int(v)] for k, v in b],
                }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {out_path}  ({len(out)} desa keys)")

    _sanity(out)
    return out


def _polarity_suppression(words_ctr_semua):
    """Tentukan, per sentimen, kata mana yang harus DISEMBUNYIKAN karena sangat
    dominan di polaritas lain (rate dinormalisasi per kelas). Keputusan diambil
    dari bucket 'Semua' (sampel terbesar) lalu diterapkan ke semua desa."""
    sents = ("Positif", "Negatif", "Netral")
    totals = {s: (sum(words_ctr_semua[s].values()) or 1) for s in sents}
    allw = set()
    for s in sents:
        allw |= set(words_ctr_semua[s])
    suppressed = {s: set() for s in sents}
    for w in allw:
        counts = {s: words_ctr_semua[s].get(w, 0) for s in sents}
        if sum(counts.values()) < MIN_TOTAL:
            continue
        rates = {s: counts[s] / totals[s] for s in sents}
        peak = max(rates.values())
        if peak <= 0:
            continue
        for s in sents:
            if rates[s] < DOMINANCE_FRAC * peak:
                suppressed[s].add(w)
    n = sum(len(v) for v in suppressed.values())
    print(f"  filter polaritas: {n} (word,sentimen) disembunyikan "
          f"(mis. bagus@Negatif={'ya' if 'bagus' in suppressed['Negatif'] else 'tidak'})")
    return suppressed


def _bigram_suppression(bg_ctr_semua):
    """Bigram disembunyikan dari sentimen s bila count-nya < peak/BG_RATIO
    (count mentah antar-sentimen). Membuang frasa positif yang bocor ke negatif
    seperti 'desa bersih' (461 di positif vs 32 di negatif)."""
    sents = ("Positif", "Negatif", "Netral")
    allb = set()
    for s in sents:
        allb |= set(bg_ctr_semua[s])
    sup = {s: set() for s in sents}
    for b in allb:
        cnt = {s: bg_ctr_semua[s].get(b, 0) for s in sents}
        peak = max(cnt.values())
        if peak <= 0:
            continue
        for s in sents:
            if cnt[s] * BG_RATIO < peak:
                sup[s].add(b)
    print(f"  filter bigram   : {sum(len(v) for v in sup.values())} (bigram,sentimen) disembunyikan "
          f"(mis. 'desa bersih'@Negatif={'ya' if 'desa bersih' in sup['Negatif'] else 'tidak'})")
    return sup


def _sanity(out):
    for desa in ("Semua", "Buluh Cina"):
        if desa not in out:
            continue
        print(f"\n  --- SANITY: {desa} ---")
        for sent in ("Positif", "Negatif"):
            words = [w for w, _ in out[desa].get(sent, {}).get("words", [])[:15]]
            print(f"    {sent:8s}: {words}")
        neg_words = {w for w, _ in out[desa].get("Negatif", {}).get("words", [])}
        flag = "BAGUS MASIH ADA DI NEGATIF!" if "bagus" in neg_words else "ok ('bagus' tidak di Negatif)"
        print(f"    => {flag}")


if __name__ == "__main__":
    # Default (hybrid: MTL/overall per-aspek + mdhugol fallback) → coverage ~100%
    build(os.path.join(DATA_DIR, "sentiment_labeled.csv"),
          os.path.join(DATA_DIR, "phrase_words.json"),
          use_mdhugol_fallback=True)

    # MTL-only (tanpa fallback mdhugol — hanya klausa ber-aspek MTL)
    mtl_csv = os.path.join(DATA_DIR, "sentiment_labeled_mtl_only.csv")
    if os.path.exists(mtl_csv):
        build(mtl_csv,
              os.path.join(DATA_DIR, "phrase_words_mtl_only.json"),
              use_mdhugol_fallback=False)
    print("\nDONE.")
