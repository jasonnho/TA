"""Rebuild per-aspect opinion keywords (opinion_words.json) + refresh the
`opinions` fields di insights_summary.json memakai ekstraksi per-KLAUSA.

Masalah lama (SA_AutoLabel.get_opinion_words): mengambil SEMUA kata dari teks
review yang menyebut aspek → (1) kata aspek lain bocor ('makan' di Kebersihan),
(2) kata positif bocor ke daftar Negatif ('bagus' di keluhan).

Solusi (sama dgn _build_phrase_words.py): kata hanya diambil dari KLAUSA yang
menyebut aspek tsb; sentimen = sentimen per-aspek (MTL/overall) dari data; lalu
filter polaritas global membuang kata yang dominan di polaritas lain. Hanya
memakai aspek yang DITANDAI model di review (sentimen sudah ada di data) → tidak
butuh model/GPU.

Output: sentiment analysis/data/opinion_words.json (top 10 / aspek / sentimen)
        + perbarui field 'opinions' di insights_summary.json (top 5).
        (+ varian _mtl_only.* bila CSV-nya ada)

Jalankan:  D:/miniconda3/envs/ta_nlp/python.exe scripts/build_opinion_words.py
"""
import os
import json
from collections import Counter, defaultdict

import pandas as pd

import build_phrase_words as pw  # reuse cleaning/stemming/segmentasi/aspek/filter

DATA_DIR = pw.DATA_DIR
TOP_OPINION = 10   # per (desa, aspek, sentimen) di opinion_words.json
TOP_INSIGHT = 5    # per kartu insight
SENTS = ("Positif", "Negatif", "Netral")

# Kata pembawa-aspek generik / non-opini yang tidak informatif sebagai keyword
# (mirror grup "Generic aspect carriers" + visit/time di STOPWORDS_ID notebook).
# Difilter SETELAH stemming.
_GENERIC_RAW = {
    "tempat", "tempatnya", "wisata", "desa", "lokasi", "pergi", "datang",
    "kunjungi", "kunjungan", "objek", "destinasi", "daerah", "area", "kawasan",
    "spot", "kesini", "kesana", "main", "liburan", "kemarin", "tadi",
    "sekarang", "nanti", "dulu", "banyak", "sedikit", "sebagian", "banget",
}
GENERIC = {pw._stem_word(w) for w in _GENERIC_RAW}


def _aspects_in_clause(clause, amap):
    """Aspek yang (a) ditandai model di review ini & (b) kata kuncinya muncul di klausa."""
    t = clause.lower()
    return [cat for cat, kws in pw.ASPECT_KEYWORDS.items()
            if cat in amap and any(kw in t for kw in kws)]


def build(csv_path, op_out, ins_out):
    name = os.path.basename(csv_path)
    print(f"\n{'='*64}\nRebuild opinion words: {name}\n{'='*64}")
    df = pd.read_csv(csv_path)
    aspects_list = df["aspects_str"].fillna("").str.split("|")
    asent_list = df["aspect_sentiments_str"].fillna("").str.split("|")
    amaps = [dict(zip(a, s)) for a, s in zip(aspects_list, asent_list)]

    # desa -> aspek -> sentimen -> Counter(word)
    ctr = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    # untuk filter polaritas global: sentimen -> Counter(word)  (lintas aspek)
    global_ctr = {s: Counter() for s in SENTS}

    n_reviews = n_clause_hit = 0
    for desa, review, amap in zip(df["nama desa wisata"],
                                  df["review"].fillna(""), amaps):
        n_reviews += 1
        text = pw.normalize_slang(str(review).lower())
        for clause in pw.segment_clauses(text):
            cats = _aspects_in_clause(clause, amap)
            if not cats:
                continue
            words = [w for w in pw.display_words(clause) if w not in GENERIC]
            if not words:
                continue
            n_clause_hit += 1
            for cat in cats:
                sent = amap[cat]
                if sent not in SENTS:
                    continue
                ctr[desa][cat][sent].update(words)
                ctr["Semua"][cat][sent].update(words)
                global_ctr[sent].update(words)

    print(f"  reviews: {n_reviews:,} | klausa ber-aspek berkontribusi: {n_clause_hit:,}")

    # Filter polaritas global (reuse logika phrase script) → {sent: set(word)}
    suppressed = pw._polarity_suppression(global_ctr)

    # ── opinion_words.json ──
    op_data = {}
    for desa, aspects in ctr.items():
        d_out = {}
        for cat, by_sent in aspects.items():
            a_out = {}
            for sent in SENTS:
                words = [w for w, _ in by_sent[sent].most_common()
                         if w not in suppressed[sent]][:TOP_OPINION]
                if words:
                    a_out[sent] = words
            if a_out:
                d_out[cat] = a_out
        op_data[desa] = d_out
    with open(op_out, "w", encoding="utf-8") as f:
        json.dump(op_data, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {op_out}  ({len(op_data)} desa)")

    # ── refresh field 'opinions' di insights_summary.json ──
    if os.path.exists(ins_out):
        with open(ins_out, encoding="utf-8") as f:
            ins = json.load(f)
        for desa, blk in ins.items():
            op_desa = op_data.get(desa, {})
            for p in blk.get("top_praised", []):
                p["opinions"] = op_desa.get(p["category"], {}).get("Positif", [])[:TOP_INSIGHT]
            for c in blk.get("top_criticized", []):
                c["opinions"] = op_desa.get(c["category"], {}).get("Negatif", [])[:TOP_INSIGHT]
        with open(ins_out, "w", encoding="utf-8") as f:
            json.dump(ins, f, ensure_ascii=False, indent=2)
        print(f"  updated -> {ins_out}")

    _sanity(op_data)
    return op_data


def _sanity(op):
    for desa in ("Semua", "Pentingsari", "Pulesari"):
        if desa not in op:
            continue
        print(f"\n  --- SANITY: {desa} ---")
        for cat in list(op[desa])[:6]:
            pos = op[desa][cat].get("Positif", [])[:6]
            neg = op[desa][cat].get("Negatif", [])[:6]
            print(f"    {cat:20s} +{pos}  -{neg}")


if __name__ == "__main__":
    build(os.path.join(DATA_DIR, "sentiment_labeled.csv"),
          os.path.join(DATA_DIR, "opinion_words.json"),
          os.path.join(DATA_DIR, "insights_summary.json"))

    mtl = os.path.join(DATA_DIR, "sentiment_labeled_mtl_only.csv")
    if os.path.exists(mtl):
        build(mtl,
              os.path.join(DATA_DIR, "opinion_words_mtl_only.json"),
              os.path.join(DATA_DIR, "insights_summary_mtl_only.json"))
    print("\nDONE.")
