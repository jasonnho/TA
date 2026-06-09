# -*- coding: utf-8 -*-
import os
import io
import json
import argparse
import base64
import urllib.parse
from collections import Counter

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

# ── CLI args ──────────────────────────────────────────────────────────────────
# parse_known_args supaya tidak konflik dengan flag dari Dash/Jupyter runner
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--mtl-only', action='store_true',
                     help='Pakai data hasil MTL tanpa fallback (sentiment_labeled_mtl_only.csv)')
_parser.add_argument('--port', type=int, default=8050)
ARGS, _ = _parser.parse_known_args()
DATA_SUFFIX = '_mtl_only' if ARGS.mtl_only else ''

# ── Data ──────────────────────────────────────────────────────────────────────
# Auto-deteksi lokasi data supaya app.py IDENTIK di repo skripsi & repo deploy:
#   deploy (hf_space_repo): data ada di ./data (sebelah app.py)
#   repo skripsi          : data ada di ../data (sentiment analysis/data)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(DATA_DIR, f'sentiment_labeled{DATA_SUFFIX}.csv'))

# Parse aspects_str back to list
if 'aspects_str' in df.columns:
    df['aspects_list'] = df['aspects_str'].fillna('Lainnya').str.split('|')
else:
    df['aspects_list'] = [['Lainnya']] * len(df)

# Per-aspect sentiment map (aspek -> sentimen) per row. Sumber: kolom
# `aspect_sentiments_str` yang pipe-separated paralel ke `aspects_str`.
# Pakai sentimen per-aspek (MTL bila ada span, fallback ke overall) supaya
# review pujian kebersihan tidak tersaring sebagai 'keluhan kebersihan' hanya
# karena overall review-nya Negatif (mis. tiket mahal).
if 'aspect_sentiments_str' in df.columns:
    df['aspect_sentiments_list'] = (df['aspect_sentiments_str']
                                    .fillna('').str.split('|'))
    df['aspect_sentiment_map'] = df.apply(
        lambda r: dict(zip(r['aspects_list'], r['aspect_sentiments_list'])),
        axis=1)
else:
    # Backward compatibility kalau CSV belum di-regenerate: fallback ke overall
    df['aspect_sentiment_map'] = df.apply(
        lambda r: {a: r['sentiment'] for a in r['aspects_list']}, axis=1)


def aspect_sentiment_mask(d, aspect, sentiment):
    """Boolean mask: row menyebut `aspect` dengan sentimen *per-aspek* `sentiment`."""
    return d['aspect_sentiment_map'].apply(lambda m: m.get(aspect) == sentiment)

# Load pre-computed JSON files
def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

tfidf_data   = load_json(f'tfidf_keywords{DATA_SUFFIX}.json')
aspect_data  = load_json(f'aspect_summary{DATA_SUFFIX}.json')
insight_data = load_json(f'insights_summary{DATA_SUFFIX}.json')
opinion_data = load_json(f'opinion_words{DATA_SUFFIX}.json')
# Word cloud & bigram phrase-based (per desa, sentimen): kata diambil per-KLAUSA
# dengan sentimen klausa sendiri (hybrid MTL + mdhugol), bukan dari teks full
# review. Mencegah kata positif ('bagus') bocor ke cloud negatif. Lihat
# _build_phrase_words.py / SA_AutoLabel.ipynb. Kosong → fallback ke teks review.
phrase_data  = load_json(f'phrase_words{DATA_SUFFIX}.json')

# Kolom teks untuk analytics (word cloud, bigram). Pakai stemmed kalau ada
# supaya variasi imbuhan ('pemandangan', 'pemandangannya') ke-collapse jadi 1
# bentuk dasar di chart frequency. Fallback ke cleaned_review (un-stemmed)
# untuk backward compatibility dgn CSV lama.
ANALYTICS_COL = ('cleaned_review_stemmed' if 'cleaned_review_stemmed' in df.columns
                 else 'cleaned_review')

LABEL_NAME = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
COLORS     = {'Positif': '#10b981', 'Negatif': '#ef4444', 'Netral': '#6366f1'}
COLORS_SOFT = {'Positif': '#34d399', 'Negatif': '#f87171', 'Netral': '#818cf8'}
DESA_LIST  = sorted(df['nama desa wisata'].dropna().unique().tolist())

# ── Province mapping ──────────────────────────────────────────────────────────
PROVINCE_MAP = {
    # Old data (lowercase, from Excel)
    'kampung blekok':     'Jawa Timur',
    'umbul ponggok':      'Jawa Tengah',
    'pujon kidul':        'Jawa Timur',
    'pujonkidul':         'Jawa Timur',
    'pentingsari':        'DI Yogyakarta',
    'penglipuran':        'Bali',
    'kete kesu':          'Sulawesi Selatan',
    'osing':              'Jawa Timur',
    'pulesari':           'DI Yogyakarta',
    # New data (from scraping)
    'gampong nusa':                'Aceh',
    'tomok':                       'Sumatera Utara',
    'nagari pariangan':            'Sumatera Barat',
    'buluh cina':                  'Riau',
    'kampung terih':               'Kepulauan Riau',
    'tanjung laut':                'Jambi',
    'pulau kumayan kota':          'Bengkulu',
    'pulau kemaro':                'Sumatera Selatan',
    'mangrove kurau':              'Kepulauan Bangka Belitung',
    'kunjir':                      'Lampung',
    'setu babakan':                'DKI Jakarta',
    'kampung naga':                'Jawa Barat',
    'kampung marengo baduy':       'Banten',
    'desa sade':                   'Nusa Tenggara Barat',
    'desa wisata waerebo':         'Nusa Tenggara Timur',
    'desa wisata sungai utik':     'Kalimantan Barat',
    'desa wisata sungai sekonyer': 'Kalimantan Tengah',
    'kampung ketupat':             'Kalimantan Selatan',
    'desa wisata pampang':         'Kalimantan Timur',
    'desa wisata setulang':        'Kalimantan Utara',
    'desa wisata budo':            'Sulawesi Utara',
    'desa wisata olele':           'Gorontalo',
    'danau paisu':                 'Sulawesi Tengah',
    'mamuju city':                 'Sulawesi Barat',
    'danau napabale':              'Sulawesi Tenggara',
    'pantai batu kuda':            'Maluku',
    'tanjung rappa pelangi':       'Maluku Utara',
    'sauwandarek':                 'Papua Barat',
    'danau emtofe':                'Papua',
}

# Fill provinsi column — NaN for old Excel data, already set for new scraped data
if 'provinsi' not in df.columns:
    df['provinsi'] = ''
_mask_prov = df['provinsi'].isna() | (df['provinsi'].astype(str).str.strip() == '')
df.loc[_mask_prov, 'provinsi'] = (
    df.loc[_mask_prov, 'nama desa wisata']
    .str.lower().str.strip()
    .map(PROVINCE_MAP)
)
df['provinsi'] = df['provinsi'].fillna('Tidak Diketahui')

# Build province → desa list mapping (for province dropdown cascade)
PROVINSI_DESA: dict = {}
for _desa in DESA_LIST:
    _prov = df[df['nama desa wisata'] == _desa]['provinsi'].mode()
    _prov = _prov.iloc[0] if not _prov.empty else 'Tidak Diketahui'
    PROVINSI_DESA.setdefault(_prov, []).append(_desa)
PROVINSI_LIST = sorted(PROVINSI_DESA.keys())

# ── Desa images (scanned from assets/images/<desa>/) ─────────────────────────
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'images')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')


def _scan_desa_images():
    """Scan assets/images/<desa>/ → dict {desa_key_lower: [web_path, ...]}."""
    result = {}
    if not os.path.isdir(IMAGES_DIR):
        return result
    for desa_key in os.listdir(IMAGES_DIR):
        desa_dir = os.path.join(IMAGES_DIR, desa_key)
        if not os.path.isdir(desa_dir):
            continue
        files = sorted(f for f in os.listdir(desa_dir)
                       if f.lower().endswith(IMAGE_EXTS))
        result[desa_key.lower()] = [
            f'/assets/images/{desa_key}/{f}' for f in files
        ]
    return result


DESA_IMAGES = _scan_desa_images()


def _build_carousel_entries(desa_list, is_semua):
    """Build [(src, caption), ...] for current desa selection."""
    keys = DESA_LIST if is_semua else [d.lower() for d in desa_list]
    entries = []
    for k in keys:
        for src in DESA_IMAGES.get(k, []):
            entries.append((src, k.title()))
    return entries


# ── Chart theme constants ────────────────────────────────────────────────────
THEME_FONT        = 'Inter, sans-serif'
THEME_TITLE_FONT  = 'Poppins, sans-serif'
THEME_TEXT_COLOR  = '#334155'
THEME_TITLE_COLOR = '#0f172a'
THEME_AXIS_COLOR  = '#64748b'
THEME_GRID_COLOR  = '#e2e8f0'
THEME_BORDER      = '#e2e8f0'


def apply_chart_theme(fig, *, show_legend=True, show_xaxis_grid=False,
                      show_yaxis_grid=True, height=None, hide_xaxis=False,
                      hide_yaxis=False):
    """Apply unified Plotly theme (Refined Light). Call in every chart callback."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=THEME_FONT, size=12, color=THEME_TEXT_COLOR),
        title=dict(
            font=dict(family=THEME_TITLE_FONT, size=16, color=THEME_TITLE_COLOR),
            x=0.02, xanchor='left', y=0.96, pad=dict(t=4, b=4),
        ),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor=THEME_BORDER,
            font=dict(family=THEME_FONT, size=12, color=THEME_TITLE_COLOR),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=-0.22,
            xanchor='center', x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color=THEME_TEXT_COLOR),
        ),
        showlegend=show_legend,
        margin=dict(t=58, b=52, l=14, r=14),
    )
    fig.update_xaxes(
        visible=not hide_xaxis,
        showline=False, zeroline=False,
        tickfont=dict(color=THEME_AXIS_COLOR, size=11),
        gridcolor=THEME_GRID_COLOR if show_xaxis_grid else 'rgba(0,0,0,0)',
    )
    fig.update_yaxes(
        visible=not hide_yaxis,
        showline=False, zeroline=False,
        tickfont=dict(color=THEME_AXIS_COLOR, size=11),
        gridcolor=THEME_GRID_COLOR if show_yaxis_grid else 'rgba(0,0,0,0)',
    )
    if height:
        fig.update_layout(height=height)
    return fig

ASPECT_ORDER = [
    'Pemandangan & Alam', 'Kebersihan', 'Pelayanan', 'Fasilitas',
    'Harga & Tiket', 'Kuliner', 'Suasana', 'Akses & Lokasi',
    'Budaya & Tradisi', 'Lainnya',
]

ASPECT_ICONS = {
    'Pemandangan & Alam': '🏞',
    'Kebersihan':         '✨',
    'Pelayanan':          '🤝',
    'Fasilitas':          '🏗',
    'Harga & Tiket':      '🎫',
    'Kuliner':            '🍴',
    'Suasana':            '🌟',
    'Akses & Lokasi':     '📍',
    'Budaya & Tradisi':   '🎭',
    'Lainnya':            '📌',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_desa(desa):
    """Normalize desa dropdown value (may be list or str) for consistent use."""
    if isinstance(desa, str):
        desa = [desa]
    if not desa or 'Semua' in desa:
        return ['Semua'], 'Semua', True
    if len(desa) == 1:
        return desa, desa[0], False
    return desa, None, False


def filter_df(desa):
    """Filter dataframe by desa selection (handles list from multi-select)."""
    if isinstance(desa, str):
        desa = [desa]
    if not desa or 'Semua' in desa:
        return df.copy()
    return df[df['nama desa wisata'].isin(desa)].copy()


def _desa_label(desa_list, is_semua):
    """Human-readable label for selected desa(s)."""
    if is_semua:
        return 'semua desa wisata'
    if len(desa_list) == 1:
        return desa_list[0].title()
    names = ', '.join(d.title() for d in desa_list[:3])
    if len(desa_list) > 3:
        names += f' (+{len(desa_list) - 3} lainnya)'
    return names


def _compute_aspect_data(d):
    """Compute aspect summary on-the-fly using per-aspect sentiment."""
    result = {}
    for aspect in ASPECT_ORDER:
        mention = d['aspects_list'].apply(lambda x: aspect in x)
        subset = d[mention]
        if subset.empty:
            continue
        per_asp = subset['aspect_sentiment_map'].apply(lambda m: m.get(aspect))
        total = len(subset)
        pos = int((per_asp == 'Positif').sum())
        neg = int((per_asp == 'Negatif').sum())
        neu = int((per_asp == 'Netral').sum())
        result[aspect] = {
            'total': total, 'Positif': pos, 'Negatif': neg, 'Netral': neu,
            'positivity_rate': (pos / total * 100) if total else 0,
        }
    return result


def _compute_insight_data(d, asp):
    """Compute top praised/criticized from filtered dataframe."""
    praised, criticized = [], []
    for aspect, info in asp.items():
        if aspect == 'Lainnya':
            continue
        if info['Positif'] > 0:
            praised.append({'category': aspect, 'count': info['Positif'],
                            'opinions': []})
        if info['Negatif'] > 0:
            criticized.append({'category': aspect, 'count': info['Negatif'],
                               'opinions': []})
    praised.sort(key=lambda x: x['count'], reverse=True)
    criticized.sort(key=lambda x: x['count'], reverse=True)
    hl_pos = f'Pengunjung memuji {praised[0]["category"].lower()}' if praised else ''
    hl_neg = f'Keluhan utama terkait {criticized[0]["category"].lower()}' if criticized else ''
    return {
        'top_praised': praised[:3], 'top_criticized': criticized[:3],
        'headline_positive': hl_pos, 'headline_negative': hl_neg,
    }


def _compute_tfidf_on_fly(desa_list, is_semua, sentiment, top_n=10):
    """Compute distinctive TF-IDF keywords for multi-desa selection.

    Single-desa case still uses pre-computed JSON. For multi-spec selections,
    contrast selected reviews against the rest. For 'Semua', no contrast set
    exists, so fall back to top-frequency words normalized to 0–1.
    """
    if is_semua:
        d_sel = df[df['sentiment'] == sentiment]
        d_other = df.iloc[0:0]
    else:
        d_sel = df[df['nama desa wisata'].isin(desa_list)
                   & (df['sentiment'] == sentiment)]
        d_other = df[(~df['nama desa wisata'].isin(desa_list))
                     & (df['sentiment'] == sentiment)]

    sel_texts = d_sel[ANALYTICS_COL].dropna().astype(str).tolist()
    if not sel_texts:
        return []
    other_texts = d_other[ANALYTICS_COL].dropna().astype(str).tolist()

    if not other_texts:
        counter = Counter()
        for t in sel_texts:
            for word in t.lower().split():
                if len(word) >= 3 and word.isalpha():
                    counter[word] += 1
        if not counter:
            return []
        max_c = counter.most_common(1)[0][1]
        return [[w, c / max_c] for w, c in counter.most_common(top_n)]

    try:
        n_sel = len(sel_texts)
        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 1),
                              min_df=2, token_pattern=r'\b[a-z]{3,}\b')
        matrix = vec.fit_transform(sel_texts + other_texts)
        sel_mean = np.asarray(matrix[:n_sel].mean(axis=0)).flatten()
        other_mean = np.asarray(matrix[n_sel:].mean(axis=0)).flatten()
        distinctiveness = sel_mean - other_mean
        features = vec.get_feature_names_out()
        paired = sorted(zip(features, distinctiveness, sel_mean),
                        key=lambda x: x[1], reverse=True)
        return [[w, float(sm)] for w, _, sm in paired[:top_n] if sm > 0]
    except ValueError:
        return []


def _merge_opinion_data(desa_list, aspect):
    """Merge opinion words from multiple villages for a given aspect."""
    merged = {}
    for dk in desa_list:
        village_op = opinion_data.get(dk, {}).get(aspect, {})
        for sent, words in village_op.items():
            merged.setdefault(sent, []).extend(words)
    return {sent: list(dict.fromkeys(words)) for sent, words in merged.items()}


def _new_wordcloud(base_color, light_color):
    def _color_func(word, font_size, position, orientation, random_state=None, **kw):
        # Largest words get darker shade, smaller words get lighter — adds depth
        return base_color if font_size > 36 else light_color

    return WordCloud(
        width=500, height=460,
        mode='RGBA', background_color=None,
        color_func=_color_func,
        max_words=70,
        collocations=False,
        prefer_horizontal=0.85,
        min_font_size=11,
        relative_scaling=0.5,
        margin=6,
    )


def _wordcloud_to_uri(wc):
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'


def make_wordcloud(texts, base_color, light_color):
    text = ' '.join(texts)
    if not text.strip():
        return ''
    return _wordcloud_to_uri(_new_wordcloud(base_color, light_color).generate(text))


def make_wordcloud_from_freq(freq, base_color, light_color):
    """Word cloud dari dict frekuensi {kata: count} (sumber: phrase_words.json)."""
    freq = {w: c for w, c in (freq or {}).items() if c > 0}
    if not freq:
        return ''
    wc = _new_wordcloud(base_color, light_color).generate_from_frequencies(freq)
    return _wordcloud_to_uri(wc)


def _phrase_freq(desa, sentiment, key):
    """Gabung dict {token: count} phrase-based untuk (desa, sentimen).
    key ∈ {'words','bigrams'}. Multi-desa → jumlahkan antar desa terpilih;
    'Semua' → bucket 'Semua'. Return {} bila data belum ada (fallback ke teks)."""
    if not phrase_data:
        return {}
    desa_list, single_key, is_semua = _normalize_desa(desa)
    keys = ['Semua'] if is_semua else desa_list
    merged = {}
    for dk in keys:
        for tok, cnt in (phrase_data.get(dk, {}).get(sentiment, {}).get(key, []) or []):
            merged[tok] = merged.get(tok, 0) + cnt
    return merged


def top_bigrams(texts, n=10):
    bigrams = []
    for t in texts:
        words = str(t).lower().split()
        bigrams += [f'{a} {b}' for a, b in zip(words, words[1:])]
    counts = Counter(bigrams).most_common(n)
    if not counts:
        return pd.DataFrame(columns=['bigram', 'count'])
    return pd.DataFrame(counts[::-1], columns=['bigram', 'count'])


def get_verdict(pct_positif):
    """Return (verdict_text, description, css_class) for sentiment percentage."""
    if pct_positif >= 85:
        return ('Sangat Positif',
                'Sebagian besar pengunjung merasa puas dengan pengalaman mereka',
                'verdict-great')
    elif pct_positif >= 70:
        return ('Cukup Positif',
                'Pengunjung umumnya puas, namun ada beberapa catatan',
                'verdict-good')
    elif pct_positif >= 50:
        return ('Cukup Beragam',
                'Pendapat pengunjung cukup beragam antara positif dan negatif',
                'verdict-mixed')
    else:
        return ('Perlu Perhatian',
                'Cukup banyak pengunjung yang belum merasa puas',
                'verdict-poor')


def aspect_health_class(positivity_rate):
    """Return CSS class for an aspect's positivity rate."""
    if positivity_rate >= 75:
        return 'health-good'
    elif positivity_rate >= 50:
        return 'health-moderate'
    return 'health-poor'


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title='Analisis Sentimen Desa Wisata',
    suppress_callback_exceptions=True,
)
# Diekspos untuk WSGI server (gunicorn app:server) saat deploy.
server = app.server


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUTS
# ══════════════════════════════════════════════════════════════════════════════

def layout_insight():
    """Page 1: Ringkasan insight — padat, informatif, bahasa sederhana."""
    return html.Div(children=[

        # ── ROW 1: Carousel + Verdict (with embedded headlines) ─────────
        html.Div(className='verdict-row', children=[
            html.Div(className='image-carousel-section', id='carousel-section',
                     children=[
                dcc.Store(id='carousel-index', data=0),
                html.Div(className='carousel-frame', children=[
                    html.Img(id='carousel-img', className='carousel-img',
                             alt='Foto desa wisata'),
                    html.Button('‹', id='carousel-prev',
                                className='carousel-btn carousel-prev',
                                **{'aria-label': 'Gambar sebelumnya'}),
                    html.Button('›', id='carousel-next',
                                className='carousel-btn carousel-next',
                                **{'aria-label': 'Gambar berikutnya'}),
                    html.Div(id='carousel-caption',
                             className='carousel-caption'),
                    html.Div(id='carousel-counter',
                             className='carousel-counter'),
                ]),
                html.Div(id='carousel-dots', className='carousel-dots'),
            ]),
            html.Div(id='verdict-left', className='verdict-left'),
        ]),

        # ── ROW 3: Sentiment Metrics (Positif/Negatif/Netral) ───────────
        html.Div(id='verdict-right', className='verdict-right verdict-metrics-row'),

        # ── ROW 4: Quick Stats (4 cards) ────────────────────────────────
        html.Div(id='stats-row', className='metrics-row'),

        # ── ROW 5: Praised vs Criticized side-by-side ───────────────────
        html.Div(className='praised-criticized-row', children=[
            html.Div(className='insight-card-compact positif', children=[
                html.Div(className='insight-mega-title', children=[
                    html.Span(className='insight-mega-icon positif',
                              children='V'),
                    'Yang Paling Disukai',
                ]),
                html.Div(id='praised-items'),
            ]),
            html.Div(className='insight-card-compact negatif', children=[
                html.Div(className='insight-mega-title', children=[
                    html.Span(className='insight-mega-icon negatif',
                              children='!'),
                    'Yang Perlu Diperbaiki',
                ]),
                html.Div(id='criticized-items'),
            ]),
        ]),

        # ── ROW 6: Aspect Grid + Sidebar ────────────────────────────────
        html.Div(className='bottom-two-col', children=[
            # Left: Aspect health grid
            html.Div(className='section compact', children=[
                html.H2('Kondisi Setiap Aspek', className='section-title'),
                html.Div(id='aspect-health-grid',
                         className='aspect-health-grid in-sidebar'),
            ]),
            # Right: Sidebar
            html.Div(className='sidebar-col', children=[
                html.Div(id='tfidf-section',
                         className='section compact'),
                html.Div(id='sample-reviews',
                         className='section compact'),
            ]),
        ]),

        # ── ROW 7: CTA ──────────────────────────────────────────────────
        html.Div(className='cta-section', children=[
            dcc.Link('Lihat Grafik & Data Lengkap',
                      href='/detail', className='cta-link'),
        ]),
    ])


def make_review_table(table_id, columns, data=None, page_size=15):
    """DataTable ulasan dengan styling konsisten. Dipakai mode pengelola
    (data diisi via callback) maupun mode pengunjung (data statis per-desa)."""
    extra = {} if data is None else {'data': data}
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        page_size=page_size,
        style_table={
            'overflowX': 'auto', 'borderRadius': '14px',
            'overflow': 'hidden',
            'border': '1px solid #e2e8f0',
        },
        style_cell={
            'textAlign': 'left', 'padding': '14px 16px',
            'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
            'whiteSpace': 'normal', 'height': 'auto', 'maxWidth': '400px',
            'border': 'none',
            'borderBottom': '1px solid #f1f5f9',
            'color': '#334155',
        },
        style_header={
            'background': 'linear-gradient(135deg, #0c1e36, #1e3a5f)',
            'color': 'white',
            'fontWeight': '700', 'textAlign': 'left',
            'padding': '14px 16px',
            'fontFamily': 'Inter, sans-serif',
            'fontSize': '12px',
            'letterSpacing': '0.6px',
            'textTransform': 'uppercase',
            'border': 'none',
        },
        style_data={'backgroundColor': 'white'},
        style_filter={
            'backgroundColor': '#f8fafc',
            'border': 'none',
            'borderBottom': '2px solid #e2e8f0',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'},
             'backgroundColor': '#f8fafc'},
            {'if': {'filter_query': '{sentiment} = "Positif"',
                    'column_id': 'sentiment'},
             'color': '#047857', 'fontWeight': '600'},
            {'if': {'filter_query': '{sentiment} = "Negatif"',
                    'column_id': 'sentiment'},
             'color': '#b91c1c', 'fontWeight': '600'},
            {'if': {'filter_query': '{sentiment} = "Netral"',
                    'column_id': 'sentiment'},
             'color': '#4338ca', 'fontWeight': '600'},
        ],
        css=[
            {'selector': '.dash-spreadsheet-container tr:hover td',
             'rule': 'background-color: #f1f5f9 !important;'},
            {'selector': '.dash-filter input',
             'rule': 'border-radius: 6px; padding: 5px 8px; '
                     'border: 1px solid #cbd5e1 !important; '
                     'background-color: white !important; '
                     'color: #334155 !important; '
                     'font-size: 12px; font-family: Inter, sans-serif;'},
            {'selector': '.dash-filter input::placeholder',
             'rule': 'color: #94a3b8 !important;'},
            {'selector': '.dash-filter input:focus',
             'rule': 'border-color: #2d6a9f !important; '
                     'outline: none !important; '
                     'box-shadow: 0 0 0 3px rgba(45,106,159,.12) !important;'},
            {'selector': '.dash-filter .column-header--sort',
             'rule': 'opacity: 1 !important; color: #94a3b8 !important;'},
        ],
        sort_action='native',
        filter_action='native',
        **extra,
    )


def layout_detail():
    """Page 2: Semua chart dan tabel detail."""
    return html.Div(children=[

        # Back link
        dcc.Link('Kembali ke Ringkasan', href='/', className='back-link'),

        # ── Gambaran Umum ────────────────────────────────────────────────
        html.Div(className='section', children=[
            html.H2('Gambaran Umum', className='section-title'),
            html.Div(className='chart-row chart-row-same-height', children=[
                html.Div(className='chart-card wide', children=[
                    dcc.Graph(id='grouped-bar', config={'displayModeBar': False}),
                ]),
                html.Div(className='chart-card narrow', children=[
                    dcc.Graph(id='pie-chart', config={'displayModeBar': False}),
                ]),
            ]),
        ]),

        # ── Apa yang Dibicarakan Pengunjung? ─────────────────────────────
        html.Div(className='section', children=[
            html.H2('Apa yang Dibicarakan Pengunjung?', className='section-title'),
            html.Div(className='chart-row', children=[
                html.Div(className='chart-card wide', children=[
                    dcc.Graph(id='aspect-bar', config={'displayModeBar': False}),
                ]),
                html.Div(className='chart-card narrow', children=[
                    html.Label('Pilih Aspek:', className='filter-label',
                               style={'marginBottom': '8px'}),
                    dcc.Dropdown(
                        id='aspect-dropdown',
                        options=[{'label': a, 'value': a}
                                 for a in ASPECT_ORDER if a != 'Lainnya'],
                        value='Pemandangan & Alam',
                        clearable=False,
                        className='modern-dropdown',
                    ),
                    dcc.Graph(id='opinion-bar', config={'displayModeBar': False}),
                ]),
            ]),
        ]),

        # ── Analisis Kata ────────────────────────────────────────────────
        html.Div(className='section', children=[
            html.H2('Analisis Kata', className='section-title'),
            dcc.Tabs(id='sentiment-tabs', value='Positif',
                     className='sentiment-tabs', children=[
                dcc.Tab(label='Positif', value='Positif',
                        className='tab', selected_className='tab-selected'),
                dcc.Tab(label='Negatif', value='Negatif',
                        className='tab', selected_className='tab-selected'),
                dcc.Tab(label='Netral', value='Netral',
                        className='tab', selected_className='tab-selected'),
            ]),
            html.Div(className='chart-row chart-row-triple', children=[
                html.Div(className='chart-card chart-card-equal', children=[
                    dcc.Graph(id='tfidf-bar', config={'displayModeBar': False}),
                ]),
                html.Div(className='chart-card chart-card-equal', children=[
                    html.Div('Word Cloud', className='wordcloud-title'),
                    html.Img(id='wordcloud-img', className='wordcloud-img'),
                ]),
                html.Div(className='chart-card chart-card-equal', children=[
                    dcc.Graph(id='bigram-chart', config={'displayModeBar': False}),
                ]),
            ]),
        ]),

        # ── Perbandingan Desa (conditional) ──────────────────────────────
        html.Div(id='comparison-section', className='section', children=[
            html.H2('Perbandingan Antar Desa', className='section-title'),
            html.Div(className='chart-card', children=[
                dcc.Graph(id='radar-chart', config={'displayModeBar': False}),
            ]),
            html.Div(className='chart-card', style={'marginTop': '20px'}, children=[
                dcc.Graph(id='heatmap-chart', config={'displayModeBar': False}),
            ]),
        ]),

        # ── Tabel Review ─────────────────────────────────────────────────
        html.Div(className='section', children=[
            html.H2('Tabel Review', className='section-title'),
            html.Div(className='table-filter-row', children=[
                html.Label('Filter Sentimen:', className='filter-label'),
                dcc.Dropdown(
                    id='table-sentiment-filter',
                    options=[{'label': 'Semua', 'value': 'Semua'}] +
                            [{'label': s, 'value': s}
                             for s in ['Positif', 'Negatif', 'Netral']],
                    value='Semua', clearable=False,
                    className='modern-dropdown modern-dropdown-sm',
                ),
                html.Label('Filter Aspek:', className='filter-label'),
                dcc.Dropdown(
                    id='table-aspect-filter',
                    options=[{'label': 'Semua', 'value': 'Semua'}] +
                            [{'label': a, 'value': a} for a in ASPECT_ORDER],
                    value='Semua', clearable=False,
                    className='modern-dropdown modern-dropdown-md',
                ),
            ]),
            make_review_table(
                'review-table',
                columns=[
                    {'name': 'Desa Wisata',  'id': 'nama desa wisata'},
                    {'name': 'Sentimen',     'id': 'sentiment'},
                    {'name': 'Aspek',        'id': 'aspects_str'},
                    {'name': 'Review',       'id': 'cleaned_review'},
                ],
            ),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# VISITOR MODE (Pengunjung) — gallery + destination detail
# ══════════════════════════════════════════════════════════════════════════════

_VISITOR_SUMMARY_CACHE = {}
VISITOR_POPULAR_MIN = 500   # ambang badge "Populer" (jumlah ulasan)
VISITOR_FEW_MAX = 50        # ambang badge "Sedikit ulasan"


def _visitor_card_summary(desa):
    """Ringkasan ringkas per-desa untuk kartu galeri pengunjung (cached)."""
    if desa in _VISITOR_SUMMARY_CACHE:
        return _VISITOR_SUMMARY_CACHE[desa]
    d_tmp = df[df['nama desa wisata'] == desa]
    total = len(d_tmp)
    pct_pos = float((d_tmp['sentiment'] == 'Positif').mean() * 100) if total else 0.0
    asp = aspect_data.get(desa) or _compute_aspect_data(d_tmp)
    ins = insight_data.get(desa) or _compute_insight_data(d_tmp, asp)
    praised = ins.get('top_praised', [])
    top_aspect = praised[0]['category'] if praised else None
    prov_mode = d_tmp['provinsi'].mode()
    prov = prov_mode.iloc[0] if not prov_mode.empty else 'Tidak Diketahui'
    imgs = DESA_IMAGES.get(desa.lower(), [])
    summary = {
        'desa': desa, 'pct_pos': pct_pos, 'total': total,
        'top_aspect': top_aspect, 'province': prov,
        'cover': imgs[0] if imgs else None,
        # Skor & jumlah ulasan per-aspek → dipakai untuk filter "unggul di aspek"
        'aspect_rates': {a: info.get('positivity_rate', 0) for a, info in asp.items()},
        'aspect_totals': {a: info.get('total', 0) for a, info in asp.items()},
    }
    _VISITOR_SUMMARY_CACHE[desa] = summary
    return summary


def _visitor_card(c, aspect=None):
    """Kartu destinasi (link ke /wisata?d=<desa>).

    Bila `aspect` diberikan (mode filter "unggul di aspek"), badge & ranking
    memakai skor kepuasan aspek tersebut, bukan kepuasan keseluruhan.
    """
    rate = c['aspect_rates'].get(aspect, 0) if aspect else c['pct_pos']
    _, _, verdict_cls = get_verdict(rate)
    if c['cover']:
        cover_el = html.Div(className='visitor-card-cover',
                            style={'backgroundImage': f'url("{c["cover"]}")'})
    else:
        cover_el = html.Div(className='visitor-card-cover no-image',
                            children=html.Span(c['desa'][:1].upper()))

    meta_children = [html.Span(f'{c["total"]:,} ulasan',
                               className='visitor-card-count')]
    chip_aspect = aspect or c['top_aspect']
    if chip_aspect:
        icon = ASPECT_ICONS.get(chip_aspect, '◈')
        meta_children.append(
            html.Span(f'{icon} {chip_aspect}', className='visitor-card-chip'))

    cover_children = [
        cover_el,
        html.Div(f'{rate:.0f}% puas', className=f'visitor-card-badge {verdict_cls}'),
    ]
    if c['total'] >= VISITOR_POPULAR_MIN:
        cover_children.append(
            html.Div('🔥 Populer', className='visitor-card-tag popular'))
    elif c['total'] < VISITOR_FEW_MAX:
        cover_children.append(
            html.Div('Sedikit ulasan', className='visitor-card-tag few'))

    return dcc.Link(
        href=f'/wisata?d={urllib.parse.quote(c["desa"])}',
        className='visitor-card', children=[
            html.Div(className='visitor-card-cover-wrap', children=cover_children),
            html.Div(className='visitor-card-body', children=[
                html.Div(c['desa'].title(), className='visitor-card-name'),
                html.Div(f'📍 {c["province"]}', className='visitor-card-prov'),
                html.Div(meta_children, className='visitor-card-meta'),
            ]),
        ])


def layout_visitor_gallery():
    """Mode pengunjung: galeri kartu destinasi yang bisa difilter & dicari."""
    return html.Div(className='visitor-page', children=[
        html.Div(className='visitor-hero', children=[
            html.H1('Jelajahi Desa Wisata Indonesia',
                    className='visitor-hero-title'),
            html.P('Pilih destinasi dan lihat ringkasan pengalaman langsung dari '
                   'ribuan ulasan pengunjung sebelumnya.',
                   className='visitor-hero-sub'),
        ]),
        html.Div(className='visitor-filter-row', children=[
            dcc.Dropdown(
                id='visitor-prov-filter',
                options=[{'label': 'Semua Provinsi', 'value': 'Semua'}] +
                        [{'label': p, 'value': p} for p in PROVINSI_LIST],
                value='Semua', clearable=False,
                className='modern-dropdown visitor-prov-dd',
            ),
            dcc.Dropdown(
                id='visitor-aspect-filter',
                options=[{'label': 'Semua Aspek', 'value': 'Semua'}] +
                        [{'label': f'Unggul di {a}', 'value': a}
                         for a in ASPECT_ORDER if a != 'Lainnya'],
                value='Semua', clearable=False,
                className='modern-dropdown visitor-aspect-dd',
            ),
            dcc.Dropdown(
                id='visitor-sort',
                options=[
                    {'label': 'Urutkan: Kepuasan tertinggi', 'value': 'kepuasan'},
                    {'label': 'Urutkan: Ulasan terbanyak', 'value': 'ulasan'},
                    {'label': 'Urutkan: Nama (A–Z)', 'value': 'nama'},
                ],
                value='kepuasan', clearable=False,
                className='modern-dropdown visitor-sort-dd',
            ),
            dcc.Input(id='visitor-search', type='text',
                      placeholder='Cari nama desa wisata…',
                      className='visitor-search-input'),
        ]),
        html.Div(id='visitor-result-count', className='visitor-result-count'),
        html.Div(id='visitor-gallery-grid', className='visitor-gallery-grid'),
    ])


def layout_visitor_detail(desa):
    """Mode pengunjung: ringkasan ramah-pengunjung untuk satu destinasi."""
    back = dcc.Link('Kembali ke daftar destinasi', href='/jelajah',
                    className='back-link')
    if not desa or desa not in DESA_LIST:
        return html.Div(className='visitor-page', children=[
            back,
            html.P('Destinasi tidak ditemukan. Silakan pilih dari daftar.',
                   className='visitor-empty'),
        ])

    d = df[df['nama desa wisata'] == desa]
    total = len(d)
    n_pos = int((d['sentiment'] == 'Positif').sum())
    n_neg = int((d['sentiment'] == 'Negatif').sum())
    n_neu = int((d['sentiment'] == 'Netral').sum())
    pct_pos = (n_pos / total * 100) if total else 0
    pct_neg = (n_neg / total * 100) if total else 0
    pct_neu = (n_neu / total * 100) if total else 0

    asp = aspect_data.get(desa) or _compute_aspect_data(d)
    ins = insight_data.get(desa) or _compute_insight_data(d, asp)
    verdict_label, verdict_desc, verdict_cls = get_verdict(pct_pos)
    prov_mode = d['provinsi'].mode()
    prov = prov_mode.iloc[0] if not prov_mode.empty else 'Tidak Diketahui'

    # ── Hero: carousel/foto + verdict + metrik ──────────────────────────────
    entries = _build_carousel_entries([desa], False)
    if entries:
        first_src, _ = entries[0]
        hero_media = html.Div(className='visitor-detail-media', children=[
            dcc.Store(id='visitor-carousel-index', data=0),
            html.Img(id='visitor-carousel-img', src=first_src,
                     className='visitor-detail-img', alt=f'Foto {desa.title()}'),
            html.Button('‹', id='visitor-carousel-prev',
                        className='carousel-btn carousel-prev',
                        **{'aria-label': 'Gambar sebelumnya'}),
            html.Button('›', id='visitor-carousel-next',
                        className='carousel-btn carousel-next',
                        **{'aria-label': 'Gambar berikutnya'}),
            html.Div(f'1 / {len(entries)}', id='visitor-carousel-counter',
                     className='carousel-counter'),
        ])
    else:
        hero_media = html.Div(className='visitor-detail-media no-image',
                              children=html.Span(desa[:1].upper()))

    hero = html.Div(className='visitor-detail-hero', children=[
        hero_media,
        html.Div(className='visitor-detail-hero-info', children=[
            html.Div(f'📍 {prov}', className='visitor-detail-prov'),
            html.H1(desa.title(), className='visitor-detail-title'),
            html.Div(className=f'visitor-detail-verdict {verdict_cls}', children=[
                html.Div(verdict_label, className='verdict-text'),
                html.Div(verdict_desc, className='verdict-desc'),
            ]),
            html.Div(f'Berdasarkan {total:,} ulasan pengunjung',
                     className='verdict-total'),
            html.Div(className='verdict-metrics', children=[
                _metric_mini(f'{n_pos:,}', f'{pct_pos:.1f}%', 'Positif', 'positif'),
                _metric_mini(f'{n_neg:,}', f'{pct_neg:.1f}%', 'Negatif', 'negatif'),
                _metric_mini(f'{n_neu:,}', f'{pct_neu:.1f}%', 'Netral', 'netral'),
            ]),
        ]),
    ])

    # ── Yang disukai / perlu diperhatikan ───────────────────────────────────
    def _items(records, sentiment, css, count_label):
        out = []
        for i, p in enumerate(records[:3], 1):
            cat = p['category']
            info = asp.get(cat, {})
            kws = p.get('opinions', [])[:4]
            samples = _pick_sample_reviews(d, cat, sentiment, n=2)
            samples = [(s[0], '') for s in samples] if samples else None
            kwargs = {}
            if sentiment == 'Positif':
                kwargs['positivity_rate'] = info.get('positivity_rate', 0)
            else:
                total_a = info.get('total', 0)
                kwargs['negativity_rate'] = (
                    (info.get('Negatif', 0) / total_a * 100) if total_a else 0)
            out.append(_insight_item(
                rank=i, category=cat, count=p['count'],
                count_label=count_label, keywords=kws,
                sample_reviews=samples, review_css=css, **kwargs))
        if not out:
            out = [html.P('Data belum tersedia.', className='visitor-empty')]
        return out

    likes_dislikes = html.Div(className='praised-criticized-row', children=[
        html.Div(className='insight-card-compact positif', children=[
            html.Div(className='insight-mega-title', children=[
                html.Span(className='insight-mega-icon positif', children='V'),
                'Yang Disukai Pengunjung',
            ]),
            html.Div(_items(ins.get('top_praised', []), 'Positif',
                            'positif', 'ulasan positif')),
        ]),
        html.Div(className='insight-card-compact negatif', children=[
            html.Div(className='insight-mega-title', children=[
                html.Span(className='insight-mega-icon negatif', children='!'),
                'Yang Perlu Diperhatikan',
            ]),
            html.Div(_items(ins.get('top_criticized', []), 'Negatif',
                            'negatif', 'keluhan')),
        ]),
    ])

    # ── Rating tiap aspek ───────────────────────────────────────────────────
    grid_children = []
    for a in ASPECT_ORDER:
        if a == 'Lainnya' or a not in asp:
            continue
        info = asp[a]
        rate = info.get('positivity_rate', 0)
        total_a = info.get('total', 0)
        h_cls = aspect_health_class(rate)
        op = opinion_data.get(desa, {}).get(a, {})
        top_words = (op.get('Positif', [])[:2] + op.get('Negatif', [])[:1])[:3]
        grid_children.append(html.Div(className=f'aspect-health-card {h_cls}', children=[
            html.Div(className='aspect-health-header', children=[
                html.Div(ASPECT_ICONS.get(a, '◈'), className='aspect-health-icon'),
                html.Div(a, className='aspect-health-name'),
            ]),
            html.Div(className='aspect-health-bar-track', children=[
                html.Div(className=f'aspect-health-bar-fill {h_cls}',
                         style={'width': f'{rate}%'}),
            ]),
            html.Div(className='aspect-health-meta', children=[
                html.Span(f'{rate:.0f}% puas', className='aspect-health-pct'),
                html.Span(f'({total_a:,} ulasan)', className='aspect-health-count'),
            ]),
            html.Div(', '.join(top_words),
                     className='aspect-health-words') if top_words else None,
        ]))

    aspect_section = html.Div(className='section', children=[
        html.H2('Penilaian Tiap Aspek', className='section-title'),
        html.P('Seberapa puas pengunjung untuk masing-masing aspek pengalaman.',
               className='section-caption'),
        html.Div(grid_children, className='aspect-health-grid'),
    ])

    # ── Contoh ulasan ───────────────────────────────────────────────────────
    review_children = [html.H2('Contoh Ulasan Pengunjung', className='section-title')]
    d_short = d[d['cleaned_review'].str.len().between(20, 150)]
    for sent, cls in [('Positif', 'positif'), ('Negatif', 'negatif')]:
        subset = d_short[d_short['sentiment'] == sent]
        if len(subset) > 0:
            sample = subset.sample(1, random_state=42).iloc[0]
            review_children.append(_review_quote(sample['cleaned_review'], cls))
    if len(review_children) == 1:
        review_children.append(
            html.P('Tidak cukup data.', className='visitor-empty'))

    # ── Semua ulasan (tabel) ────────────────────────────────────────────────
    table_cols = [
        {'name': 'Sentimen', 'id': 'sentiment'},
        {'name': 'Aspek',    'id': 'aspects_str'},
        {'name': 'Ulasan',   'id': 'cleaned_review'},
    ]
    table_data = (d[['sentiment', 'aspects_str', 'cleaned_review']]
                  .dropna(subset=['cleaned_review'])
                  .to_dict('records'))
    table_section = html.Div(className='section', children=[
        html.H2('Semua Ulasan Pengunjung', className='section-title'),
        html.P('Telusuri seluruh ulasan untuk destinasi ini — ketik di kotak '
               'filter pada tiap kolom untuk mencari ulasan tertentu.',
               className='section-caption'),
        make_review_table('visitor-review-table', table_cols, data=table_data),
    ])

    return html.Div(className='visitor-page', children=[
        back,
        hero,
        likes_dislikes,
        aspect_section,
        html.Div(className='section', children=review_children),
        table_section,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SHELL LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='prev-desa', data=['Semua']),

    # ── Header (sticky, nav only) ─────────────────────────────────────────
    html.Div(className='header', children=[
        html.Div(className='header-accent-bar'),
        html.Div(className='header-inner', children=[
            html.Div(className='header-title-block', children=[
                html.H1('Dashboard Analisis Sentimen'),
                html.Span(
                    'Mode: MTL-Only (tanpa fallback)' if ARGS.mtl_only
                    else 'Mode: Hybrid (default)',
                    className=('mode-badge mtl-only' if ARGS.mtl_only
                               else 'mode-badge default'),
                ),
            ]),
            html.Div(className='header-controls', children=[
                html.Div(id='pengelola-nav', className='header-nav-tabs', children=[
                    dcc.Link('Ringkasan', href='/', id='nav-insight',
                              className='header-nav-link active'),
                    dcc.Link('Detail & Grafik', href='/detail', id='nav-detail',
                              className='header-nav-link'),
                ]),
                html.Div(className='mode-toggle', children=[
                    dcc.Link('Mode Pengelola', href='/', id='mode-pengelola',
                              className='mode-toggle-link active'),
                    dcc.Link('Mode Pengunjung', href='/jelajah',
                              id='mode-pengunjung',
                              className='mode-toggle-link'),
                ]),
            ]),
        ]),
    ]),

    # ── App Body: Sidebar + Main ──────────────────────────────────────────
    html.Div(className='app-body', children=[

        # ── Filter Sidebar ────────────────────────────────────────────────
        html.Div(id='filter-sidebar', className='filter-sidebar', children=[
            html.Div('Filter', className='sidebar-section-title'),

            html.Div(className='sidebar-filter-group', children=[
                html.Label('Provinsi', className='sidebar-filter-label'),
                html.Div(className='sidebar-filter-options', children=[
                    dcc.RadioItems(
                        id='provinsi-dropdown',
                        options=[{'label': 'Semua Provinsi', 'value': 'Semua'}] +
                                [{'label': p, 'value': p} for p in PROVINSI_LIST],
                        value='Semua',
                        className='sidebar-radio-list',
                        inputClassName='sidebar-option-input',
                        labelClassName='sidebar-option-label',
                    ),
                ]),
            ]),

            html.Div(className='sidebar-filter-group', children=[
                html.Label('Desa Wisata', className='sidebar-filter-label'),
                html.Div(className='sidebar-filter-options', children=[
                    dcc.Checklist(
                        id='desa-dropdown',
                        options=[{'label': 'Semua Desa Wisata', 'value': 'Semua'}] +
                                [{'label': d.title(), 'value': d} for d in DESA_LIST],
                        value=['Semua'],
                        className='sidebar-checklist',
                        inputClassName='sidebar-option-input',
                        labelClassName='sidebar-option-label',
                    ),
                ]),
            ]),
        ]),

        # ── Page Content + Footer ─────────────────────────────────────────
        html.Div(className='page', children=[
            html.Div(id='page-content'),

            html.Div(className='footer', children=[
                html.P(
                    'Data: Ulasan Google Maps Desa Wisata Indonesia | '
                    + ('Label sentimen: H-MTL (IndoBERT-Large-P2 fine-tuned), '
                       'aspek dari span MTL (tanpa fallback ke baseline)'
                       if ARGS.mtl_only else
                       'Label sentimen: hybrid mdhugol/IndoBERT (review-level) '
                       '+ H-MTL (per-aspek, fallback ke sentence-level bila tidak ada span)')
                ),
            ]),
        ]),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def _is_visitor_path(pathname):
    """True kalau pathname masuk ke mode pengunjung (/jelajah atau /wisata)."""
    return bool(pathname) and (pathname.startswith('/jelajah')
                               or pathname.startswith('/wisata'))


def _desa_from_search(search):
    """Ambil desa terpilih dari query string '?d=<desa>' (URL-encoded)."""
    if not search:
        return None
    qs = urllib.parse.parse_qs(search.lstrip('?'))
    vals = qs.get('d')
    return vals[0] if vals else None


@callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('url', 'search'),
)
def display_page(pathname, search):
    if pathname == '/detail':
        return layout_detail()
    if pathname == '/jelajah':
        return layout_visitor_gallery()
    if pathname == '/wisata':
        return layout_visitor_detail(_desa_from_search(search))
    return layout_insight()


@callback(
    Output('nav-insight', 'className'),
    Output('nav-detail', 'className'),
    Output('mode-pengelola', 'className'),
    Output('mode-pengunjung', 'className'),
    Output('pengelola-nav', 'style'),
    Output('filter-sidebar', 'style'),
    Input('url', 'pathname'),
)
def update_nav_active(pathname):
    visitor = _is_visitor_path(pathname)
    hidden = {'display': 'none'}
    if visitor:
        # Mode pengunjung: sembunyikan nav & sidebar analitik
        return ('header-nav-link', 'header-nav-link',
                'mode-toggle-link', 'mode-toggle-link active',
                hidden, hidden)
    # Mode pengelola
    if pathname == '/detail':
        nav_ins, nav_det = 'header-nav-link', 'header-nav-link active'
    else:
        nav_ins, nav_det = 'header-nav-link active', 'header-nav-link'
    return (nav_ins, nav_det,
            'mode-toggle-link active', 'mode-toggle-link',
            None, None)


# ── Smart "Semua" toggle ────────────────────────────────────────────────────
@callback(
    Output('desa-dropdown', 'value'),
    Output('prev-desa', 'data'),
    Input('desa-dropdown', 'value'),
    Input('prev-desa', 'data'),
    Input('provinsi-dropdown', 'value'),
    prevent_initial_call=True,
)
def smart_semua_toggle(current, prev, provinsi):
    """Province filter auto-selects desa in that province.
    Desa filter keeps Semua/specific logic intact."""
    from dash import ctx
    triggered = ctx.triggered_id

    # Province changed → auto-populate desa dropdown
    if triggered == 'provinsi-dropdown':
        if not provinsi or provinsi == 'Semua':
            return ['Semua'], ['Semua']
        desa_in_prov = PROVINSI_DESA.get(provinsi, [])
        val = desa_in_prov if desa_in_prov else ['Semua']
        return val, val

    # Desa changed → original smart-toggle logic
    if not current:
        return ['Semua'], ['Semua']
    prev = prev or []
    had_semua = 'Semua' in prev
    has_semua = 'Semua' in current
    has_specific = any(v != 'Semua' for v in current)
    if has_semua and has_specific:
        if not had_semua:
            return ['Semua'], ['Semua']
        else:
            new_val = [v for v in current if v != 'Semua']
            return new_val, new_val
    return current, current


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 CALLBACKS (Insight)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output('verdict-left', 'children'),
    Output('verdict-right', 'children'),
    Output('stats-row', 'children'),
    Output('praised-items', 'children'),
    Output('criticized-items', 'children'),
    Output('aspect-health-grid', 'children'),
    Output('tfidf-section', 'children'),
    Output('sample-reviews', 'children'),
    Input('desa-dropdown', 'value'),
)
def update_insight_page(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    label = _desa_label(desa_list, is_semua)

    # Resolve data sources: pre-computed JSON for single key, on-the-fly for multi
    if single_key:
        key = single_key
        ins = insight_data.get(key, {})
        asp = aspect_data.get(key, {})
    else:
        key = None
        d_tmp = filter_df(desa)
        asp = _compute_aspect_data(d_tmp)
        ins = _compute_insight_data(d_tmp, asp)

    d = filter_df(desa)
    total = len(d)
    n_pos = int((d['sentiment'] == 'Positif').sum())
    n_neg = int((d['sentiment'] == 'Negatif').sum())
    n_neu = int((d['sentiment'] == 'Netral').sum())
    pct_pos = (n_pos / total * 100) if total else 0
    pct_neg = (n_neg / total * 100) if total else 0
    pct_neu = (n_neu / total * 100) if total else 0

    # ── ROW 1: Verdict (with embedded headline chips) ────────────────────
    headline_pos = ins.get('headline_positive', '')
    headline_neg = ins.get('headline_negative', '')
    verdict_label, verdict_desc_text, verdict_cls = get_verdict(pct_pos)

    headline_chips = []
    if headline_pos:
        headline_chips.append(html.Div(
            className='verdict-headline-chip positif', children=[
                html.Span('✓', className='hc-icon'),
                headline_pos,
            ]))
    if headline_neg:
        headline_chips.append(html.Div(
            className='verdict-headline-chip negatif', children=[
                html.Span('!', className='hc-icon'),
                headline_neg,
            ]))

    v_left_children = [
        html.Div(verdict_label, className='verdict-text'),
        html.Div(verdict_desc_text, className='verdict-desc'),
        html.Div(f'Total {total:,} ulasan dari {label}',
                 className='verdict-total'),
    ]
    if headline_chips:
        v_left_children.append(
            html.Div(className='verdict-headlines', children=headline_chips))

    v_left = html.Div(className=f'verdict-left-inner {verdict_cls}',
                       children=v_left_children)

    v_right = html.Div(className='verdict-metrics', children=[
        _metric_mini(f'{n_pos:,}', f'{pct_pos:.1f}%', 'Positif', 'positif'),
        _metric_mini(f'{n_neg:,}', f'{pct_neg:.1f}%', 'Negatif', 'negatif'),
        _metric_mini(f'{n_neu:,}', f'{pct_neu:.1f}%', 'Netral', 'netral'),
    ])

    # ── ROW 3: Quick stats ───────────────────────────────────────────────
    aspects_list = [a for a in ASPECT_ORDER if a != 'Lainnya' and a in asp]
    good_aspects = sum(1 for a in aspects_list
                       if asp[a].get('positivity_rate', 0) >= 75)

    if aspects_list:
        weakest_name, weakest_rate = min(
            ((a, asp[a]['positivity_rate']) for a in aspects_list),
            key=lambda x: x[1])
    else:
        weakest_name, weakest_rate = 'N/A', 0

    # Rank only for single specific desa
    if not is_semua and single_key:
        rankings = sorted(
            [(k, v.get('sentiment_pct', {}).get('Positif', 0))
             for k, v in insight_data.items() if k != 'Semua'],
            key=lambda x: x[1], reverse=True)
        rank = next((i + 1 for i, (k, _) in enumerate(rankings)
                     if k == single_key), None)
        stat4 = _stat_card(f'#{rank}', 'Peringkat Desa',
                           f'dari {len(rankings)} desa', '', '🏆')
    else:
        n_desa = len(desa_list) if not is_semua else len(DESA_LIST)
        stat4 = _stat_card(f'{pct_pos:.1f}%', 'Rata-rata Kepuasan',
                           f'{n_desa} desa wisata', '', '🎯')

    stats = [
        _stat_card(f'{total:,}', 'Total Ulasan', label, '', '📊'),
        _stat_card(f'{good_aspects}', 'Aspek Positif',
                   f'dari {len(aspects_list)} aspek', 'positif', '✅'),
        _stat_card(f'{weakest_rate:.0f}%', 'Aspek Terlemah',
                   weakest_name, 'negatif', '⚡'),
        stat4,
    ]

    # ── ROW 4: Praised / Criticized ──────────────────────────────────────
    show_village = is_semua or (not single_key)  # show village names in multi/semua

    praised = ins.get('top_praised', [])
    praised_children = []
    for i, p in enumerate(praised[:3], 1):
        cat = p['category']
        rate = asp.get(cat, {}).get('positivity_rate', 0)
        kws = p.get('opinions', [])[:4]
        raw_samples = _pick_sample_reviews(d, cat, 'Positif', n=3)
        samples = raw_samples if raw_samples else None
        if samples and not show_village:
            samples = [(s[0], '') for s in samples]
        attributions = None
        if is_semua and kws:
            attributions = {kw: _get_village_attribution(kw, cat, 'Positif')
                           for kw in kws}
        praised_children.append(_insight_item(
            rank=i, category=cat,
            count=p['count'],
            count_label='ulasan positif',
            keywords=kws,
            positivity_rate=rate,
            sample_reviews=samples,
            review_css='positif',
            keyword_attributions=attributions,
        ))
    if not praised_children:
        praised_children = html.P('Data belum tersedia.',
                                  style={'color': '#a0aec0'})

    criticized = ins.get('top_criticized', [])
    criticized_children = []
    for i, c in enumerate(criticized[:3], 1):
        cat = c['category']
        info = asp.get(cat, {})
        total_a = info.get('total', 0)
        neg_rate = (info.get('Negatif', 0) / total_a * 100) if total_a else 0
        kws = c.get('opinions', [])[:4]
        raw_samples = _pick_sample_reviews(d, cat, 'Negatif', n=3)
        samples = raw_samples if raw_samples else None
        if samples and not show_village:
            samples = [(s[0], '') for s in samples]
        attributions = None
        if is_semua and kws:
            attributions = {kw: _get_village_attribution(kw, cat, 'Negatif')
                           for kw in kws}
        criticized_children.append(_insight_item(
            rank=i, category=cat,
            count=c['count'],
            count_label='keluhan',
            keywords=kws,
            negativity_rate=neg_rate,
            sample_reviews=samples,
            review_css='negatif',
            keyword_attributions=attributions,
        ))
    if not criticized_children:
        criticized_children = html.P('Data belum tersedia.',
                                     style={'color': '#a0aec0'})

    # ── ROW 5 LEFT: Aspect health grid ───────────────────────────────────
    grid_children = []
    for a in ASPECT_ORDER:
        if a == 'Lainnya' or a not in asp:
            continue
        info = asp[a]
        rate = info.get('positivity_rate', 0)
        total_a = info.get('total', 0)
        h_cls = aspect_health_class(rate)
        if single_key:
            op = opinion_data.get(single_key, {}).get(a, {})
        else:
            op = _merge_opinion_data(desa_list, a)
        top_words = (op.get('Positif', [])[:2] + op.get('Negatif', [])[:1])[:3]

        icon = ASPECT_ICONS.get(a, '◈')
        grid_children.append(html.Div(className=f'aspect-health-card {h_cls}', children=[
            html.Div(className='aspect-health-header', children=[
                html.Div(icon, className='aspect-health-icon'),
                html.Div(a, className='aspect-health-name'),
            ]),
            html.Div(className='aspect-health-bar-track', children=[
                html.Div(className=f'aspect-health-bar-fill {h_cls}',
                         style={'width': f'{rate}%'}),
            ]),
            html.Div(className='aspect-health-meta', children=[
                html.Span(f'{rate:.0f}% puas', className='aspect-health-pct'),
                html.Span(f'({total_a:,} ulasan)', className='aspect-health-count'),
            ]),
            html.Div(', '.join(top_words),
                      className='aspect-health-words') if top_words else None,
        ]))

    # ── ROW 5 RIGHT-A: TF-IDF keywords ──────────────────────────────────
    if single_key:
        tfidf_key = f'{single_key}_Positif'
        tfidf_words = tfidf_data.get(tfidf_key, [])[:6]
    else:
        tfidf_words = []

    if tfidf_words:
        tfidf_children = [
            html.H2('Kata Paling Khas', className='section-title'),
            html.P('Kata yang paling sering muncul di review desa ini, '
                   'namun jarang muncul di desa lain (skor TF-IDF tertinggi).',
                   className='section-caption'),
            html.Div(className='keyword-tags', children=[
                html.Span(w[0], className='keyword-tag-sm') for w in tfidf_words
            ]),
        ]
    else:
        tfidf_children = [
            html.H2('Kata Paling Khas', className='section-title'),
            html.P('Pilih satu desa untuk melihat kata khas.'
                   if not single_key else 'Data belum tersedia.',
                   style={'color': '#a0aec0', 'fontSize': '13px'}),
        ]

    # ── ROW 5 RIGHT-B: Sample reviews ────────────────────────────────────
    review_children = [
        html.H2('Contoh Ulasan', className='section-title'),
    ]
    d_short = d[d['cleaned_review'].str.len().between(20, 150)]
    for sent, cls in [('Positif', 'positif'), ('Negatif', 'negatif')]:
        subset = d_short[d_short['sentiment'] == sent]
        if len(subset) > 0:
            sample = subset.sample(1, random_state=42).iloc[0]
            review_children.append(
                _review_quote(sample['cleaned_review'], cls))
    if len(review_children) == 1:
        review_children.append(
            html.P('Tidak cukup data.', style={'color': '#a0aec0'}))

    return (
        v_left,
        v_right,
        stats,
        praised_children,
        criticized_children,
        grid_children,
        tfidf_children,
        review_children,
    )


# ── Page 1 helper components ────────────────────────────────────────────────

def _metric_mini(value, pct, label, css_cls):
    """Mini metric card for verdict-right area."""
    return html.Div(className=f'verdict-metric-card {css_cls}', children=[
        html.Div(value, className='verdict-metric-value'),
        html.Div(pct, className='verdict-metric-pct'),
        html.Div(label, className='verdict-metric-label'),
    ])


def _stat_card(value, label, sublabel, css_cls, icon=''):
    """Quick stat card for ROW 3."""
    children = []
    if icon:
        children.append(html.Div(icon, className='metric-icon'))
    children += [
        html.Div(value, className='metric-value'),
        html.Div(label, className='metric-label'),
        html.Div(sublabel, className='metric-sub'),
    ]
    return html.Div(className=f'metric-card {css_cls}', children=children)


def _review_quote(text, css_cls, village_name=None):
    """Blockquote-style review sample."""
    display = text if len(text) <= 150 else text[:147] + '...'
    children = [html.Div(f'"{display}"', className='review-quote-text')]
    if village_name:
        children.append(
            html.Div(f'— {village_name.title()}',
                      className='review-quote-source'))
    return html.Div(className=f'review-quote {css_cls}', children=children)


def _pick_sample_reviews(d, category, sentiment, n=3):
    """Pick up to n representative reviews matching category + per-aspect sentiment."""
    subset = d[aspect_sentiment_mask(d, category, sentiment)]
    if subset.empty:
        return []
    medium = subset[subset['cleaned_review'].str.len().between(20, 200)]
    src = medium if not medium.empty else subset
    count = min(n, len(src))
    rows = src.sample(count, random_state=42)
    return [(row['cleaned_review'], row.get('nama desa wisata', ''))
            for _, row in rows.iterrows()]


def _get_village_attribution(keyword, category, sentiment):
    """Find which villages contain this keyword for the given aspect+sentiment."""
    villages = []
    for village_key in DESA_LIST:
        words = opinion_data.get(village_key, {}).get(category, {}).get(
            sentiment, [])
        if keyword in words:
            villages.append(village_key)
    return villages


def _build_keywords_display(keywords, attributions=None, css_variant='positif'):
    """Render keywords as colored chips, optionally with village tooltips."""
    if not keywords:
        return None
    chips = []
    for kw in keywords:
        villages = (attributions or {}).get(kw, [])
        if villages:
            tooltip = ', '.join(v.title() for v in villages)
            chips.append(html.Span(
                className='keyword-tooltip-wrapper', children=[
                    html.Span(kw, className=f'keyword-chip {css_variant} attributed'),
                    html.Span(f'Desa: {tooltip}',
                              className='keyword-tooltip-content'),
                ]))
        else:
            chips.append(html.Span(kw, className=f'keyword-chip {css_variant}'))
    return html.Div(chips, className='keyword-chips')


def _insight_item(rank, category, count, count_label, keywords,
                  positivity_rate=None, negativity_rate=None,
                  sample_reviews=None, review_css='positif',
                  keyword_attributions=None):
    """Build one ranked insight item row with expandable reviews."""
    icon = ASPECT_ICONS.get(category, '◈')

    header = html.Div(className='insight-item-header', children=[
        html.Div(className=f'insight-item-aspect-icon {review_css}',
                 children=icon),
        html.Div(className='insight-item-title', children=[
            html.Div(category, className='insight-item-category'),
            html.Div(count_label, className='insight-item-sublabel'),
        ]),
        html.Div(className=f'insight-item-count-badge {review_css}', children=[
            html.Div(f'{count:,}', className='insight-item-count-num'),
        ]),
    ])

    body_children = [header]

    if positivity_rate is not None:
        body_children.append(html.Div(className='insight-item-bar-row', children=[
            html.Div(className='mini-bar-track', children=[
                html.Div(className=f'mini-bar-fill {aspect_health_class(positivity_rate)}',
                         style={'width': f'{positivity_rate}%'}),
            ]),
            html.Span(f'{positivity_rate:.0f}% puas',
                      className='mini-bar-label'),
        ]))
    elif negativity_rate is not None:
        body_children.append(html.Div(className='insight-item-bar-row', children=[
            html.Div(className='mini-bar-track', children=[
                html.Div(className='mini-bar-fill health-poor',
                         style={'width': f'{negativity_rate}%'}),
            ]),
            html.Span(f'{negativity_rate:.0f}% negatif',
                      className='mini-bar-label'),
        ]))

    if keywords:
        body_children.append(
            _build_keywords_display(keywords, keyword_attributions, review_css))

    if sample_reviews:
        first = sample_reviews[0]
        body_children.append(
            _review_quote(first[0], review_css,
                          village_name=first[1] if first[1] else None))
        if len(sample_reviews) > 1:
            extra = [_review_quote(sr[0], review_css,
                                   village_name=sr[1] if sr[1] else None)
                     for sr in sample_reviews[1:]]
            body_children.append(
                html.Details(className='extra-reviews', children=[
                    html.Summary('Lihat ulasan lainnya',
                                 className='extra-reviews-toggle'),
                    html.Div(extra, className='extra-reviews-list'),
                ]))

    children = [
        html.Div(str(rank), className=f'insight-item-rank rank-{rank}'),
        html.Div(className='insight-item-body', children=body_children),
    ]
    return html.Div(className=f'insight-item {review_css}', children=children)


# ── Carousel: index state + view render ─────────────────────────────────────
@callback(
    Output('carousel-index', 'data'),
    Input('carousel-prev', 'n_clicks'),
    Input('carousel-next', 'n_clicks'),
    Input('desa-dropdown', 'value'),
    State('carousel-index', 'data'),
    prevent_initial_call=False,
)
def update_carousel_index(prev_clicks, next_clicks, desa, idx):
    from dash import ctx
    triggered = ctx.triggered_id
    desa_list, _, is_semua = _normalize_desa(desa)
    entries = _build_carousel_entries(desa_list, is_semua)
    total = len(entries)
    if total == 0:
        return 0
    if triggered == 'desa-dropdown':
        return 0
    if triggered == 'carousel-prev':
        return ((idx or 0) - 1) % total
    if triggered == 'carousel-next':
        return ((idx or 0) + 1) % total
    return idx or 0


@callback(
    Output('carousel-img', 'src'),
    Output('carousel-caption', 'children'),
    Output('carousel-counter', 'children'),
    Output('carousel-dots', 'children'),
    Output('carousel-section', 'style'),
    Input('desa-dropdown', 'value'),
    Input('carousel-index', 'data'),
)
def update_carousel_view(desa, idx):
    desa_list, _, is_semua = _normalize_desa(desa)
    entries = _build_carousel_entries(desa_list, is_semua)
    total = len(entries)
    if total == 0:
        return '', '', '', [], {'display': 'none'}
    idx = (idx or 0) % total
    src, caption = entries[idx]
    counter = f'{idx + 1} / {total}'
    dots = []
    if total <= 12:
        dots = [
            html.Span(className=f'carousel-dot{" active" if i == idx else ""}')
            for i in range(total)
        ]
    return src, caption, counter, dots, {}


# ══════════════════════════════════════════════════════════════════════════════
# VISITOR MODE CALLBACKS (Pengunjung)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output('visitor-gallery-grid', 'children'),
    Output('visitor-result-count', 'children'),
    Input('visitor-prov-filter', 'value'),
    Input('visitor-aspect-filter', 'value'),
    Input('visitor-sort', 'value'),
    Input('visitor-search', 'value'),
)
def update_visitor_gallery(provinsi, aspect, sort_key, search):
    if provinsi and provinsi != 'Semua':
        desas = list(PROVINSI_DESA.get(provinsi, []))
    else:
        desas = list(DESA_LIST)
    if search:
        s = search.strip().lower()
        desas = [d for d in desas if s in d.lower()]

    summaries = [_visitor_card_summary(d) for d in desas]
    aspect_active = bool(aspect) and aspect != 'Semua'
    if aspect_active:
        # Hanya desa yang benar-benar membahas aspek tersebut
        summaries = [c for c in summaries if c['aspect_totals'].get(aspect, 0) > 0]

    def _kepuasan(c):
        return c['aspect_rates'].get(aspect, 0) if aspect_active else c['pct_pos']

    if sort_key == 'ulasan':
        summaries.sort(key=lambda c: c['total'], reverse=True)
    elif sort_key == 'nama':
        summaries.sort(key=lambda c: c['desa'].lower())
    else:  # 'kepuasan' (default) — pakai skor aspek bila filter aspek aktif
        summaries.sort(key=_kepuasan, reverse=True)

    n = len(summaries)
    if n == 0:
        return (html.P('Tidak ada desa wisata yang cocok dengan pencarian.',
                       className='visitor-empty'), '')
    suffix = f' yang unggul di {aspect}' if aspect_active else ''
    count_text = f'Menampilkan {n} destinasi{suffix}'
    cards = [_visitor_card(c, aspect if aspect_active else None) for c in summaries]
    return cards, count_text


@callback(
    Output('visitor-carousel-img', 'src'),
    Output('visitor-carousel-counter', 'children'),
    Output('visitor-carousel-index', 'data'),
    Input('visitor-carousel-prev', 'n_clicks'),
    Input('visitor-carousel-next', 'n_clicks'),
    State('visitor-carousel-index', 'data'),
    State('url', 'search'),
    prevent_initial_call=True,
)
def update_visitor_carousel(prev_clicks, next_clicks, idx, search):
    from dash import ctx
    desa = _desa_from_search(search)
    entries = _build_carousel_entries([desa], False) if desa else []
    total = len(entries)
    if total == 0:
        return '', '', 0
    idx = idx or 0
    if ctx.triggered_id == 'visitor-carousel-prev':
        idx = (idx - 1) % total
    elif ctx.triggered_id == 'visitor-carousel-next':
        idx = (idx + 1) % total
    src, _ = entries[idx]
    return src, f'{idx + 1} / {total}', idx


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 CALLBACKS (Detail) — unchanged from original
# ══════════════════════════════════════════════════════════════════════════════

# ── Overview: Grouped bar + Pie ──────────────────────────────────────────────
@callback(Output('grouped-bar', 'figure'), Input('desa-dropdown', 'value'))
def update_grouped_bar(desa):
    d = filter_df(desa)
    pivot = d.groupby(['nama desa wisata', 'sentiment']).size().reset_index(name='count')
    pivot['nama desa wisata'] = pivot['nama desa wisata'].str.title()

    fig = px.bar(
        pivot, x='nama desa wisata', y='count', color='sentiment',
        color_discrete_map=COLORS, barmode='group',
        labels={'nama desa wisata': '', 'count': 'Jumlah Review',
                'sentiment': 'Sentimen'},
        title='Distribusi Sentimen per Desa Wisata',
        category_orders={'sentiment': ['Positif', 'Negatif', 'Netral']},
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=4),
        hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>',
    )

    fig = apply_chart_theme(fig, height=380)
    fig.update_layout(xaxis_tickangle=-18, legend_title_text='',
                      margin=dict(t=58, b=78, l=14, r=14))
    return fig


@callback(Output('pie-chart', 'figure'), Input('desa-dropdown', 'value'))
def update_pie(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    d = filter_df(desa)
    counts = d['sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment', 'count']
    total = int(counts['count'].sum()) if not counts.empty else 0
    title = _desa_label(desa_list, is_semua).title()

    fig = px.pie(
        counts, names='sentiment', values='count',
        color='sentiment', color_discrete_map=COLORS,
        title=f'Proporsi Sentimen — {title}', hole=0.62,
        category_orders={'sentiment': ['Positif', 'Negatif', 'Netral']},
    )
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        textfont=dict(family=THEME_FONT, size=12, color=THEME_TITLE_COLOR),
        marker=dict(line=dict(color='white', width=3)),
        hovertemplate='<b>%{label}</b><br>%{value} ulasan (%{percent})<extra></extra>',
        sort=False,
    )
    fig = apply_chart_theme(fig, show_legend=False, height=420,
                            hide_xaxis=True, hide_yaxis=True)
    fig.update_layout(
        title=dict(y=0.98, yanchor='top', pad=dict(t=0, b=12)),
        margin=dict(t=92, b=42, l=56, r=56),
        annotations=[
            dict(text=f'<b>{total:,}</b>', x=0.5, y=0.54, showarrow=False,
                 font=dict(family=THEME_TITLE_FONT, size=32,
                           color=THEME_TITLE_COLOR)),
            dict(text='TOTAL ULASAN', x=0.5, y=0.42, showarrow=False,
                 font=dict(family=THEME_FONT, size=10, color=THEME_AXIS_COLOR)),
        ],
    )
    return fig


# ── Aspect Analysis ──────────────────────────────────────────────────────────
@callback(Output('aspect-bar', 'figure'), Input('desa-dropdown', 'value'))
def update_aspect_bar(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    if single_key:
        asp = aspect_data.get(single_key, {})
    else:
        asp = _compute_aspect_data(filter_df(desa))
    if not asp:
        fig = go.Figure()
        fig.add_annotation(text='Data aspek belum tersedia', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=420,
                                 hide_xaxis=True, hide_yaxis=True)

    rows = []
    for a in ASPECT_ORDER:
        if a in asp:
            for sent in ['Positif', 'Negatif', 'Netral']:
                rows.append({'Aspek': a, 'Sentimen': sent,
                             'Jumlah': asp[a].get(sent, 0)})

    bar_df = pd.DataFrame(rows)
    aspect_cats = [a for a in ASPECT_ORDER if a in asp]

    fig = px.bar(
        bar_df, y='Aspek', x='Jumlah', color='Sentimen', orientation='h',
        color_discrete_map=COLORS, barmode='stack',
        title='Sentimen per Topik yang Dibicarakan',
        category_orders={'Aspek': aspect_cats,
                         'Sentimen': ['Positif', 'Negatif', 'Netral']},
        text='Jumlah',
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=3),
        textposition='inside',
        textfont=dict(family=THEME_FONT, size=10, color='white'),
        texttemplate='%{text}',
        hovertemplate='<b>%{y}</b><br>%{fullData.name}: %{x}<extra></extra>',
    )
    fig = apply_chart_theme(fig, height=420, show_xaxis_grid=True,
                            show_yaxis_grid=False)
    fig.update_layout(
        legend_title_text='',
        yaxis=dict(autorange='reversed'),
        bargap=0.35,
        margin=dict(t=58, b=52, l=14, r=80),
    )
    fig.update_xaxes(automargin=True)
    # Total count annotation at end of each stacked bar
    totals = {a: asp[a].get('total', 0) for a in aspect_cats}
    max_total = max(totals.values()) if totals else 0
    if max_total:
        fig.update_xaxes(range=[0, max_total * 1.12])
    for a in aspect_cats:
        fig.add_annotation(
            x=totals[a], y=a, xshift=10, text=f'<b>{totals[a]:,}</b>',
            showarrow=False, xanchor='left',
            font=dict(family=THEME_FONT, size=11, color=THEME_TITLE_COLOR),
        )
    return fig


@callback(
    Output('opinion-bar', 'figure'),
    Input('desa-dropdown',   'value'),
    Input('aspect-dropdown', 'value'),
)
def update_opinion_bar(desa, aspect):
    import re
    desa_list, single_key, is_semua = _normalize_desa(desa)
    if single_key:
        opinions = opinion_data.get(single_key, {}).get(aspect, {})
    else:
        opinions = _merge_opinion_data(desa_list, aspect)
    if not opinions:
        fig = go.Figure()
        fig.add_annotation(text='Belum ada data opini', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=350,
                                 hide_xaxis=True, hide_yaxis=True)

    # Count real occurrences in reviews matching aspect + per-aspect sentiment
    d = filter_df(desa)

    rows = []
    for sent in ['Positif', 'Negatif', 'Netral']:
        words = opinions.get(sent, [])[:12]
        if not words:
            continue
        d_sent = d[aspect_sentiment_mask(d, aspect, sent)]
        # Match terhadap kolom stemmed (opinion_data dihasilkan dari teks stemmed
        # juga). Kalau di-match ke un-stem, semua frekuensi akan 0.
        text = ' '.join(d_sent[ANALYTICS_COL].dropna().astype(str)).lower()
        for word in words:
            pattern = r'\b' + re.escape(word.lower()) + r'\b'
            freq = len(re.findall(pattern, text))
            if freq > 0:
                rows.append({'Kata': word, 'Sentimen': sent, 'Frekuensi': freq})

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text='Belum ada data', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=350,
                                 hide_xaxis=True, hide_yaxis=True)

    word_counts = (pd.DataFrame(rows)
                   .sort_values('Frekuensi', ascending=False)
                   .drop_duplicates(subset=['Kata'], keep='first')
                   .head(10)
                   .iloc[::-1]
                   .reset_index(drop=True))
    word_counts['Sentimen'] = word_counts['Sentimen'].astype(str)
    word_counts['Frekuensi'] = word_counts['Frekuensi'].astype(int)

    max_freq = int(word_counts['Frekuensi'].max())
    fig = go.Figure()
    for sent in ['Positif', 'Negatif', 'Netral']:
        sub = word_counts[word_counts['Sentimen'] == sent]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub['Frekuensi'].tolist(),
            y=sub['Kata'].tolist(),
            orientation='h',
            name=sent,
            marker=dict(color=COLORS.get(sent, '#475569'),
                        line=dict(width=0), cornerradius=6),
            text=sub['Frekuensi'].tolist(),
            textposition='outside',
            textfont=dict(family=THEME_FONT, size=11, color=THEME_TITLE_COLOR),
            hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>',
        ))
    fig.update_layout(
        title=f'Opini — {aspect}',
        yaxis=dict(categoryorder='array',
                   categoryarray=word_counts['Kata'].tolist()),
    )
    fig = apply_chart_theme(fig, show_legend=False, height=350,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(
        bargap=0.32,
        margin=dict(t=58, b=30, l=14, r=72),
        xaxis=dict(range=[0, max_freq * 1.22]),
    )
    fig.update_xaxes(automargin=True)
    return fig


# ── Word Analysis (TF-IDF + Wordcloud + Bigrams, side-by-side) ──────────────
@callback(
    Output('tfidf-bar', 'figure'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_tfidf_bar(desa, sentiment):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    if single_key:
        keywords = tfidf_data.get(f'{single_key}_{sentiment}', [])
    else:
        keywords = _compute_tfidf_on_fly(desa_list, is_semua, sentiment)
    if not keywords:
        fig = go.Figure()
        fig.add_annotation(text='Data TF-IDF belum tersedia',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=THEME_AXIS_COLOR, size=13))
        return apply_chart_theme(fig, show_legend=False, height=420,
                                 hide_xaxis=True, hide_yaxis=True)

    kw_df = pd.DataFrame(keywords[:10], columns=['Kata Kunci', 'Skor TF-IDF'])
    kw_df = kw_df.iloc[::-1]
    max_score = float(kw_df['Skor TF-IDF'].max()) if not kw_df.empty else 0
    color = COLORS.get(sentiment, '#475569')

    fig = px.bar(
        kw_df, x='Skor TF-IDF', y='Kata Kunci', orientation='h',
        color_discrete_sequence=[color],
        title=f'Kata Kunci Teratas — {sentiment}',
        text='Skor TF-IDF',
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=6),
        textposition='outside',
        texttemplate='%{x:.2f}',
        textfont=dict(family=THEME_FONT, size=10, color=THEME_TITLE_COLOR),
        hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>',
    )
    fig = apply_chart_theme(fig, show_legend=False, height=380,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(
        bargap=0.35,
        margin=dict(t=58, b=30, l=14, r=72),
        xaxis=dict(range=[0, max_score * 1.22]) if max_score else {},
    )
    fig.update_xaxes(automargin=True)
    return fig


@callback(
    Output('wordcloud-img', 'src'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_wordcloud(desa, sentiment):
    base = COLORS.get(sentiment, '#475569')
    light = COLORS_SOFT.get(sentiment, '#94a3b8')
    # Phrase-based: kata per-klausa dengan sentimen klausa sendiri (bagus tidak
    # bocor ke cloud negatif). Fallback ke teks review bila JSON belum di-generate.
    freq = _phrase_freq(desa, sentiment, 'words')
    if freq:
        return make_wordcloud_from_freq(freq, base, light)
    d = filter_df(desa)
    subset = d[d['sentiment'] == sentiment][ANALYTICS_COL].dropna().tolist()
    return make_wordcloud(subset, base, light)


@callback(
    Output('bigram-chart', 'figure'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_bigram(desa, sentiment):
    color = COLORS.get(sentiment, '#475569')
    # Phrase-based: bigram per-klausa dengan sentimen klausa sendiri. Fallback ke
    # teks review bila JSON belum di-generate.
    freq = _phrase_freq(desa, sentiment, 'bigrams')
    if freq:
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        bg_df = pd.DataFrame(top[::-1], columns=['bigram', 'count'])
    else:
        d = filter_df(desa)
        subset = d[d['sentiment'] == sentiment][ANALYTICS_COL].dropna().tolist()
        bg_df = top_bigrams(subset, n=10)
    if bg_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Tidak ada data', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=420,
                                 hide_xaxis=True, hide_yaxis=True)
    max_count = int(bg_df['count'].max()) if not bg_df.empty else 0
    fig = px.bar(
        bg_df, x='count', y='bigram', orientation='h',
        color_discrete_sequence=[color],
        labels={'count': 'Frekuensi', 'bigram': ''},
        title=f'Frasa Populer — {sentiment}',
        text='count',
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=6),
        textposition='outside',
        textfont=dict(family=THEME_FONT, size=11, color=THEME_TITLE_COLOR),
        hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>',
    )
    fig = apply_chart_theme(fig, show_legend=False, height=380,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(
        bargap=0.35,
        margin=dict(t=58, b=30, l=14, r=90),
        xaxis=dict(range=[0, max_count * 1.35]) if max_count else {},
    )
    fig.update_xaxes(automargin=True)
    return fig


# ── Comparison: Radar + Heatmap ──────────────────────────────────────────────
@callback(Output('radar-chart', 'figure'), Input('desa-dropdown', 'value'))
def update_radar(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    desa_to_show = DESA_LIST if is_semua else desa_list
    aspects_for_radar = [a for a in ASPECT_ORDER if a != 'Lainnya']

    fig = go.Figure()
    palette = ['#0c1e36', '#f59e0b', '#10b981', '#ef4444', '#6366f1',
               '#0ea5e9', '#d946ef', '#14b8a6', '#f97316', '#84cc16']
    for idx, d_name in enumerate(desa_to_show):
        asp = aspect_data.get(d_name)
        if not asp:
            asp = _compute_aspect_data(df[df['nama desa wisata'] == d_name])
        values = []
        for a in aspects_for_radar:
            if a in asp and asp[a]['total'] > 0:
                values.append(asp[a]['positivity_rate'])
            else:
                values.append(0)
        values.append(values[0])
        labels = aspects_for_radar + [aspects_for_radar[0]]
        color = palette[idx % len(palette)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels,
            fill='toself', name=d_name.title(),
            fillcolor=f'rgba({r},{g},{b},0.27)',
            line=dict(color=color, width=2.5),
            hovertemplate='<b>' + d_name.title() + '</b><br>%{theta}: %{r:.1f}%<extra></extra>',
        ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(248,250,252,.6)',
            radialaxis=dict(
                visible=True, range=[0, 100], ticksuffix='%',
                tickfont=dict(color=THEME_AXIS_COLOR, size=10),
                gridcolor=THEME_GRID_COLOR, linecolor=THEME_GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color=THEME_TITLE_COLOR, size=11,
                              family=THEME_FONT),
                gridcolor=THEME_GRID_COLOR, linecolor=THEME_GRID_COLOR,
            ),
        ),
        title=dict(
            text='Tingkat Kepuasan per Aspek (%)',
            font=dict(family=THEME_TITLE_FONT, size=16, color=THEME_TITLE_COLOR),
            x=0.02, xanchor='left', y=0.97, pad=dict(t=4, b=4),
        ),
        font=dict(family=THEME_FONT, size=12, color=THEME_TEXT_COLOR),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=70, b=70, l=70, r=70),
        legend=dict(
            orientation='h', yanchor='top', y=-0.08,
            xanchor='center', x=0.5,
            font=dict(size=10, color=THEME_TEXT_COLOR),
            bgcolor='rgba(0,0,0,0)',
        ),
        hoverlabel=dict(bgcolor='white', bordercolor=THEME_BORDER,
                        font=dict(family=THEME_FONT, color=THEME_TITLE_COLOR)),
    )
    return fig


@callback(Output('heatmap-chart', 'figure'), Input('desa-dropdown', 'value'))
def update_heatmap(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    desa_to_show = DESA_LIST if is_semua else desa_list
    aspects_for_hm = [a for a in ASPECT_ORDER if a != 'Lainnya']

    matrix = []
    for d_name in desa_to_show:
        row = []
        asp = aspect_data.get(d_name)
        if not asp:
            asp = _compute_aspect_data(df[df['nama desa wisata'] == d_name])
        for a in aspects_for_hm:
            if a in asp and asp[a]['total'] > 0:
                row.append(asp[a]['positivity_rate'])
            else:
                row.append(None)
        matrix.append(row)

    # Custom 3-stop diverging: rose → soft cream → emerald
    custom_scale = [
        [0.0, '#ef4444'], [0.3, '#fca5a5'], [0.5, '#fef3c7'],
        [0.7, '#86efac'], [1.0, '#10b981'],
    ]

    text_matrix = []
    text_color_matrix = []
    for row in matrix:
        tr, cr = [], []
        for v in row:
            if v is None:
                tr.append('–')
                cr.append(THEME_AXIS_COLOR)
            else:
                tr.append(f'{v:.0f}%')
                # white text on very high/low, dark text on mid values
                cr.append('white' if (v >= 70 or v <= 30) else THEME_TITLE_COLOR)
        text_matrix.append(tr)
        text_color_matrix.append(cr)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[a for a in aspects_for_hm],
        y=[d.title() for d in desa_to_show],
        colorscale=custom_scale,
        zmin=0, zmax=100,
        text=text_matrix,
        texttemplate='<b>%{text}</b>',
        textfont=dict(family=THEME_FONT, size=13),
        xgap=3, ygap=3,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
        colorbar=dict(
            title=dict(text='% Positif',
                       font=dict(family=THEME_FONT, size=11,
                                 color=THEME_AXIS_COLOR)),
            tickfont=dict(family=THEME_FONT, size=10, color=THEME_AXIS_COLOR),
            thickness=10, len=0.75, outlinewidth=0,
        ),
    ))
    fig.update_layout(
        title=dict(
            text='Peta Kepuasan per Aspek (%)',
            font=dict(family=THEME_TITLE_FONT, size=16, color=THEME_TITLE_COLOR),
            x=0.02, xanchor='left', y=0.97, pad=dict(t=4, b=4),
        ),
        font=dict(family=THEME_FONT, size=11, color=THEME_TEXT_COLOR),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=480,
        margin=dict(t=58, b=90, l=14, r=14),
        xaxis=dict(tickangle=-28, showgrid=False, zeroline=False,
                   tickfont=dict(color=THEME_AXIS_COLOR, size=12)),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(color=THEME_AXIS_COLOR, size=12)),
        hoverlabel=dict(bgcolor='white', bordercolor=THEME_BORDER,
                        font=dict(family=THEME_FONT, color=THEME_TITLE_COLOR)),
    )
    return fig


# ── Review Table ─────────────────────────────────────────────────────────────
@callback(
    Output('review-table', 'data'),
    Input('desa-dropdown',          'value'),
    Input('table-sentiment-filter', 'value'),
    Input('table-aspect-filter',    'value'),
)
def update_table(desa, sentiment_filter, aspect_filter):
    d = filter_df(desa)
    # Kalau dua-duanya difilter: pakai sentimen per-aspek supaya konsisten
    # dengan section 'Perlu Diperbaiki' (mis. 'Negatif + Kebersihan' = review
    # yang aspek Kebersihan-nya negatif, bukan review yang overall Negatif &
    # kebetulan menyebut Kebersihan).
    if sentiment_filter != 'Semua' and aspect_filter != 'Semua':
        d = d[aspect_sentiment_mask(d, aspect_filter, sentiment_filter)]
    elif sentiment_filter != 'Semua':
        d = d[d['sentiment'] == sentiment_filter]
    elif aspect_filter != 'Semua':
        d = d[d['aspects_list'].apply(lambda x: aspect_filter in x)]
    cols = ['nama desa wisata', 'sentiment', 'aspects_str', 'cleaned_review']
    return d[cols].dropna(subset=['cleaned_review']).head(500).to_dict('records')


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # host/port via env supaya identik dgn versi deploy; gunicorn (deploy) tidak
    # memakai blok ini — ia mengimpor `server`. debug hanya berlaku saat run lokal.
    port = int(os.environ.get('PORT', ARGS.port))
    app.run(host='0.0.0.0', port=port, debug=True)
