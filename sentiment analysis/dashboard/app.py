# -*- coding: utf-8 -*-
import os
import io
import json
import base64
from collections import Counter

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output, dash_table, callback
import plotly.express as px
import plotly.graph_objects as go

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(DATA_DIR, 'sentiment_labeled.csv'))

# Parse aspects_str back to list
if 'aspects_str' in df.columns:
    df['aspects_list'] = df['aspects_str'].fillna('Lainnya').str.split('|')
else:
    df['aspects_list'] = [['Lainnya']] * len(df)

# Load pre-computed JSON files
def load_json(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

tfidf_data   = load_json('tfidf_keywords.json')
aspect_data  = load_json('aspect_summary.json')
insight_data = load_json('insights_summary.json')
opinion_data = load_json('opinion_words.json')

LABEL_NAME = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
COLORS     = {'Positif': '#10b981', 'Negatif': '#ef4444', 'Netral': '#6366f1'}
COLORS_SOFT = {'Positif': '#34d399', 'Negatif': '#f87171', 'Netral': '#818cf8'}
DESA_LIST  = sorted(df['nama desa wisata'].dropna().unique().tolist())

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
    """Compute aspect summary on-the-fly from a filtered dataframe."""
    result = {}
    for aspect in ASPECT_ORDER:
        mask = d['aspects_str'].str.contains(aspect, na=False)
        subset = d[mask]
        if subset.empty:
            continue
        total = len(subset)
        pos = int((subset['sentiment'] == 'Positif').sum())
        neg = int((subset['sentiment'] == 'Negatif').sum())
        neu = int((subset['sentiment'] == 'Netral').sum())
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


def _merge_opinion_data(desa_list, aspect):
    """Merge opinion words from multiple villages for a given aspect."""
    merged = {}
    for dk in desa_list:
        village_op = opinion_data.get(dk, {}).get(aspect, {})
        for sent, words in village_op.items():
            merged.setdefault(sent, []).extend(words)
    return {sent: list(dict.fromkeys(words)) for sent, words in merged.items()}


def make_wordcloud(texts, base_color, light_color):
    text = ' '.join(texts)
    if not text.strip():
        return ''

    def _color_func(word, font_size, position, orientation, random_state=None, **kw):
        # Largest words get darker shade, smaller words get lighter — adds depth
        return base_color if font_size > 36 else light_color

    wc = WordCloud(
        width=900, height=380,
        mode='RGBA', background_color=None,
        color_func=_color_func,
        max_words=80,
        collocations=False,
        prefer_horizontal=0.95,
        min_font_size=11,
        relative_scaling=0.5,
        margin=6,
    ).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    return f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'


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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUTS
# ══════════════════════════════════════════════════════════════════════════════

def layout_insight():
    """Page 1: Ringkasan insight — padat, informatif, bahasa sederhana."""
    return html.Div(children=[

        # ── ROW 1: Headline Banner ───────────────────────────────────────
        html.Div(id='headline-banner', className='headline-banner'),

        # ── ROW 2: Verdict + Sentiment Stats (horizontal) ───────────────
        html.Div(className='verdict-row', children=[
            html.Div(id='verdict-left', className='verdict-left'),
            html.Div(id='verdict-right', className='verdict-right'),
        ]),

        # ── ROW 3: Quick Stats (4 cards) ────────────────────────────────
        html.Div(id='stats-row', className='metrics-row'),

        # ── ROW 4: Praised vs Criticized side-by-side ───────────────────
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

        # ── ROW 5: Aspect Grid + Sidebar ────────────────────────────────
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

        # ── ROW 6: CTA ──────────────────────────────────────────────────
        html.Div(className='cta-section', children=[
            dcc.Link('Lihat Grafik & Data Lengkap',
                      href='/detail', className='cta-link'),
        ]),
    ])


def layout_detail():
    """Page 2: Semua chart dan tabel detail."""
    return html.Div(children=[

        # Back link
        dcc.Link('Kembali ke Ringkasan', href='/', className='back-link'),

        # ── Gambaran Umum ────────────────────────────────────────────────
        html.Div(className='section', children=[
            html.H2('Gambaran Umum', className='section-title'),
            html.Div(className='chart-row', children=[
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
            html.Div(className='chart-row', children=[
                html.Div(className='chart-card wide', children=[
                    dcc.RadioItems(
                        id='word-view-toggle',
                        options=[
                            {'label': ' Grafik Batang', 'value': 'bar'},
                            {'label': ' Word Cloud', 'value': 'wc'},
                        ],
                        value='bar', inline=True, className='view-toggle',
                    ),
                    dcc.Graph(id='tfidf-bar', config={'displayModeBar': False},
                              style={'display': 'block'}),
                    html.Img(id='wordcloud-img', className='wordcloud-img',
                             style={'display': 'none'}),
                ]),
                html.Div(className='chart-card narrow', children=[
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
            dash_table.DataTable(
                id='review-table',
                columns=[
                    {'name': 'Desa Wisata',  'id': 'nama desa wisata'},
                    {'name': 'Sentimen',     'id': 'sentiment'},
                    {'name': 'Aspek',        'id': 'aspects_str'},
                    {'name': 'Review',       'id': 'cleaned_review'},
                ],
                page_size=15,
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
                     'rule': 'border-radius: 8px; padding: 6px 10px; '
                             'border: 1px solid #e2e8f0; font-size: 12px;'},
                ],
                sort_action='native',
                filter_action='native',
            ),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SHELL LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div(className='page', children=[
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='prev-desa', data=['Semua']),

    # ── Header ────────────────────────────────────────────────────────────
    html.Div(className='header', children=[
        html.Div(className='header-accent-bar'),
        html.Div(className='header-inner', children=[
            html.H1('Dashboard Analisis Sentimen'),
            html.P('Ulasan Desa Wisata Indonesia — IndoBERT Sentiment Analysis'),
        ]),
    ]),

    # ── Nav + Filter bar ──────────────────────────────────────────────────
    html.Div(className='nav-filter-bar', children=[
        html.Div(className='nav-tabs', children=[
            dcc.Link('Ringkasan', href='/', id='nav-insight',
                      className='nav-link active'),
            dcc.Link('Detail & Grafik', href='/detail', id='nav-detail',
                      className='nav-link'),
        ]),
        html.Div(className='filter-section', children=[
            html.Label('Filter Desa Wisata:', className='filter-label'),
            dcc.Dropdown(
                id='desa-dropdown',
                options=[{'label': 'Semua Desa Wisata', 'value': 'Semua'}] +
                        [{'label': d.title(), 'value': d} for d in DESA_LIST],
                value=['Semua'],
                multi=True,
                className='desa-dropdown modern-dropdown',
                placeholder='Pilih desa wisata...',
            ),
        ]),
    ]),

    # ── Page Content (swapped by URL) ─────────────────────────────────────
    html.Div(id='page-content'),

    # ── Footer ────────────────────────────────────────────────────────────
    html.Div(className='footer', children=[
        html.P('Data: Ulasan Google Maps Desa Wisata Indonesia | '
               'Model: mdhugol/indonesia-bert-sentiment-classification'),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/detail':
        return layout_detail()
    return layout_insight()


@callback(
    Output('nav-insight', 'className'),
    Output('nav-detail', 'className'),
    Input('url', 'pathname'),
)
def update_nav_active(pathname):
    if pathname == '/detail':
        return 'nav-link', 'nav-link active'
    return 'nav-link active', 'nav-link'


# ── Smart "Semua" toggle ────────────────────────────────────────────────────
@callback(
    Output('desa-dropdown', 'value'),
    Output('prev-desa', 'data'),
    Input('desa-dropdown', 'value'),
    Input('prev-desa', 'data'),
    prevent_initial_call=True,
)
def smart_semua_toggle(current, prev):
    """If user adds 'Semua' to specific selections, keep only 'Semua'.
    If user adds a specific desa while 'Semua' is selected, remove 'Semua'."""
    from dash import no_update, ctx
    if not current:
        return ['Semua'], ['Semua']
    prev = prev or []
    had_semua = 'Semua' in prev
    has_semua = 'Semua' in current
    has_specific = any(v != 'Semua' for v in current)
    if has_semua and has_specific:
        if not had_semua:
            # User just added Semua → keep only Semua
            return ['Semua'], ['Semua']
        else:
            # User added specific while Semua was there → remove Semua
            new_val = [v for v in current if v != 'Semua']
            return new_val, new_val
    return current, current


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 CALLBACKS (Insight)
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output('headline-banner', 'children'),
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

    # ── ROW 1: Headline banner ───────────────────────────────────────────
    headline_pos = ins.get('headline_positive', '')
    headline_neg = ins.get('headline_negative', '')
    banner = [
        html.Div(className='headline-item positif', children=[
            html.Span('✓', className='headline-icon positif'),
            headline_pos,
        ]) if headline_pos else None,
        html.Div(className='headline-item negatif', children=[
            html.Span('⚠', className='headline-icon negatif'),
            headline_neg,
        ]) if headline_neg else None,
    ]

    # ── ROW 2: Verdict left + right ──────────────────────────────────────
    verdict_label, verdict_desc_text, verdict_cls = get_verdict(pct_pos)

    v_left = html.Div(className=f'verdict-left-inner {verdict_cls}', children=[
        html.Div(verdict_label, className='verdict-text'),
        html.Div(verdict_desc_text, className='verdict-desc'),
        html.Div(f'Total {total:,} ulasan dari {label}',
                 className='verdict-total'),
    ])

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
            count_text=f'{p["count"]:,} ulasan positif',
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
            count_text=f'{c["count"]:,} keluhan',
            keywords=kws,
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
        banner,
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
    """Pick up to n representative reviews matching category + sentiment."""
    mask = d['aspects_str'].str.contains(category, na=False) & \
           (d['sentiment'] == sentiment)
    subset = d[mask]
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


def _build_keywords_display(keywords, attributions=None):
    """Render keywords, optionally with custom CSS village attribution tooltips."""
    if not attributions:
        return html.Div(f'Kata kunci: {", ".join(keywords)}',
                        className='insight-item-keywords')
    children = [html.Span('Kata kunci: ')]
    for i, kw in enumerate(keywords):
        villages = attributions.get(kw, [])
        if villages:
            tooltip = ', '.join(v.title() for v in villages)
            children.append(html.Span(
                className='keyword-tooltip-wrapper', children=[
                    html.Span(kw, className='keyword-attributed'),
                    html.Span(f'Desa: {tooltip}',
                              className='keyword-tooltip-content'),
                ]))
        else:
            children.append(html.Span(kw))
        if i < len(keywords) - 1:
            children.append(', ')
    return html.Div(children, className='insight-item-keywords')


def _insight_item(rank, category, count_text, keywords, positivity_rate=None,
                  sample_reviews=None, review_css='positif',
                  keyword_attributions=None):
    """Build one ranked insight item row with expandable reviews."""
    body_children = [
        html.Div(category, className='insight-item-category'),
        html.Div(count_text, className='insight-item-count'),
        _build_keywords_display(keywords, keyword_attributions)
        if keywords else None,
        html.Div(className='insight-item-bar-row', children=[
            html.Div(className='mini-bar-track', children=[
                html.Div(className=f'mini-bar-fill {aspect_health_class(positivity_rate)}',
                         style={'width': f'{positivity_rate}%'}),
            ]),
            html.Span(f'{positivity_rate:.0f}% puas',
                      className='mini-bar-label'),
        ]) if positivity_rate is not None else None,
    ]

    if sample_reviews:
        # First review shown directly
        first = sample_reviews[0]
        body_children.append(
            _review_quote(first[0], review_css,
                          village_name=first[1] if first[1] else None))
        # Additional reviews in expandable section
        if len(sample_reviews) > 1:
            extra = [_review_quote(sr[0], review_css,
                                   village_name=sr[1] if sr[1] else None)
                     for sr in sample_reviews[1:]]
            body_children.append(
                html.Details(className='extra-reviews', children=[
                    html.Summary(f'Lihat {len(extra)} ulasan lainnya',
                                 className='extra-reviews-toggle'),
                    html.Div(extra, className='extra-reviews-list'),
                ]))

    children = [
        html.Div(str(rank), className=f'insight-item-rank rank-{rank}'),
        html.Div(className='insight-item-body', children=body_children),
    ]
    return html.Div(className='insight-item', children=children)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 CALLBACKS (Detail) — unchanged from original
# ══════════════════════════════════════════════════════════════════════════════

# ── Overview: Grouped bar + Pie ──────────────────────────────────────────────
@callback(Output('grouped-bar', 'figure'), Input('desa-dropdown', 'value'))
def update_grouped_bar(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    pivot = df.groupby(['nama desa wisata', 'sentiment']).size().reset_index(name='count')
    pivot['nama desa wisata'] = pivot['nama desa wisata'].str.title()
    selected_names = {d.title() for d in desa_list} if not is_semua else set()

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
    if selected_names:
        for tr in fig.data:
            opacities = [1.0 if name in selected_names else 0.28
                         for name in tr.x]
            tr.marker.opacity = opacities

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
    fig = apply_chart_theme(fig, show_legend=False, height=380,
                            hide_xaxis=True, hide_yaxis=True)
    fig.update_layout(
        margin=dict(t=58, b=28, l=28, r=28),
        annotations=[
            dict(text=f'<b>{total:,}</b>', x=0.5, y=0.56, showarrow=False,
                 font=dict(family=THEME_TITLE_FONT, size=34,
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
        margin=dict(t=58, b=52, l=14, r=40),
    )
    # Total count annotation at end of each stacked bar
    totals = {a: asp[a].get('total', 0) for a in aspect_cats}
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

    rows = []
    for sent in ['Positif', 'Negatif', 'Netral']:
        for word in opinions.get(sent, [])[:7]:
            rows.append({'Kata': word, 'Sentimen': sent})

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text='Belum ada data', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=350,
                                 hide_xaxis=True, hide_yaxis=True)

    op_df = pd.DataFrame(rows)
    word_counts = op_df['Kata'].value_counts().reset_index()
    word_counts.columns = ['Kata', 'Frekuensi']

    word_sent = op_df.groupby('Kata')['Sentimen'].first().to_dict()
    word_counts['Sentimen'] = word_counts['Kata'].map(word_sent)
    word_counts = word_counts.head(10).iloc[::-1]

    fig = px.bar(
        word_counts, y='Kata', x='Frekuensi',
        orientation='h', color='Sentimen', color_discrete_map=COLORS,
        title=f'Opini — {aspect}',
        text='Frekuensi',
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=6),
        textposition='outside',
        textfont=dict(family=THEME_FONT, size=11, color=THEME_TITLE_COLOR),
        hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>',
    )
    fig = apply_chart_theme(fig, show_legend=False, height=350,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(bargap=0.32, margin=dict(t=58, b=30, l=14, r=44))
    return fig


# ── Word Analysis (TF-IDF + Wordcloud toggle) ───────────────────────────────
@callback(
    Output('tfidf-bar',     'style'),
    Output('wordcloud-img', 'style'),
    Input('word-view-toggle', 'value'),
)
def toggle_word_view(view):
    if view == 'bar':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block', 'width': '100%',
            'borderRadius': '8px'}


@callback(
    Output('tfidf-bar', 'figure'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_tfidf_bar(desa, sentiment):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    if single_key:
        tfidf_key = f'{single_key}_{sentiment}'
    else:
        tfidf_key = None
    keywords = tfidf_data.get(tfidf_key, []) if tfidf_key else []
    if not keywords:
        fig = go.Figure()
        fig.add_annotation(text='Pilih satu desa untuk lihat kata kunci TF-IDF'
                                if not tfidf_key else 'Data TF-IDF belum tersedia',
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color=THEME_AXIS_COLOR, size=13))
        return apply_chart_theme(fig, show_legend=False, height=420,
                                 hide_xaxis=True, hide_yaxis=True)

    kw_df = pd.DataFrame(keywords, columns=['Kata Kunci', 'Skor TF-IDF'])
    kw_df = kw_df.iloc[::-1]
    color = COLORS.get(sentiment, '#475569')

    fig = px.bar(
        kw_df, x='Skor TF-IDF', y='Kata Kunci', orientation='h',
        color_discrete_sequence=[color],
        title=f'Kata Kunci Teratas (TF-IDF) — {sentiment}',
        text='Skor TF-IDF',
    )
    fig.update_traces(
        marker=dict(line=dict(width=0), cornerradius=6),
        textposition='outside',
        texttemplate='%{x:.2f}',
        textfont=dict(family=THEME_FONT, size=10, color=THEME_TITLE_COLOR),
        hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>',
    )
    fig = apply_chart_theme(fig, show_legend=False, height=420,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(bargap=0.35, margin=dict(t=58, b=30, l=14, r=56))
    return fig


@callback(
    Output('wordcloud-img', 'src'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_wordcloud(desa, sentiment):
    d = filter_df(desa)
    subset = d[d['sentiment'] == sentiment]['cleaned_review'].dropna().tolist()
    base = COLORS.get(sentiment, '#475569')
    light = COLORS_SOFT.get(sentiment, '#94a3b8')
    return make_wordcloud(subset, base, light)


@callback(
    Output('bigram-chart', 'figure'),
    Input('desa-dropdown',  'value'),
    Input('sentiment-tabs', 'value'),
)
def update_bigram(desa, sentiment):
    d = filter_df(desa)
    subset = d[d['sentiment'] == sentiment]['cleaned_review'].dropna().tolist()
    bg_df = top_bigrams(subset, n=10)
    color = COLORS.get(sentiment, '#475569')
    if bg_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Tidak ada data', x=0.5, y=0.5,
                           showarrow=False, font=dict(color=THEME_AXIS_COLOR))
        return apply_chart_theme(fig, show_legend=False, height=420,
                                 hide_xaxis=True, hide_yaxis=True)
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
    fig = apply_chart_theme(fig, show_legend=False, height=420,
                            show_yaxis_grid=False, hide_xaxis=True)
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        bargap=0.35,
        margin=dict(t=58, b=30, l=14, r=44),
    )
    return fig


# ── Comparison: Radar + Heatmap (conditional visibility) ─────────────────────
@callback(
    Output('comparison-section', 'style'),
    Input('desa-dropdown', 'value'),
)
def toggle_comparison(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    if is_semua or len(desa_list) > 1:
        return {'display': 'block'}
    return {'display': 'none'}


@callback(Output('radar-chart', 'figure'), Input('desa-dropdown', 'value'))
def update_radar(desa):
    desa_list, single_key, is_semua = _normalize_desa(desa)
    selected = set() if is_semua else {d for d in desa_list}
    aspects_for_radar = [a for a in ASPECT_ORDER if a != 'Lainnya']

    fig = go.Figure()
    palette = ['#0c1e36', '#f59e0b', '#10b981', '#ef4444', '#6366f1',
               '#0ea5e9', '#d946ef', '#14b8a6', '#f97316', '#84cc16']
    for idx, d_name in enumerate(DESA_LIST):
        asp = aspect_data.get(d_name, {})
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
        is_highlighted = (not selected) or (d_name in selected)
        opacity = 0.85 if is_highlighted else 0.18
        fill_alpha = 0.27 if is_highlighted else 0.08

        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels,
            fill='toself', name=d_name.title(),
            fillcolor=f'rgba({r},{g},{b},{fill_alpha})',
            line=dict(color=color, width=2.5 if is_highlighted else 1),
            opacity=opacity,
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
    aspects_for_hm = [a for a in ASPECT_ORDER if a != 'Lainnya']

    matrix = []
    for d_name in DESA_LIST:
        row = []
        asp = aspect_data.get(d_name, {})
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
        y=[d.title() for d in DESA_LIST],
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
    if sentiment_filter != 'Semua':
        d = d[d['sentiment'] == sentiment_filter]
    if aspect_filter != 'Semua':
        d = d[d['aspects_list'].apply(lambda x: aspect_filter in x)]
    cols = ['nama desa wisata', 'sentiment', 'aspects_str', 'cleaned_review']
    return d[cols].dropna(subset=['cleaned_review']).head(500).to_dict('records')


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050)
