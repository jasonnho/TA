---
title: Analisis Sentimen Desa Wisata
emoji: 🏞️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Dashboard Analisis Sentimen Berbasis Aspek — Desa Wisata

Dashboard interaktif hasil **Aspect-Based Sentiment Analysis (ABSA)** terhadap
ulasan Google Maps sejumlah desa wisata di Indonesia. Model: IndoBERT
multi-task / single-task + CRF.

Dibangun dengan [Dash](https://dash.plotly.com/). Data yang ditampilkan adalah
hasil pelabelan otomatis yang sudah diproses sebelumnya (CSV/JSON), jadi
dashboard ini tidak menjalankan model saat runtime.
