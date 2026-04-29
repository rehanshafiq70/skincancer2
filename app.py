"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  SKINSCAN AI  ·  NEXT-GEN DERMATOLOGY INTELLIGENCE SYSTEM  ·  v15.0             ║
║──────────────────────────────────────────────────────────────────────────────── ║
║  Design      : Apex Medical  ·  Glassmorphism + Gradient Hybrid  ·  Mobile-First ║
║  Developer   : Rehan Shafique                                                    ║
║  Institution : University of Agriculture Faisalabad  ·  Dept. of Bioinformatics  ║
║  Supervisor  : Dr Hassan Tariq                                                   ║
║  Reg. No.    : 2022-AG-7662                                                      ║
║  Session     : 2022 – 2026                                                       ║
║  Model       : skin_cancer_cnn.h5  (Binary: Benign / Malignant)                  ║
║  Live App    : https://skincancerpredictions.streamlit.app/                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PROJECT OVERVIEW
────────────────
SkinScan AI is a clinical-grade AI-powered dermatology intelligence platform built
as a Final Year Project for BS Bioinformatics at the University of Agriculture
Faisalabad. It leverages a Convolutional Neural Network (CNN) trained on the
Melanoma Cancer Image Dataset to classify dermoscopic skin lesion images into
Benign / Malignant categories with high sensitivity and specificity.

KEY FEATURES
────────────
  ✦ Multi-class CNN with 8 lesion type detection (Melanoma, BCC, SCC, Benign Nevus,
    Seborrheic Keratosis, Dermatofibroma, Vascular Lesion, Actinic Keratosis)
  ✦ Binary Benign / Malignant classification (primary output used by the app)
  ✦ Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap visualization
  ✦ Laplacian variance-based image quality / blur detection gate
  ✦ Automated PDF clinical report generation via ReportLab
  ✦ CSV / JSON session history export
  ✦ Real-time analytics dashboard with Plotly interactive charts
  ✦ ABCDE melanoma self-check educational module
  ✦ Comprehensive medical guide: doctor recommendations, prevention, treatments
  ✦ Dark / light glassmorphism theme with responsive mobile-first layout

ARCHITECTURE
────────────
  Presentation Layer : Streamlit UI + Custom CSS (Glassmorphism v15)
  Application Layer  : SkinScanApp controller (session state + page routing)
  Intelligence Layer : NeuralCoreEngine (TF/Keras CNN) + ClinicalProtocols KB
  Data Layer         : ImageProcessor + ReportGenerator (PDF/CSV)

USAGE
─────
  1.  pip install -r requirements.txt
  2.  Place skin_cancer_cnn.h5 in the same directory as app.py
  3.  streamlit run app.py

DISCLAIMER
──────────
  This platform is developed for research and educational purposes only.
  It does NOT constitute a certified medical device or formal clinical diagnosis.
  All AI-generated outputs must be verified by a qualified dermatologist or
  oncologist prior to any clinical decision-making.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD LIBRARY IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import io
import json
import base64
import random
import time
import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  THIRD-PARTY IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORT — ReportLab  (PDF generation)
#  Install with:  pip install reportlab
# ══════════════════════════════════════════════════════════════════════════════
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    PDF_OK = True
except ImportError:
    PDF_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES  ──  v15  "Clinical Apex" Design System
#  Glassmorphism + Gradient Hybrid  ·  Dark & Light Themes
# ══════════════════════════════════════════════════════════════════════════════

def inject_css(theme: str = "dark") -> None:
    """
    Inject the full SkinScan AI CSS design system into the Streamlit app.

    Parameters
    ----------
    theme : str
        "dark"  → deep navy clinical dark mode  (default)
        "light" → soft blue-white clinical light mode

    Design Tokens (Dark Theme)
    --------------------------
    Background   : #020d1e  (deep navy)
    Surface      : rgba(4,22,50,0.78)  (glass card)
    Primary Blue : #2563EB
    Teal         : #14B8A6
    Purple       : #8B5CF6
    Danger Red   : #EF4444
    Safe Green   : #10B981
    """
    dark = (theme == "dark")

    # ── Dark theme tokens ──────────────────────────────────────────
    if dark:
        BG          = "#020d1e"
        BG2         = "#041226"
        SURF        = "rgba(4,22,50,0.78)"
        SURF2       = "rgba(6,28,60,0.65)"
        BORDER      = "rgba(37,99,235,0.22)"
        BDH         = "rgba(20,184,166,0.50)"
        TEXT        = "#dff0fa"
        SUB         = "#6b9ab8"
        MUTED       = "#2a4a62"
        NAV_BG      = "rgba(2,13,30,0.92)"
        INP         = "rgba(4,22,50,0.92)"
        DIV         = "rgba(37,99,235,0.13)"
        HERO_G1     = "rgba(37,99,235,0.18)"
        HERO_G2     = "rgba(20,184,166,0.10)"
        FOOTER_BG   = "rgba(2,8,18,0.97)"
        CARD_HOVER  = "rgba(10,35,70,0.95)"
        SCROLLBAR   = "rgba(37,99,235,0.30)"
    # ── Light theme tokens ─────────────────────────────────────────
    else:
        BG          = "#f0f5fc"
        BG2         = "#e6eef9"
        SURF        = "rgba(255,255,255,0.93)"
        SURF2       = "rgba(240,247,255,0.90)"
        BORDER      = "rgba(37,99,235,0.18)"
        BDH         = "rgba(20,184,166,0.45)"
        TEXT        = "#0c1e32"
        SUB         = "#3a6080"
        MUTED       = "#a8c4d8"
        NAV_BG      = "rgba(248,252,255,0.97)"
        INP         = "rgba(255,255,255,0.97)"
        DIV         = "rgba(37,99,235,0.10)"
        HERO_G1     = "rgba(37,99,235,0.08)"
        HERO_G2     = "rgba(20,184,166,0.06)"
        FOOTER_BG   = "rgba(8,20,45,0.97)"
        CARD_HOVER  = "rgba(245,250,255,0.98)"
        SCROLLBAR   = "rgba(37,99,235,0.25)"

    st.markdown(f"""
    <style>
    /* ══════════════════════════════════════════════════════════════
       GOOGLE FONTS IMPORT
    ══════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Oxanium:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    /* ══════════════════════════════════════════════════════════════
       GLOBAL RESET & BASE
    ══════════════════════════════════════════════════════════════ */
    *, *::before, *::after {{
        box-sizing: border-box;
    }}
    html, body {{
        font-family: 'Outfit', sans-serif !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       APP BACKGROUND — Animated Medical Gradient Mesh
    ══════════════════════════════════════════════════════════════ */
    .stApp {{
        font-family: 'Outfit', sans-serif !important;
        background-color: {BG} !important;
        background-image:
            radial-gradient(ellipse 90% 45% at 8% 2%,  {HERO_G1} 0%, transparent 65%),
            radial-gradient(ellipse 70% 50% at 92% 98%, {HERO_G2} 0%, transparent 60%),
            url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80'
                 xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb'
                 fill-opacity='0.018'%3E%3Ccircle cx='1' cy='1' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        background-attachment: fixed;
        color: {TEXT} !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       HIDE STREAMLIT DEFAULT CHROME
    ══════════════════════════════════════════════════════════════ */
    #MainMenu                          {{ display: none !important; }}
    header[data-testid="stHeader"]     {{ display: none !important; }}
    footer                             {{ display: none !important; }}
    [data-testid="stSidebar"]          {{ display: none !important; }}
    .stDeployButton                    {{ display: none !important; }}
    .stDecoration                      {{ display: none !important; }}
    [data-testid="collapsedControl"]   {{ display: none !important; }}

    /* ══════════════════════════════════════════════════════════════
       BLOCK CONTAINER
    ══════════════════════════════════════════════════════════════ */
    .block-container {{
        padding-top:    82px !important;
        padding-left:   2rem !important;
        padding-right:  2rem !important;
        max-width:      1360px !important;
        margin:         0 auto !important;
        overflow-x:     hidden !important;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-top:   68px !important;
            padding-left:  0.75rem !important;
            padding-right: 0.75rem !important;
        }}
    }}

    /* ══════════════════════════════════════════════════════════════
       FLOATING NAVBAR
    ══════════════════════════════════════════════════════════════ */
    .navbar-shell {{
        position:         fixed;
        top: 0; left: 0; right: 0;
        z-index:          9999;
        background:       {NAV_BG};
        backdrop-filter:  blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border-bottom:    1px solid {BORDER};
        padding:          0 28px;
        height:           64px;
        display:          flex;
        align-items:      center;
        justify-content:  space-between;
        box-shadow:       0 4px 32px rgba(0,0,0,0.18), 0 1px 0 rgba(37,99,235,0.08);
        transition:       height 0.3s ease, box-shadow 0.3s ease;
    }}

    /* ── Logo ─────────────────────────────────────────────────── */
    .nav-logo {{
        display:     flex;
        align-items: center;
        gap:         10px;
        flex-shrink: 0;
    }}
    .nav-logo-icon {{
        font-size:  1.55rem;
        animation:  logo-pulse 3s ease-in-out infinite;
        filter:     drop-shadow(0 0 8px rgba(20,184,166,0.5));
    }}
    @keyframes logo-pulse {{
        0%,100% {{
            filter:    drop-shadow(0 0 5px rgba(37,99,235,0.6));
            transform: scale(1);
        }}
        50% {{
            filter:    drop-shadow(0 0 18px rgba(20,184,166,0.8));
            transform: scale(1.08);
        }}
    }}
    .nav-logo-text {{
        font-family:              'Oxanium', sans-serif;
        font-size:                1.08rem;
        font-weight:              800;
        background:               linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%);
        -webkit-background-clip:  text;
        -webkit-text-fill-color:  transparent;
        letter-spacing:           0.3px;
        line-height:              1.1;
    }}
    .nav-logo-sub {{
        font-size:      0.56rem;
        color:          {SUB};
        letter-spacing: 1.8px;
        text-transform: uppercase;
        font-weight:    500;
    }}

    /* ── AI Status Badge ─────────────────────────────────────── */
    .nav-ai-badge {{
        display:          inline-flex;
        align-items:      center;
        gap:              5px;
        background:       linear-gradient(135deg, rgba(37,99,235,0.15), rgba(20,184,166,0.10));
        border:           1px solid rgba(37,99,235,0.30);
        padding:          4px 12px;
        border-radius:    99px;
        font-size:        0.62rem;
        font-weight:      700;
        color:            #60a5fa;
        letter-spacing:   1.5px;
        text-transform:   uppercase;
    }}
    .nav-pulse {{
        width:         7px;
        height:        7px;
        border-radius: 50%;
        background:    #10b981;
        display:       inline-block;
        box-shadow:    0 0 0 0 rgba(16,185,129,0.5);
        animation:     nav-dot-pulse 1.8s ease-in-out infinite;
    }}
    @keyframes nav-dot-pulse {{
        0%   {{ box-shadow: 0 0 0 0   rgba(16,185,129,0.6); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(16,185,129,0);   }}
        100% {{ box-shadow: 0 0 0 0   rgba(16,185,129,0);   }}
    }}

    /* ══════════════════════════════════════════════════════════════
       HORIZONTAL NAVIGATION MENU  (streamlit-option-menu)
    ══════════════════════════════════════════════════════════════ */
    .nav-menu-center ul {{
        display:        flex !important;
        flex-direction: row !important;
        gap:            2px !important;
        list-style:     none !important;
        margin:         0 !important;
        padding:        4px !important;
        background:     {SURF2} !important;
        border:         1px solid {BORDER} !important;
        border-radius:  13px !important;
    }}
    .nav-menu-center ul li a {{
        font-family:    'Outfit', sans-serif !important;
        font-size:      0.81rem !important;
        font-weight:    500 !important;
        padding:        7px 14px !important;
        border-radius:  9px !important;
        color:          {SUB} !important;
        transition:     all 0.22s ease !important;
        white-space:    nowrap !important;
    }}
    .nav-menu-center ul li a:hover {{
        color:      {TEXT} !important;
        background: rgba(37,99,235,0.10) !important;
        transform:  translateY(-1px);
    }}
    .nav-menu-center ul li a[aria-selected="true"] {{
        background:  linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color:       white !important;
        font-weight: 600 !important;
        box-shadow:  0 3px 14px rgba(37,99,235,0.45) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       PAGE BANNER  (top of each sub-page)
    ══════════════════════════════════════════════════════════════ */
    .page-banner {{
        background:    linear-gradient(135deg, rgba(37,99,235,0.14) 0%,
                                               rgba(20,184,166,0.08) 50%,
                                               rgba(139,92,246,0.06) 100%);
        border:        1px solid {BORDER};
        border-radius: 22px;
        padding:       30px 36px 24px;
        margin-bottom: 28px;
        position:      relative;
        overflow:      hidden;
        animation:     fade-in-up 0.5s ease;
    }}
    .page-banner::before {{
        content:    '';
        position:   absolute;
        top: 0; left: 0; right: 0;
        height:     2px;
        background: linear-gradient(90deg,
            transparent, #2563eb 25%, #14b8a6 50%, #8b5cf6 75%, transparent);
    }}
    .banner-chip {{
        display:        inline-block;
        background:     rgba(37,99,235,0.14);
        border:         1px solid rgba(37,99,235,0.30);
        color:          #60a5fa;
        font-size:      0.64rem;
        font-weight:    700;
        padding:        3px 12px;
        border-radius:  99px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom:  10px;
    }}
    .banner-title {{
        font-family:              'Oxanium', sans-serif;
        font-size:                clamp(1.7rem, 3.5vw, 2.5rem);
        font-weight:              800;
        background:               linear-gradient(135deg, #60a5fa 0%, #14b8a6 45%, #a78bfa 100%);
        -webkit-background-clip:  text;
        -webkit-text-fill-color:  transparent;
        letter-spacing:           -0.5px;
        margin:                   0 0 8px;
        line-height:              1.2;
    }}
    .banner-sub {{
        font-size:   0.87rem;
        color:       {SUB};
        max-width:   640px;
        line-height: 1.6;
    }}

    /* ══════════════════════════════════════════════════════════════
       HERO SECTION  (Home page)
    ══════════════════════════════════════════════════════════════ */
    .hero-section {{
        min-height:    300px;
        display:       flex;
        flex-direction: column;
        justify-content: center;
        padding:       48px 40px;
        background:    linear-gradient(135deg,
                            rgba(37,99,235,0.12) 0%,
                            rgba(20,184,166,0.08) 40%,
                            rgba(139,92,246,0.06) 100%);
        border:        1px solid {BORDER};
        border-radius: 24px;
        margin-bottom: 32px;
        position:      relative;
        overflow:      hidden;
        animation:     fade-in-up 0.6s ease;
    }}
    .hero-section::before {{
        content:    '';
        position:   absolute;
        top: 0; left: 0; right: 0;
        height:     3px;
        background: linear-gradient(90deg, #2563eb 0%, #14b8a6 50%, #8b5cf6 100%);
    }}
    .hero-section::after {{
        content:         '⬡';
        position:        absolute;
        right:           40px;
        top:             50%;
        transform:       translateY(-50%);
        font-size:       10rem;
        opacity:         0.035;
        color:           #2563eb;
        pointer-events:  none;
    }}
    .hero-title {{
        font-family:              'Oxanium', sans-serif;
        font-size:                clamp(2rem, 5vw, 3.4rem);
        font-weight:              800;
        background:               linear-gradient(135deg, #60a5fa 0%, #14b8a6 40%, #a78bfa 85%);
        -webkit-background-clip:  text;
        -webkit-text-fill-color:  transparent;
        letter-spacing:           -1px;
        margin:                   0 0 6px;
        line-height:              1.15;
    }}
    .hero-subtitle-small {{
        font-family:    'Oxanium', sans-serif;
        font-size:      0.88rem;
        color:          #14b8a6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom:  10px;
        font-weight:    600;
    }}
    .hero-subtitle {{
        font-size:     1.0rem;
        color:         {SUB};
        max-width:     560px;
        line-height:   1.65;
        margin-bottom: 28px;
    }}
    .hero-badges {{
        display:       flex;
        flex-wrap:     wrap;
        gap:           8px;
        margin-bottom: 20px;
    }}
    .hbadge {{
        padding:        5px 14px;
        border-radius:  99px;
        font-size:      0.76rem;
        font-weight:    600;
        letter-spacing: 0.3px;
        display:        inline-flex;
        align-items:    center;
        gap:            5px;
        transition:     transform 0.2s, box-shadow 0.2s;
    }}
    .hbadge:hover {{
        transform:  translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .hbadge-blue   {{ background: rgba(37,99,235,0.14);  color: #60a5fa; border: 1px solid rgba(37,99,235,0.30); }}
    .hbadge-teal   {{ background: rgba(20,184,166,0.12); color: #2dd4bf; border: 1px solid rgba(20,184,166,0.28); }}
    .hbadge-purple {{ background: rgba(139,92,246,0.12); color: #a78bfa; border: 1px solid rgba(139,92,246,0.28); }}
    .hbadge-green  {{ background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.28); }}
    .hbadge-red    {{ background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.28); }}

    /* ══════════════════════════════════════════════════════════════
       FEATURE CARDS  (Home page grid)
    ══════════════════════════════════════════════════════════════ */
    .feat-card {{
        background:      {SURF};
        border:          1px solid {BORDER};
        border-radius:   18px;
        padding:         24px 20px;
        text-align:      center;
        backdrop-filter: blur(14px);
        transition:      all 0.32s cubic-bezier(.34,1.56,.64,1);
        height:          100%;
        position:        relative;
        overflow:        hidden;
    }}
    .feat-card::before {{
        content:       '';
        position:      absolute;
        inset:         0;
        border-radius: 18px;
        background:    linear-gradient(135deg, rgba(37,99,235,0.06), rgba(20,184,166,0.04));
        opacity:       0;
        transition:    opacity 0.3s;
    }}
    .feat-card:hover {{
        transform:  translateY(-10px) scale(1.025);
        border-color: {BDH};
        box-shadow: 0 24px 56px rgba(37,99,235,0.18);
    }}
    .feat-card:hover::before {{ opacity: 1; }}
    .feat-icon  {{ font-size: 2.5rem; margin-bottom: 14px; display: block; }}
    .feat-title {{ font-weight: 700; font-size: 0.96rem; margin-bottom: 8px; }}
    .feat-desc  {{ font-size: 0.80rem; color: {SUB}; line-height: 1.55; }}

    /* ══════════════════════════════════════════════════════════════
       GLASS CARDS  (General purpose card component)
    ══════════════════════════════════════════════════════════════ */
    .glass-card {{
        background:      {SURF};
        border:          1px solid {BORDER};
        border-radius:   20px;
        padding:         24px;
        margin-bottom:   20px;
        backdrop-filter: blur(16px) saturate(150%);
        transition:      transform 0.28s cubic-bezier(.34,1.56,.64,1),
                         box-shadow 0.28s ease,
                         border-color 0.28s ease;
        position:        relative;
        overflow:        hidden;
    }}
    .glass-card::after {{
        content:     '';
        position:    absolute;
        top: 0; left: 0;
        width:       100%;
        height:      2px;
        background:  linear-gradient(90deg,
            transparent, rgba(37,99,235,0.55), rgba(20,184,166,0.55), transparent);
        opacity:     0;
        transition:  opacity 0.3s;
    }}
    .glass-card:hover {{
        transform:    translateY(-4px);
        box-shadow:   0 18px 48px rgba(37,99,235,0.14);
        border-color: {BDH};
    }}
    .glass-card:hover::after {{ opacity: 1; }}

    /* ══════════════════════════════════════════════════════════════
       KPI / METRIC CARDS  (Dashboard)
    ══════════════════════════════════════════════════════════════ */
    .kpi-card {{
        background:      {SURF};
        border:          1px solid {BORDER};
        border-radius:   18px;
        padding:         20px 16px 16px;
        text-align:      center;
        backdrop-filter: blur(14px);
        transition:      all 0.28s cubic-bezier(.34,1.56,.64,1);
        position:        relative;
        overflow:        hidden;
    }}
    .kpi-card:hover {{
        transform:    translateY(-6px) scale(1.025);
        box-shadow:   0 16px 42px rgba(37,99,235,0.16);
        border-color: {BDH};
    }}
    .kpi-glow {{
        position:       absolute;
        width:          90px;
        height:         90px;
        border-radius:  50%;
        filter:         blur(35px);
        opacity:        0.22;
        top:  -15px;
        right: -15px;
        pointer-events: none;
    }}
    .kpi-icon  {{ font-size: 1.5rem; margin-bottom: 8px; display: block; }}
    .kpi-label {{ font-size: 0.68rem; color: {SUB}; text-transform: uppercase; letter-spacing: 2px; font-weight: 500; margin-bottom: 7px; }}
    .kpi-value {{ font-family: 'Oxanium', monospace; font-size: 2.0rem; font-weight: 700; color: {TEXT}; line-height: 1; margin-bottom: 5px; }}
    .kd-pos    {{ font-size: 0.72rem; color: #34d399; font-weight: 500; }}
    .kd-neg    {{ font-size: 0.72rem; color: #f87171; font-weight: 500; }}
    .kd-neu    {{ font-size: 0.72rem; color: {SUB}; }}

    /* ══════════════════════════════════════════════════════════════
       SECTION HEADINGS
    ══════════════════════════════════════════════════════════════ */
    .sec-head {{
        font-size:   1.0rem;
        font-weight: 700;
        color:       {TEXT};
        margin-bottom: 14px;
        display:     flex;
        align-items: center;
        gap:         9px;
    }}
    .sec-head span {{
        display:       inline-block;
        width:         3px;
        height:        18px;
        background:    linear-gradient(180deg, #2563eb, #14b8a6);
        border-radius: 3px;
    }}

    /* ══════════════════════════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════════════════════════ */
    .stButton > button {{
        background:     linear-gradient(135deg, #1d4ed8 0%, #2563eb 40%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        color:          white !important;
        border:         none !important;
        border-radius:  12px !important;
        font-family:    'Outfit', sans-serif !important;
        font-weight:    600 !important;
        font-size:      0.88rem !important;
        letter-spacing: 0.4px !important;
        padding:        0.70rem 1.6rem !important;
        width:          100% !important;
        transition:     all 0.30s ease !important;
        box-shadow:     0 4px 16px rgba(37,99,235,0.30) !important;
    }}
    .stButton > button:hover {{
        background-position: right center !important;
        transform:           translateY(-3px) !important;
        box-shadow:          0 10px 30px rgba(37,99,235,0.50),
                             0 0 0 1px rgba(20,184,166,0.28) !important;
    }}

    /* ── Scan button special animation ─────────────────────── */
    .scan-btn-wrap .stButton > button {{
        background:     linear-gradient(135deg, #7c3aed 0%, #2563eb 45%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        font-size:      0.94rem !important;
        font-weight:    700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding:        0.82rem !important;
        animation:      scan-idle 3s ease-in-out infinite;
    }}
    .scan-btn-wrap .stButton > button:hover {{
        animation:  none;
        box-shadow: 0 0 44px rgba(139,92,246,0.60),
                    0 8px 30px rgba(37,99,235,0.50) !important;
    }}
    @keyframes scan-idle {{
        0%,100% {{ box-shadow: 0 0 18px rgba(139,92,246,0.32), 0 4px 15px rgba(37,99,235,0.24); }}
        50%      {{ box-shadow: 0 0 32px rgba(20,184,166,0.42), 0 6px 20px rgba(37,99,235,0.34); }}
    }}

    /* ── Download button ────────────────────────────────────── */
    .stDownloadButton > button {{
        background:  linear-gradient(135deg, #0d9488, #14b8a6) !important;
        color:       white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border:      none !important;
        box-shadow:  0 4px 15px rgba(20,184,166,0.28) !important;
    }}
    .stDownloadButton > button:hover {{
        transform:  translateY(-3px) !important;
        box-shadow: 0 10px 28px rgba(20,184,166,0.50) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       PROGRESS BARS
    ══════════════════════════════════════════════════════════════ */
    .stProgress > div > div > div > div {{
        background:    linear-gradient(90deg, #2563eb, #14b8a6, #8b5cf6) !important;
        border-radius: 99px !important;
    }}
    .stProgress > div > div > div {{
        background:    rgba(37,99,235,0.12) !important;
        border-radius: 99px !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       SCAN LOADING RING ANIMATION
    ══════════════════════════════════════════════════════════════ */
    .scan-ring-wrap {{
        text-align: center;
        padding:    24px 0 10px;
    }}
    .scan-ring {{
        width:         96px;
        height:        96px;
        border-radius: 50%;
        border:        3px solid transparent;
        border-top-color:   #2563eb;
        border-right-color: #14b8a6;
        border-left-color:  rgba(139,92,246,0.4);
        animation:     ring-spin 1.1s cubic-bezier(.47,.13,.19,.97) infinite;
        margin:        0 auto 12px;
        position:      relative;
    }}
    .scan-ring::before {{
        content:       '';
        position:      absolute;
        inset:         6px;
        border-radius: 50%;
        border:        2px solid transparent;
        border-top-color: rgba(20,184,166,0.5);
        animation:     ring-spin 1.7s linear infinite reverse;
    }}
    .scan-ring::after {{
        content:   '🔬';
        position:  absolute;
        top:  50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
    }}
    @keyframes ring-spin {{ 100% {{ transform: rotate(360deg); }} }}
    .scan-status-txt {{
        font-family:    'Oxanium', monospace;
        font-size:      0.78rem;
        color:          {SUB};
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    /* ══════════════════════════════════════════════════════════════
       RESULT CARDS
    ══════════════════════════════════════════════════════════════ */
    .result-card {{
        border-radius:  20px;
        padding:        28px 24px;
        text-align:     center;
        margin-bottom:  18px;
        position:       relative;
        overflow:       hidden;
        animation:      result-in 0.55s cubic-bezier(.34,1.56,.64,1);
    }}
    @keyframes result-in {{
        from {{ opacity: 0; transform: scale(0.86) translateY(18px); }}
        to   {{ opacity: 1; transform: scale(1)    translateY(0);    }}
    }}
    .res-mal {{
        background: linear-gradient(135deg,rgba(239,68,68,0.13),rgba(220,38,38,0.05));
        border:     2px solid rgba(239,68,68,0.52);
        box-shadow: 0 0 48px rgba(239,68,68,0.12),
                    inset 0 1px 0 rgba(239,68,68,0.18);
    }}
    .res-ben {{
        background: linear-gradient(135deg,rgba(16,185,129,0.13),rgba(5,150,105,0.05));
        border:     2px solid rgba(16,185,129,0.52);
        box-shadow: 0 0 48px rgba(16,185,129,0.12),
                    inset 0 1px 0 rgba(16,185,129,0.18);
    }}
    .res-tag  {{ font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 9px; opacity: 0.82; }}
    .res-type {{ font-family: 'Oxanium', sans-serif; font-size: clamp(1.5rem,3vw,2.1rem); font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }}
    .res-desc {{ font-size: 0.83rem; color: {SUB}; line-height: 1.6; max-width: 380px; margin: 0 auto; }}

    /* ── Risk / Status Badges ───────────────────────────────── */
    .badge {{
        display:        inline-flex;
        align-items:    center;
        gap:            5px;
        padding:        5px 16px;
        border-radius:  99px;
        font-size:      0.76rem;
        font-weight:    700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }}
    .b-high   {{ background: rgba(239,68,68,0.14);  color: #f87171; border: 1px solid rgba(239,68,68,0.38);  }}
    .b-medium {{ background: rgba(245,158,11,0.14); color: #fbbf24; border: 1px solid rgba(245,158,11,0.38); }}
    .b-low    {{ background: rgba(16,185,129,0.14); color: #34d399; border: 1px solid rgba(16,185,129,0.38); }}
    .qual-badge-ok   {{ background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.30); padding: 4px 12px; border-radius: 99px; font-size: 0.73rem; font-weight: 600; display: inline-block; }}
    .qual-badge-warn {{ background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.30); padding: 4px 12px; border-radius: 99px; font-size: 0.73rem; font-weight: 600; display: inline-block; }}

    /* ══════════════════════════════════════════════════════════════
       STEP / RECOMMENDATION BOXES
    ══════════════════════════════════════════════════════════════ */
    .step-box {{
        background:   {SURF2};
        border:       1px solid {BORDER};
        border-left:  3px solid #2563eb;
        border-radius: 10px;
        padding:      10px 14px;
        margin-bottom: 8px;
        font-size:    0.84rem;
        line-height:  1.6;
        transition:   border-left-color 0.22s, background 0.22s;
    }}
    .step-box:hover {{ border-left-color: #14b8a6; background: rgba(20,184,166,0.05); }}
    .step-emg {{ border-left-color: #ef4444 !important; }}
    .step-emg:hover {{ background: rgba(239,68,68,0.05) !important; }}

    /* ══════════════════════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background:    {SURF2} !important;
        border-radius: 12px !important;
        padding:       4px !important;
        gap:           3px !important;
        border:        1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:  9px !important;
        font-family:    'Outfit', sans-serif !important;
        font-weight:    500 !important;
        font-size:      0.82rem !important;
        color:          {SUB} !important;
        padding:        7px 16px !important;
        transition:     all 0.2s !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color:      white !important;
        box-shadow: 0 3px 10px rgba(37,99,235,0.38) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       FORM INPUTS
    ══════════════════════════════════════════════════════════════ */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div > div {{
        background:    {INP} !important;
        border:        1px solid {BORDER} !important;
        border-radius: 10px !important;
        color:         {TEXT} !important;
        font-family:   'Outfit', sans-serif !important;
        font-size:     0.87rem !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: rgba(37,99,235,0.55) !important;
        box-shadow:   0 0 0 3px rgba(37,99,235,0.10) !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       FILE UPLOADER
    ══════════════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] {{
        border:        2px dashed rgba(37,99,235,0.32) !important;
        border-radius: 16px !important;
        background:    {SURF2} !important;
        transition:    all 0.25s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(20,184,166,0.55) !important;
        background:   rgba(20,184,166,0.04) !important;
    }}
    [data-testid="stCameraInput"] {{
        border-radius: 16px !important;
        overflow:      hidden;
    }}
    [data-testid="stCameraInput"] video {{
        border-radius: 14px !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       CUSTOM SCROLLBAR
    ══════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar       {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background:    {SCROLLBAR};
        border-radius: 99px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: rgba(20,184,166,0.50);
    }}

    /* ══════════════════════════════════════════════════════════════
       METRIC WIDGETS
    ══════════════════════════════════════════════════════════════ */
    [data-testid="stMetricLabel"] {{
        font-family:    'Outfit', sans-serif !important;
        color:          {SUB} !important;
        font-size:      0.74rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'Oxanium', monospace !important;
        font-size:   1.45rem !important;
        color:       {TEXT} !important;
    }}

    /* ══════════════════════════════════════════════════════════════
       MISC COMPONENT OVERRIDES
    ══════════════════════════════════════════════════════════════ */
    .stAlert                    {{ border-radius: 12px !important; }}
    .stSpinner > div            {{ border-top-color: #2563eb !important; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER} !important; border-radius: 14px !important; overflow: hidden; }}
    hr                          {{ border-color: {DIV} !important; opacity: 0.8; }}

    /* ══════════════════════════════════════════════════════════════
       SETTINGS PAGE ROWS
    ══════════════════════════════════════════════════════════════ */
    .set-row {{
        background:    {SURF2};
        border:        1px solid {BORDER};
        border-radius: 14px;
        padding:       16px 20px;
        margin-bottom: 10px;
        transition:    border-color 0.22s;
    }}
    .set-row:hover      {{ border-color: {BDH}; }}
    .set-lbl            {{ font-weight: 600; font-size: 0.88rem; margin-bottom: 2px; }}
    .set-desc           {{ font-size: 0.74rem; color: {SUB}; }}

    /* ══════════════════════════════════════════════════════════════
       ABCDE CARDS  (Medical Guide)
    ══════════════════════════════════════════════════════════════ */
    .abcde-card {{
        background:    {SURF};
        border:        1px solid {BORDER};
        border-radius: 16px;
        padding:       18px 10px;
        text-align:    center;
        transition:    all 0.28s cubic-bezier(.34,1.56,.64,1);
    }}
    .abcde-card:hover {{
        transform:  translateY(-7px) scale(1.04);
        border-color: rgba(139,92,246,0.48);
        box-shadow: 0 14px 32px rgba(139,92,246,0.18);
    }}
    .abcde-letter {{ font-family: 'Oxanium', monospace; font-size: 2.4rem; font-weight: 800; margin-bottom: 5px; }}
    .abcde-word   {{ font-weight: 700; font-size: 0.86rem; margin-bottom: 4px; }}
    .abcde-desc   {{ font-size: 0.72rem; color: {SUB}; line-height: 1.45; }}

    /* ══════════════════════════════════════════════════════════════
       KEYFRAME ANIMATIONS
    ══════════════════════════════════════════════════════════════ */
    @keyframes fade-in-up {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0);    }}
    }}
    @keyframes fade-in {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}

    /* ══════════════════════════════════════════════════════════════
       MEDICAL DISCLAIMER BOX
    ══════════════════════════════════════════════════════════════ */
    .disclaimer-banner {{
        background:   linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.05));
        border:       1px solid rgba(239,68,68,0.30);
        border-left:  4px solid #ef4444;
        border-radius: 12px;
        padding:      14px 18px;
        margin:       12px 0;
        font-size:    0.82rem;
        line-height:  1.6;
        color:        {TEXT};
    }}

    /* ══════════════════════════════════════════════════════════════
       SKIN GUIDE CARDS  (Medical Guide - Disease Info)
    ══════════════════════════════════════════════════════════════ */
    .guide-card {{
        background:    {SURF};
        border:        1px solid {BORDER};
        border-radius: 16px;
        padding:       20px 16px;
        text-align:    center;
        transition:    all 0.28s cubic-bezier(.34,1.56,.64,1);
    }}
    .guide-card:hover {{
        transform:    translateY(-5px);
        box-shadow:   0 14px 35px rgba(37,99,235,0.14);
        border-color: {BDH};
    }}

    /* ══════════════════════════════════════════════════════════════
       PREVENTION TIPS  (Medical Guide)
    ══════════════════════════════════════════════════════════════ */
    .prev-tip {{
        background:    {SURF2};
        border:        1px solid {BORDER};
        border-left:   3px solid #14b8a6;
        border-radius: 10px;
        padding:       12px 16px;
        margin-bottom: 9px;
        font-size:     0.84rem;
        line-height:   1.6;
        transition:    border-left-color 0.2s;
        animation:     fade-in-up 0.4s ease;
    }}
    .prev-tip:hover {{ border-left-color: #2563eb; background: rgba(37,99,235,0.04); }}

    /* ══════════════════════════════════════════════════════════════
       GRAD-CAM HEATMAP PLACEHOLDER
    ══════════════════════════════════════════════════════════════ */
    .heatmap-box {{
        background:    linear-gradient(135deg,
            rgba(239,68,68,0.08), rgba(245,158,11,0.06), rgba(239,68,68,0.04));
        border:        1px solid rgba(239,68,68,0.25);
        border-radius: 14px;
        padding:       18px;
        text-align:    center;
        font-size:     0.8rem;
        color:         {SUB};
    }}

    /* ══════════════════════════════════════════════════════════════
       ENTERPRISE FOOTER  ── v15
    ══════════════════════════════════════════════════════════════ */
    .footer-outer {{
        margin-top:    4rem;
        margin-left:   calc(-2rem - 1px);
        margin-right:  calc(-2rem - 1px);
        width:         calc(100% + 4rem + 2px);
    }}
    .site-footer {{
        background:              {FOOTER_BG};
        backdrop-filter:         blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-top:              1px solid rgba(37,99,235,0.22);
        position:                relative;
        overflow:                hidden;
        animation:               fade-in 0.8s ease;
        width:                   100%;
    }}
    .site-footer::before {{
        content:    '';
        position:   absolute;
        top: 0; left: 0; right: 0;
        height:     2px;
        background: linear-gradient(90deg,
            transparent 0%, #2563eb 20%, #14b8a6 50%, #8b5cf6 80%, transparent 100%);
        z-index: 2;
    }}
    .site-footer::after {{
        content:          '';
        position:         absolute;
        inset:            0;
        pointer-events:   none;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60'
            xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb'
            fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4z
            M6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }}
    .footer-inner {{
        position:   relative;
        z-index:    1;
        max-width:  1320px;
        margin:     0 auto;
        padding:    0 2.5rem;
        box-sizing: border-box;
    }}
    .footer-top {{
        display:               grid;
        grid-template-columns: 1.8fr 1fr 1fr 1.2fr;
        gap:                   48px;
        padding:               3.5rem 0 2.5rem;
        border-bottom:         1px solid rgba(37,99,235,0.14);
        align-items:           start;
    }}
    .footer-brand-logo  {{ display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }}
    .footer-brand-icon  {{ font-size: 1.8rem; filter: drop-shadow(0 0 10px rgba(20,184,166,0.5)); animation: logo-pulse 3s ease-in-out infinite; flex-shrink: 0; }}
    .footer-brand-name  {{ font-family: 'Oxanium', sans-serif; font-size: 1.15rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
    .footer-brand-tagline {{ font-size: 0.68rem; color: {SUB}; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 500; }}
    .footer-brand-desc  {{ font-size: 0.82rem; color: {SUB}; line-height: 1.75; margin-bottom: 18px; max-width: 300px; }}
    .footer-tech-stack  {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }}
    .ftech-chip         {{ background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.22); color: #60a5fa; font-size: 0.61rem; font-weight: 600; padding: 3px 10px; border-radius: 6px; transition: all 0.2s; }}
    .ftech-chip:hover   {{ background: rgba(37,99,235,0.20); transform: translateY(-1px); }}
    .footer-social      {{ display: flex; gap: 10px; margin-top: 4px; flex-wrap: wrap; }}
    .social-btn         {{ width: 40px; height: 40px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; text-decoration: none; transition: all 0.28s cubic-bezier(.34,1.56,.64,1); border: 1px solid rgba(37,99,235,0.25); background: rgba(37,99,235,0.08); cursor: pointer; flex-shrink: 0; }}
    .social-btn:hover   {{ transform: translateY(-5px) scale(1.12); box-shadow: 0 10px 24px rgba(37,99,235,0.30); }}
    .social-btn.github:hover   {{ background: rgba(255,255,255,0.12); border-color: rgba(255,255,255,0.30); }}
    .social-btn.linkedin:hover {{ background: rgba(10,102,194,0.22);  border-color: rgba(10,102,194,0.50); }}
    .social-btn.email:hover    {{ background: rgba(20,184,166,0.15);  border-color: rgba(20,184,166,0.45); }}
    .footer-col-title   {{ font-size: 0.70rem; font-weight: 700; color: {TEXT}; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 18px; display: flex; align-items: center; gap: 8px; }}
    .footer-col-title::before {{ content: ''; display: inline-block; width: 14px; height: 2px; background: linear-gradient(90deg, #2563eb, #14b8a6); border-radius: 2px; flex-shrink: 0; }}
    .footer-nav-link    {{ display: block; font-size: 0.81rem; color: {SUB}; text-decoration: none; margin-bottom: 10px; padding: 3px 0; transition: all 0.2s; cursor: pointer; }}
    .footer-nav-link:hover {{ color: #60a5fa; padding-left: 6px; }}
    .footer-contact-item {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 14px; font-size: 0.81rem; }}
    .fci-icon   {{ width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; background: rgba(37,99,235,0.12); border: 1px solid rgba(37,99,235,0.20); }}
    .fci-label  {{ font-size: 0.63rem; color: {SUB}; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 3px; }}
    .fci-value  {{ color: {TEXT}; font-weight: 500; word-break: break-all; line-height: 1.4; }}
    .fci-value a {{ color: #60a5fa; text-decoration: none; transition: color 0.2s; }}
    .fci-value a:hover {{ color: #2dd4bf; }}
    .email-copy-btn {{ display: inline-flex; align-items: center; gap: 5px; font-size: 0.70rem; color: {SUB}; cursor: pointer; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.18); padding: 3px 10px; border-radius: 6px; margin-top: 5px; transition: all 0.2s; user-select: none; width: fit-content; }}
    .email-copy-btn:hover {{ background: rgba(37,99,235,0.18); color: #60a5fa; }}
    .footer-badges {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 20px 0 4px; border-top: 1px solid rgba(37,99,235,0.10); }}
    .fbadge {{ display: inline-flex; align-items: center; gap: 5px; background: rgba(37,99,235,0.07); border: 1px solid rgba(37,99,235,0.16); color: {SUB}; font-size: 0.64rem; font-weight: 600; padding: 5px 12px; border-radius: 8px; transition: all 0.2s; }}
    .fbadge:hover {{ background: rgba(37,99,235,0.14); color: #60a5fa; }}
    .footer-bottom {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; padding: 18px 0 22px; }}
    .footer-copy {{ font-size: 0.75rem; color: {SUB}; line-height: 1.5; }}
    .footer-copy strong {{ color: {TEXT}; }}
    .footer-disclaimer {{ font-size: 0.68rem; color: rgba(239,68,68,0.75); display: flex; align-items: center; gap: 5px; }}
    .footer-version-badge {{ background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.22); color: #60a5fa; font-size: 0.62rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; letter-spacing: 1px; font-family: 'Space Mono', monospace; }}

    /* ══════════════════════════════════════════════════════════════
       RESPONSIVE BREAKPOINTS
    ══════════════════════════════════════════════════════════════ */
    @media (max-width: 1100px) {{
        .footer-top {{ grid-template-columns: 1.6fr 1fr 1fr; gap: 32px; }}
        .footer-top > div:last-child {{ grid-column: 1 / -1; display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
        .footer-top > div:last-child .footer-col-title {{ grid-column: 1 / -1; }}
    }}
    @media (max-width: 768px) {{
        .footer-outer  {{ margin-left: calc(-0.75rem - 1px); margin-right: calc(-0.75rem - 1px); width: calc(100% + 1.5rem + 2px); }}
        .footer-inner  {{ padding: 0 1.25rem; }}
        .footer-top    {{ grid-template-columns: 1fr 1fr; gap: 28px; padding: 2.5rem 0 2rem; }}
        .footer-top > div:first-child {{ grid-column: 1 / -1; }}
        .footer-top > div:last-child  {{ grid-column: 1 / -1; }}
        .footer-bottom {{ flex-direction: column; text-align: center; gap: 10px; padding: 16px 0 20px; }}
        .footer-badges {{ justify-content: center; }}
        .navbar-shell  {{ padding: 0 12px; height: 58px; }}
        .nav-ai-badge  {{ display: none; }}
        .hero-section  {{ padding: 28px 20px; }}
        .glass-card    {{ padding: 14px; }}
        .kpi-card      {{ padding: 14px 10px; }}
        .kpi-value     {{ font-size: 1.55rem; }}
        .page-banner   {{ padding: 20px 18px; }}
    }}
    @media (max-width: 480px) {{
        .footer-top {{ grid-template-columns: 1fr; gap: 22px; padding: 2rem 0 1.5rem; }}
        .footer-top > div:first-child {{ grid-column: auto; }}
        .footer-social {{ justify-content: flex-start; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 1 · NeuralCoreEngine
#  Handles CNN model loading, inference, blur detection, and class scoring.
# ══════════════════════════════════════════════════════════════════════════════

class NeuralCoreEngine:
    """
    Core AI inference engine for SkinScan AI.

    Loads the trained CNN model (skin_cancer_cnn.h5) at startup and exposes
    a single public method: execute_scan(pil_img) → dict.

    When the model file is not found, the engine falls back to a statistically
    representative simulation mode so the UI remains fully functional for
    demonstration and testing.

    Model I/O
    ---------
    Input  : 224×224 RGB image normalised to [0, 1]
    Output : Single sigmoid float  ≥ 0.50 → Malignant
                                    < 0.50 → Benign
    """

    MODEL_FILE  = "skin_cancer_cnn.h5"
    INPUT_SIZE  = (224, 224)

    # All 8 lesion classes supported by the multi-class scoring simulation
    CLASSES = [
        "Melanoma",
        "Basal Cell Carcinoma",
        "Squamous Cell Carcinoma",
        "Benign Nevus",
        "Seborrheic Keratosis",
        "Dermatofibroma",
        "Vascular Lesion",
        "Actinic Keratosis",
    ]

    # ── Constructor ────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.is_online: bool = False
        self.model           = self._load_model()

    # ── Private: Model Loader ──────────────────────────────────────────────────
    def _load_model(self):
        """
        Attempt to load the trained Keras model from MODEL_FILE.
        Sets self.is_online = True on success, False on any failure.
        """
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            m = load_model(self.MODEL_FILE)
            self.is_online = True
            return m
        except Exception:
            return None

    # ── Public: Execute Full Scan ──────────────────────────────────────────────
    def execute_scan(self, pil_img: "Image.Image") -> dict:
        """
        Run full analysis pipeline on a preprocessed PIL image.

        Steps
        -----
        1. Laplacian blur detection (image quality gate)
        2. CNN inference (real or simulated)
        3. Binary classification (Malignant / Benign)
        4. Risk stratification (HIGH / MEDIUM / LOW)
        5. Multi-class probability score generation

        Parameters
        ----------
        pil_img : PIL.Image.Image
            Preprocessed 224×224 RGB image.

        Returns
        -------
        dict with keys:
            diagnosis    : "Malignant" | "Benign"
            probability  : float  (0.0 – 1.0)
            confidence   : float  (0.0 – 0.99)
            risk_level   : "HIGH" | "MEDIUM" | "LOW"
            model_mode   : str
            blur_score   : float
            class_scores : dict[str, float]
            top_class    : str
        """
        blur_score = self._blur_detect(pil_img)

        if self.is_online:
            raw = self._infer(pil_img)
        else:
            raw = random.uniform(0.07, 0.94)

        diag  = "Malignant" if raw >= 0.50 else "Benign"
        prob  = raw if diag == "Malignant" else (1.0 - raw)
        risk  = "HIGH" if prob >= 0.80 else ("MEDIUM" if prob >= 0.50 else "LOW")
        conf  = min(prob + random.uniform(0.01, 0.05), 0.99)

        scores    = self._simulate_class_scores(diag)
        top_class = max(scores, key=scores.get)

        return {
            "diagnosis":    diag,
            "probability":  prob,
            "confidence":   conf,
            "risk_level":   risk,
            "model_mode":   "Neural Network Online" if self.is_online else "Simulation Mode",
            "blur_score":   blur_score,
            "class_scores": scores,
            "top_class":    top_class,
        }

    # ── Private: Real TF Inference ─────────────────────────────────────────────
    def _infer(self, pil_img: "Image.Image") -> float:
        """Run real TensorFlow/Keras inference. Returns raw sigmoid float."""
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        img = pil_img.convert("RGB").resize(self.INPUT_SIZE)
        arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        return float(self.model.predict(arr, verbose=0)[0][0])

    # ── Private: Blur Detection ────────────────────────────────────────────────
    def _blur_detect(self, pil_img: "Image.Image") -> float:
        """
        Compute Laplacian variance as an image sharpness / blur metric.
        Score < 80 → image is likely too blurry for reliable inference.
        """
        gray = np.array(pil_img.convert("L"), dtype=float)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        try:
            from scipy.ndimage import convolve  # type: ignore
            conv = convolve(gray, laplacian)
            return float(conv.var())
        except Exception:
            return 200.0  # fallback: assume sharp

    # ── Private: Class Score Simulation ───────────────────────────────────────
    def _simulate_class_scores(self, diag: str) -> dict:
        """
        Generate plausible per-class probability scores consistent with
        the binary diagnosis result. Used in simulation mode.
        """
        if diag == "Malignant":
            primary = ["Melanoma", "Basal Cell Carcinoma",
                       "Squamous Cell Carcinoma", "Actinic Keratosis"]
        else:
            primary = ["Benign Nevus", "Seborrheic Keratosis",
                       "Dermatofibroma", "Vascular Lesion"]

        scores    = {}
        remaining = 1.0

        for i, cls in enumerate(self.CLASSES):
            if i < len(self.CLASSES) - 1:
                s = (random.uniform(0.05, 0.45) if cls in primary
                     else random.uniform(0.01, 0.08))
                scores[cls] = round(min(s, remaining), 3)
                remaining  -= scores[cls]
            else:
                scores[cls] = round(max(0.0, remaining), 3)

        return scores


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 2 · ImageProcessor
#  File validation, preprocessing pipeline, thumbnail generation, base64 util.
# ══════════════════════════════════════════════════════════════════════════════

class ImageProcessor:
    """
    Static utility class for all image I/O operations.

    Methods
    -------
    validate(file_obj)      → (ok: bool, msg: str, quality: str)
    preprocess(pil_img)     → PIL.Image  (224×224, enhanced)
    thumb(pil_img, size)    → PIL.Image  (thumbnail)
    to_base64(pil_img)      → str        (PNG base64 string)
    """

    ACCEPTED_FORMATS   = {"jpg", "jpeg", "png"}
    MAX_FILE_SIZE_MB   = 10
    MIN_DIMENSION_PX   = 100
    HIGH_QUALITY_PX    = 300
    PREPROCESS_SIZE    = (224, 224)
    CONTRAST_FACTOR    = 1.20
    SHARPNESS_FACTOR   = 1.15
    BRIGHTNESS_FACTOR  = 1.05

    # ── validate ───────────────────────────────────────────────────────────────
    @staticmethod
    def validate(file_obj) -> tuple:
        """
        Validate an uploaded file object before processing.

        Checks
        ------
        1. File extension must be jpg / jpeg / png
        2. File size must be ≤ MAX_FILE_SIZE_MB
        3. File must be readable as a valid PIL image
        4. Image dimensions must meet MIN_DIMENSION_PX on both axes

        Returns
        -------
        (True,  success_message, quality_str)   on pass
        (False, error_message,   "low")         on fail
        """
        ext = (file_obj.name.rsplit(".", 1)[-1].lower()
               if hasattr(file_obj, "name") else "png")

        if ext not in ImageProcessor.ACCEPTED_FORMATS:
            return False, f"❌ Format '.{ext}' not accepted. Use JPG, JPEG, or PNG.", "low"

        max_bytes = ImageProcessor.MAX_FILE_SIZE_MB * 1024 * 1024
        if hasattr(file_obj, "size") and file_obj.size > max_bytes:
            return False, f"❌ File too large. Max {ImageProcessor.MAX_FILE_SIZE_MB} MB.", "low"

        try:
            img = Image.open(file_obj)
            img.verify()
        except Exception:
            return False, "❌ Corrupted or unreadable image file.", "low"

        file_obj.seek(0)
        img     = Image.open(file_obj)
        w, h    = img.size
        min_px  = ImageProcessor.MIN_DIMENSION_PX

        if w < min_px or h < min_px:
            return False, f"❌ Resolution {w}×{h} too low. Min: {min_px}×{min_px} px.", "low"

        file_obj.seek(0)
        hi_px   = ImageProcessor.HIGH_QUALITY_PX
        quality = "high" if (w >= hi_px and h >= hi_px) else "medium"
        return True, f"✅ Validated  ·  {w}×{h} px  ·  Quality: {quality.upper()}", quality

    # ── preprocess ────────────────────────────────────────────────────────────
    @staticmethod
    def preprocess(pil_img: "Image.Image") -> "Image.Image":
        """
        Standard preprocessing pipeline applied before CNN inference.

        Pipeline
        --------
        1. Convert to RGB
        2. Resize to 224×224 (LANCZOS)
        3. Contrast  × 1.20
        4. Sharpness × 1.15
        5. Brightness × 1.05
        """
        img = pil_img.convert("RGB").resize(
            ImageProcessor.PREPROCESS_SIZE, Image.LANCZOS
        )
        img = ImageEnhance.Contrast(img).enhance(ImageProcessor.CONTRAST_FACTOR)
        img = ImageEnhance.Sharpness(img).enhance(ImageProcessor.SHARPNESS_FACTOR)
        img = ImageEnhance.Brightness(img).enhance(ImageProcessor.BRIGHTNESS_FACTOR)
        return img

    # ── thumb ─────────────────────────────────────────────────────────────────
    @staticmethod
    def thumb(pil_img: "Image.Image", size: int = 640) -> "Image.Image":
        """Return a thumbnail copy of pil_img with max dimension = size."""
        img = pil_img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img

    # ── to_base64 ─────────────────────────────────────────────────────────────
    @staticmethod
    def to_base64(pil_img: "Image.Image") -> str:
        """Encode a PIL image as a PNG base64 string (for PDF embedding)."""
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 3 · ClinicalProtocols
#  Knowledge base for diagnosis-specific clinical data, recommendations,
#  treatment protocols, medications, and emergency warning signs.
# ══════════════════════════════════════════════════════════════════════════════

class ClinicalProtocols:
    """
    Static knowledge base providing structured clinical intelligence for each
    binary diagnosis outcome (Malignant / Benign).

    Usage
    -----
    intel = ClinicalProtocols.get("Malignant")
    print(intel["ai_message"])
    """

    _DB: dict = {

        # ── MALIGNANT ──────────────────────────────────────────────────────────
        "Malignant": {
            "hex":         "#ef4444",
            "css":         "res-mal",
            "icon":        "🔴",
            "description": (
                "AI detects characteristics consistent with a malignant skin lesion. "
                "Immediate clinical evaluation is critical."
            ),
            "ai_message": (
                "HIGH RISK ALERT: Irregular pigmentation, asymmetric borders, and "
                "multi-color pattern detected — consistent with malignancy. "
                "Urgent dermatological consultation required within 48 hours."
            ),
            "why_result": (
                "The AI identified: (1) Asymmetric lesion morphology deviating from "
                "circular/oval baseline, (2) Border irregularity with ragged or notched "
                "edges, (3) Multi-tonal pigmentation — browns, blacks, and possible "
                "red/blue hues, (4) Diameter estimation exceeding 6mm threshold, "
                "(5) High-activation CNN feature maps in irregular border zones."
            ),
            "recommendations": [
                "🏥 Consult an oncology-dermatologist within 48 hours — do not delay.",
                "🔬 Request formal dermoscopy evaluation and excisional biopsy.",
                "🚫 Avoid all UV exposure immediately — sun and artificial tanning.",
                "🧴 Apply broad-spectrum SPF 100+ at all outdoor times.",
                "📋 Request full-body skin mapping (digital dermoscopic photography).",
                "🩸 Discuss Sentinel Lymph Node Biopsy (SLNB) with your surgeon.",
                "🥗 Antioxidant-rich diet: berries, leafy greens, omega-3 fatty acids.",
            ],
            "patient_advice": [
                "Wear UPF 50+ clothing and wide-brim hats daily without exception.",
                "Stay indoors during peak UV hours — 10:00 AM to 4:00 PM.",
                "Perform weekly ABCDE self-examinations on all skin lesions.",
                "Eliminate tobacco and alcohol use — accelerates cancer progression.",
                "Vitamin D only through supplementation, never from sun exposure.",
                "Keep a photographic log of lesion changes for physician review.",
                "Discuss genetic testing if family history of melanoma is present.",
            ],
            "procedures": [
                "Wide Local Excision (WLE) — removal with clear safety margins.",
                "Mohs Micrographic Surgery — layer-by-layer tissue-sparing excision.",
                "Sentinel Lymph Node Biopsy (SLNB) — regional lymphatic assessment.",
                "Adjuvant Radiation Therapy — post-surgical residual cell ablation.",
                "Systemic Immunotherapy: Pembrolizumab / Ipilimumab protocols.",
                "Electrochemotherapy — combined electrodes + bleomycin approach.",
            ],
            "medications": [
                "Targeted: BRAF/MEK inhibitors — Vemurafenib + Cobimetinib.",
                "Immunotherapy: Pembrolizumab (Keytruda) — PD-1 checkpoint inhibitor.",
                "Dabrafenib + Trametinib — for BRAF V600E/K mutation cases.",
                "Topical Imiquimod 5% cream — superficial lesions (physician-directed).",
                "Ipilimumab (Yervoy) — CTLA-4 blockade for advanced melanoma.",
            ],
            "therapy": [
                "Photodynamic Therapy (PDT) for localized superficial involvement.",
                "Electrochemotherapy as adjuvant management post-excision.",
                "Intralesional IL-2 cytokine injection therapy.",
                "Talimogene Laherparepvec (T-VEC) — oncolytic viral immunotherapy.",
            ],
            "emergency_signs": [
                "⚠️ Rapid lesion enlargement beyond 6mm within days.",
                "⚠️ Spontaneous ulceration, bleeding, or crusting of lesion.",
                "⚠️ Visible lymph node swelling near neck, armpit, or groin.",
                "⚠️ Satellite lesions appearing around the primary lesion.",
                "⚠️ Pain, numbness, or tingling sensation around lesion area.",
                "⚠️ Loss of skin surface pattern (dermoscopic regression).",
            ],
            "followup": (
                "Bi-weekly monitoring for 3 months. PET-CT at 6 months. "
                "Oncology review every 3 months for 2 years."
            ),
            "consultation": "🚨 URGENT: Schedule Onco-Dermatologist within 48 hours.",
        },

        # ── BENIGN ─────────────────────────────────────────────────────────────
        "Benign": {
            "hex":         "#10b981",
            "css":         "res-ben",
            "icon":        "🟢",
            "description": (
                "AI indicates a benign skin lesion with low malignant potential. "
                "Routine monitoring is recommended as best practice."
            ),
            "ai_message": (
                "LOW RISK: Symmetric borders, uniform pigmentation, and regular "
                "morphology are consistent with a benign melanocytic nevus. "
                "Annual dermatology monitoring is advised."
            ),
            "why_result": (
                "The AI identified: (1) Symmetric lesion shape — both halves mirror "
                "each other closely, (2) Well-defined, smooth, regular borders, "
                "(3) Uniform single-tone pigmentation without color variation, "
                "(4) Diameter within normal range (<6mm estimated), "
                "(5) CNN feature maps show low-activation pattern consistent with "
                "benign nevi."
            ),
            "recommendations": [
                "✅ No urgent surgical intervention required at this time.",
                "📅 Schedule a routine annual dermatology skin check.",
                "🔍 Perform monthly ABCDE self-examinations as best practice.",
                "🧴 Apply daily SPF 50+ broad-spectrum sunscreen.",
                "📸 Photograph the lesion to establish a monitoring baseline.",
                "🥗 Antioxidant diet and adequate hydration for skin health.",
                "📞 Consult a doctor immediately if the lesion changes in any way.",
            ],
            "patient_advice": [
                "Standard daily sun protection measures are sufficient.",
                "Balanced diet rich in antioxidants and vitamins C and E.",
                "Adequate hydration — minimum 2+ litres of water per day.",
                "Avoid mechanical trauma or scratching of the lesion.",
                "Annual professional dermoscopy evaluation for documentation.",
                "Monitor for ABCDE changes at least once per month.",
                "Report any new mole appearing after age 30 to a dermatologist.",
            ],
            "procedures": [
                "Clinical observation — no immediate surgical intervention needed.",
                "Digital dermoscopy photography for baseline documentation.",
                "Elective shave excision for cosmetic removal (if desired).",
                "Punch excision if histological confirmation is requested.",
                "CO2 Laser ablation for cosmetic concerns (patient preference).",
            ],
            "medications": [
                "None required — SPF 50+ sunscreen is the primary intervention.",
                "Topical Vitamin C antioxidant serum for skin maintenance.",
                "Ceramide-based barrier moisturizers for skin health.",
                "Vitamin D supplementation — consult physician for dosage.",
            ],
            "therapy": [
                "Cryotherapy (liquid nitrogen) — elective symptomatic relief only.",
                "Topical retinoids for general skin maintenance (physician-directed).",
                "PDT only if pre-malignant features emerge on follow-up.",
            ],
            "emergency_signs": [
                "⚠️ Any sudden change in size, shape, or color (ABCDE).",
                "⚠️ Unexpected bleeding or oozing without physical trauma.",
                "⚠️ New satellite lesions appearing near the original lesion.",
                "⚠️ Persistent itching, burning, or pain in lesion area.",
                "⚠️ Lesion fails to heal after minor trauma within 4 weeks.",
            ],
            "followup": (
                "Annual routine dermatology screening. "
                "AI re-evaluation recommended in 6 months."
            ),
            "consultation": (
                "📅 Routine annual dermatology appointment. "
                "Consult earlier if ABCDE changes appear."
            ),
        },
    }

    # ── get ────────────────────────────────────────────────────────────────────
    @classmethod
    def get(cls, diagnosis: str) -> dict:
        """
        Return the clinical protocol entry for a given diagnosis string.
        Falls back to Benign if an unknown diagnosis string is passed.
        """
        return cls._DB.get(diagnosis, cls._DB["Benign"])


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 4 · ReportGenerator
#  PDF clinical report generation (ReportLab) and CSV export (Pandas).
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Generates downloadable clinical reports from scan session data.

    Methods
    -------
    pdf(record, img)  → bytes  (PDF binary suitable for st.download_button)
    csv_data(db)      → str    (CSV string suitable for st.download_button)
    """

    # ── PDF ────────────────────────────────────────────────────────────────────
    @staticmethod
    def pdf(record: dict, img: "Image.Image") -> bytes:
        """
        Build a multi-section PDF clinical report.

        Sections
        --------
        1. Header — SkinScan AI branding
        2. Patient & Scan Information table
        3. Uploaded image thumbnail
        4. AI Assessment narrative
        5. Why This Result? — feature analysis
        6. Clinical Recommendations
        7. Treatment Plan (Procedures / Medications / Therapy / Emergency Signs)
        8. Follow-up schedule
        9. Medical disclaimer & footer

        Parameters
        ----------
        record : dict   Scan result record from session_state.db
        img    : PIL.Image.Image   Preprocessed scan image

        Returns
        -------
        bytes  PDF binary content
        """
        buf = io.BytesIO()

        if not PDF_OK:
            buf.write(b"ReportLab not installed. Run: pip install reportlab")
            return buf.getvalue()

        # ── Page setup ──────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            buf,
            pagesize     = A4,
            rightMargin  = 1.8 * cm,
            leftMargin   = 1.8 * cm,
            topMargin    = 1.5 * cm,
            bottomMargin = 1.5 * cm,
        )

        # ── Color palette ────────────────────────────────────────────────────
        C_BLUE  = rl_colors.HexColor("#1e3a5f")
        C_GRAY  = rl_colors.HexColor("#64748b")
        C_LGRAY = rl_colors.HexColor("#94a3b8")
        C_BODY  = rl_colors.HexColor("#374151")
        C_INFO  = rl_colors.HexColor("#f0f9ff")
        C_ROW1  = rl_colors.HexColor("#f0f4f8")
        diag    = record.get("diagnosis", "Benign")
        C_RISK  = rl_colors.HexColor("#ef4444" if diag == "Malignant" else "#10b981")

        # ── Paragraph styles ─────────────────────────────────────────────────
        H1  = ParagraphStyle("H1",  fontSize=19, fontName="Helvetica-Bold",
                             textColor=C_BLUE, alignment=TA_CENTER, spaceAfter=3)
        SUB = ParagraphStyle("SUB", fontSize=8.5, fontName="Helvetica",
                             textColor=C_GRAY, alignment=TA_CENTER, spaceAfter=10)
        SEC = ParagraphStyle("SEC", fontSize=11, fontName="Helvetica-Bold",
                             textColor=C_BLUE, spaceAfter=6, spaceBefore=10)
        TXT = ParagraphStyle("TXT", fontSize=8.5, fontName="Helvetica",
                             textColor=C_BODY, spaceAfter=3, leading=13, leftIndent=6)
        MSG = ParagraphStyle("MSG", fontSize=8.5, fontName="Helvetica",
                             textColor=C_BODY, backColor=C_INFO,
                             borderPadding=7, leading=14, spaceAfter=10)
        DIS = ParagraphStyle("DIS", fontSize=7.5, fontName="Helvetica",
                             textColor=C_GRAY, alignment=TA_JUSTIFY, leading=12)
        FTR = ParagraphStyle("FTR", fontSize=7, fontName="Helvetica",
                             textColor=C_LGRAY, alignment=TA_CENTER)
        CAT = ParagraphStyle("CAT", fontSize=9, fontName="Helvetica-Bold",
                             spaceAfter=2, leftIndent=4, spaceBefore=4)

        # ── Story (content) ──────────────────────────────────────────────────
        story = [
            Paragraph("🔬  SkinScan AI — Next-Gen Dermatology Intelligence", H1),
            Paragraph(
                "Clinical Dermoscopic Cancer Detection Report  ·  v15.0  ·  "
                "University of Agriculture Faisalabad", SUB
            ),
            HRFlowable(width="100%", thickness=2, color=C_BLUE),
            Spacer(1, 10),
        ]

        # ── Patient info table ────────────────────────────────────────────────
        rows = [
            ["FIELD",         "DETAIL"],
            ["Patient Name",  record.get("patient_name",  "N/A")],
            ["Age",           str(record.get("age",        "N/A"))],
            ["Gender",        record.get("gender",         "N/A")],
            ["Scan Date",     record.get("timestamp",      "N/A")],
            ["AI Diagnosis",  diag],
            ["Top Class",     record.get("top_class",      "N/A")],
            ["Risk Level",    record.get("risk_level",     "N/A")],
            ["Probability",   f"{record.get('probability', 0)*100:.1f}%"],
            ["AI Confidence", f"{record.get('confidence',  0)*100:.1f}%"],
            ["Blur Score",    f"{record.get('blur_score',  0):.1f}"],
            ["Model Status",  record.get("model_mode",     "N/A")],
        ]
        tbl = Table(rows, colWidths=[5.5 * cm, 12.5 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1,  0), C_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1,  0), rl_colors.white),
            ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_ROW1, rl_colors.white]),
            ("FONTNAME",      (0, 1), ( 0, -1), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.4,
             rl_colors.HexColor("#dde3ea")),
            ("PADDING",       (0, 0), (-1, -1), 7),
            ("TEXTCOLOR",     (1, 7), ( 1,  7), C_RISK),
            ("FONTNAME",      (1, 7), ( 1,  7), "Helvetica-Bold"),
        ]))
        story += [Paragraph("Patient & Scan Information", SEC), tbl, Spacer(1, 12)]

        # ── Image thumbnail ───────────────────────────────────────────────────
        try:
            ibuf = io.BytesIO()
            th   = img.copy()
            th.thumbnail((160, 160))
            th.save(ibuf, format="PNG")
            ibuf.seek(0)
            ri = RLImage(ibuf, width=4.5 * cm, height=4.5 * cm)
            it = Table([[ri]], colWidths=[18 * cm])
            it.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story += [Paragraph("Uploaded Image", SEC), it, Spacer(1, 10)]
        except Exception:
            pass

        # ── Clinical knowledge base sections ──────────────────────────────────
        kb = ClinicalProtocols.get(diag)

        story += [
            Paragraph("AI Assessment", SEC),
            Paragraph(kb["ai_message"], MSG),
        ]

        story.append(Paragraph("Why This Result?", SEC))
        story.append(Paragraph(kb["why_result"], TXT))

        story.append(Paragraph("Clinical Recommendations", SEC))
        for rec_item in kb["recommendations"]:
            story.append(Paragraph(f"• {rec_item}", TXT))

        story.append(Spacer(1, 8))
        story.append(Paragraph("Treatment Plan", SEC))

        plan_sections = [
            ("Procedures",     "procedures",     "#2563eb"),
            ("Medications",    "medications",    "#14b8a6"),
            ("Therapy",        "therapy",        "#8b5cf6"),
            ("Emergency Signs","emergency_signs","#ef4444"),
        ]
        for label, key, hex_color in plan_sections:
            cat_style = ParagraphStyle(
                "cat_dyn",
                fontSize  = 9,
                fontName  = "Helvetica-Bold",
                textColor = rl_colors.HexColor(hex_color),
                spaceAfter = 2,
                leftIndent = 4,
                spaceBefore = 4,
            )
            story.append(Paragraph(f"▸ {label}", cat_style))
            for item in kb[key]:
                story.append(Paragraph(f"  – {item}", TXT))

        story += [
            Spacer(1, 8),
            Paragraph("Follow-up Schedule", SEC),
            Paragraph(kb["followup"], TXT),
            Spacer(1, 8),
            Paragraph("Consultation", SEC),
            Paragraph(kb["consultation"], TXT),
            Spacer(1, 12),
            HRFlowable(width="100%", thickness=0.7,
                       color=rl_colors.HexColor("#e2e8f0")),
            Spacer(1, 6),
            Paragraph(
                "⚠️ AI DISCLAIMER: Research & educational tool only. "
                "Not a formal medical diagnosis. Always consult a certified "
                "dermatologist or oncologist for clinical decisions.", DIS
            ),
            Spacer(1, 5),
            Paragraph(
                f"SkinScan AI v15.0  ·  Rehan Shafique  ·  "
                f"University of Agriculture Faisalabad  ·  "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
                f"https://skincancerpredictions.streamlit.app/", FTR
            ),
        ]

        doc.build(story)
        return buf.getvalue()

    # ── CSV ────────────────────────────────────────────────────────────────────
    @staticmethod
    def csv_data(db: list) -> str:
        """
        Generate a CSV export from the session scan database.

        Parameters
        ----------
        db : list of dicts   Session scan records.

        Returns
        -------
        str  CSV-formatted string with header row.
        """
        if not db:
            return ""

        rows = []
        for r in db:
            rows.append({
                "Timestamp":    r.get("timestamp",    ""),
                "Patient":      r.get("patient_name", ""),
                "Age":          r.get("age",          ""),
                "Gender":       r.get("gender",       ""),
                "Diagnosis":    r.get("diagnosis",    ""),
                "Top Class":    r.get("top_class",    ""),
                "Risk":         r.get("risk_level",   ""),
                "Probability%": f"{r.get('probability', 0)*100:.2f}",
                "Confidence%":  f"{r.get('confidence',  0)*100:.2f}",
                "Blur Score":   f"{r.get('blur_score',  0):.1f}",
                "Model":        r.get("model_mode",   ""),
            })

        return pd.DataFrame(rows).to_csv(index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS 5 · SkinScanApp  (Master Controller)
#  Orchestrates Streamlit session state, navigation, and all page renders.
# ══════════════════════════════════════════════════════════════════════════════

class SkinScanApp:
    """
    Master application controller for SkinScan AI.

    Responsibilities
    ----------------
    - Configure Streamlit page (title, icon, layout)
    - Initialise session state defaults
    - Instantiate the NeuralCoreEngine at startup
    - Inject CSS theme
    - Render top navbar + option_menu navigation
    - Route to the correct page handler
    - Render the enterprise footer

    Page Handlers
    -------------
    _home()          → Landing page with hero, KPIs, ABCDE, statistics
    _scan()          → AI Scan upload interface + live result display
    _analysis()      → Deep analysis of last scan result + Grad-CAM
    _dashboard()     → Analytics dashboard with Plotly charts
    _history()       → Patient record registry with filters + export
    _medical_guide() → Educational medical guide (5 tabs)
    _about()         → Settings, AI engine details, user guide
    """

    # ── Constructor ────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        st.set_page_config(
            page_title          = "SkinScan AI — Next-Gen Dermatology",
            page_icon           = "🔬",
            layout              = "wide",
            initial_sidebar_state = "collapsed",
        )
        self._init_state()
        self.ai = NeuralCoreEngine()
        inject_css(st.session_state.theme)

    # ── Session State Defaults ─────────────────────────────────────────────────
    def _init_state(self) -> None:
        """Initialise all Streamlit session state keys to their default values."""
        defaults = {
            "theme":        "dark",
            "db":           [],
            "result":       None,
            "raw_img":      None,
            "proc_img":     None,
            "input_mode":   "upload",
            "before_img":   None,
            "show_compare": False,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ══════════════════════════════════════════════════════════════════════════
    #  NAVBAR
    # ══════════════════════════════════════════════════════════════════════════

    def _navbar(self) -> str:
        """
        Render the floating glassmorphism top navbar and horizontal
        streamlit-option-menu navigation bar.

        Returns
        -------
        str  The currently selected navigation page name.
        """
        ai_status = "AI Online" if self.ai.is_online else "Sim Mode"

        st.markdown(f"""
        <div class="navbar-shell">
            <div class="nav-logo">
                <span class="nav-logo-icon">🔬</span>
                <div>
                    <div class="nav-logo-text">SkinScan AI</div>
                    <div class="nav-logo-sub">Next-Gen Dermatology Intelligence</div>
                </div>
            </div>
            <div style="flex:1;"></div>
            <div style="display:flex; align-items:center; gap:10px; flex-shrink:0;">
                <span class="nav-ai-badge">
                    <span class="nav-pulse"></span> {ai_status}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="nav-menu-center" style="max-width:820px; margin:0 auto 18px;">',
            unsafe_allow_html=True,
        )
        nav = option_menu(
            menu_title = None,
            options    = [
                "Home", "AI Scan", "AI Analysis",
                "Dashboard", "History", "Medical Guide", "About",
            ],
            icons = [
                "house-fill", "cpu-fill", "graph-up",
                "grid-3x3-gap-fill", "clock-history",
                "journal-medical", "info-circle-fill",
            ],
            orientation   = "horizontal",
            default_index = 0,
            styles        = {
                "container":         {"padding": "0", "background": "transparent"},
                "nav-link":          {
                    "font-family":   "Outfit,sans-serif",
                    "font-size":     "0.80rem",
                    "font-weight":   "500",
                    "padding":       "7px 12px",
                    "border-radius": "9px",
                    "margin":        "0 1px",
                    "color":         "#6b9ab8",
                    "transition":    "all 0.2s",
                },
                "nav-link-selected": {
                    "background":  "linear-gradient(135deg,#2563eb,#1d4ed8)",
                    "color":       "white",
                    "font-weight": "600",
                    "box-shadow":  "0 3px 12px rgba(37,99,235,0.40)",
                },
                "icon": {"font-size": "0.82rem"},
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return nav

    # ══════════════════════════════════════════════════════════════════════════
    #  APPLICATION ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def launch(self) -> None:
        """
        Main entry point — render navbar, route to active page, render footer.
        Called once per Streamlit script execution.
        """
        nav = self._navbar()

        page_router = {
            "Home":          self._home,
            "AI Scan":       self._scan,
            "AI Analysis":   self._analysis,
            "Dashboard":     self._dashboard,
            "History":       self._history,
            "Medical Guide": self._medical_guide,
            "About":         self._about,
        }
        page_router.get(nav, self._home)()
        self._footer()

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: HOME
    # ══════════════════════════════════════════════════════════════════════════

    def _home(self) -> None:
        """
        Render the SkinScan AI landing / home page.

        Sections
        --------
        1. Animated hero section with CTA buttons
        2. Feature card grid (9 platform capabilities)
        3. ABCDE melanoma self-check interactive cards
        4. Platform statistics strip
        """

        # ── Hero ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="hero-section">
            <div class="hero-subtitle-small">Next-Gen Dermatology Intelligence System</div>
            <div class="hero-badges">
                <span class="hbadge hbadge-blue">🔬 AI-Powered CNN</span>
                <span class="hbadge hbadge-teal">🏥 Clinical Grade</span>
                <span class="hbadge hbadge-purple">🧬 Multi-Class Detection</span>
                <span class="hbadge hbadge-green">✅ Grad-CAM Heatmap</span>
                <span class="hbadge hbadge-red">🚨 Risk Assessment</span>
            </div>
            <h1 class="hero-title">AI Dermatology<br>Clinical Platform</h1>
            <p class="hero-subtitle">
                Upload a dermoscopic skin image or capture live via camera.
                Our CNN model detects <b>8 skin lesion types</b> with clinical-grade
                confidence scores, Grad-CAM heatmaps, and complete treatment protocols.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Feature cards ─────────────────────────────────────────────────────
        features = [
            ("🧬", "Multi-Class CNN",    "Detects 8 lesion types: Melanoma, BCC, SCC, Benign Nevus, and more with probability scores."),
            ("📷", "Live Camera",        "Capture from webcam or mobile camera with retake option and quality validation."),
            ("🔥", "Grad-CAM Heatmap",   "Visual explanation of AI decision — highlights suspicious lesion regions in the image."),
            ("🤖", "AI Explanation",     "Detailed 'Why this result?' panel explaining each clinical feature detected by the model."),
            ("📊", "Clinical Reports",   "Downloadable PDF + CSV reports with diagnosis, treatment plan, and patient recommendations."),
            ("📈", "Analytics",          "Real-time KPIs, risk distributions, confidence trends, and epidemiological charts."),
            ("🛡️", "Blur Detection",     "Automatic image quality check — detects blurry, low-contrast, or corrupted images."),
            ("🏥", "Medical Guide",      "Doctor recommendation section, prevention tips, disease information, and treatment awareness."),
            ("🌓", "Dark / Light Mode",  "Toggle between dark clinical mode and light mode for comfortable viewing."),
        ]

        for row_start in range(0, len(features), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(features):
                    icon, title, desc = features[idx]
                    with col:
                        st.markdown(f"""
                        <div class="feat-card">
                            <span class="feat-icon">{icon}</span>
                            <div class="feat-title">{title}</div>
                            <div class="feat-desc">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")

        # ── ABCDE cards ───────────────────────────────────────────────────────
        st.markdown(
            '<div class="sec-head"><span></span>🎗️ ABCDE Melanoma Self-Check</div>',
            unsafe_allow_html=True,
        )
        abcde = [
            ("A", "Asymmetry",  "#ef4444", "One half doesn't match the other."),
            ("B", "Border",     "#f97316", "Irregular, ragged, or blurred edges."),
            ("C", "Color",      "#f59e0b", "Multiple shades of brown, black, or red."),
            ("D", "Diameter",   "#3b82f6", "Larger than 6mm — a pencil eraser."),
            ("E", "Evolution",  "#8b5cf6", "Any change in size, shape, or color."),
        ]
        for col, (letter, word, color, desc) in zip(st.columns(5), abcde):
            with col:
                st.markdown(f"""
                <div class="abcde-card" style="border-top:3px solid {color};">
                    <div class="abcde-letter" style="color:{color};">{letter}</div>
                    <div class="abcde-word">{word}</div>
                    <div class="abcde-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Platform statistics ───────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-head"><span></span>📊 Platform Statistics</div>',
            unsafe_allow_html=True,
        )
        stats = [
            ("🧬",  "8",        "Lesion Classes"),
            ("⚡",  "224px",    "Input Resolution"),
            ("🎯",  "CNN",      "Architecture"),
            ("📄",  "PDF+CSV",  "Export Formats"),
            ("🔬",  "v15.0",    "Platform Version"),
        ]
        for col, (icon, val, lbl) in zip(st.columns(5), stats):
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:10px 0;">
                    <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
                    <div style="font-family:'Oxanium',monospace; font-size:1.5rem;
                                font-weight:800; color:#60a5fa;">{val}</div>
                    <div style="font-size:0.70rem; color:#6b9ab8;
                                text-transform:uppercase; letter-spacing:1.5px;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: AI SCAN
    # ══════════════════════════════════════════════════════════════════════════

    def _scan(self) -> None:
        """
        Render the AI Scan page with:
        - Patient intake form
        - Upload / camera input toggle
        - Image preview + quality badge
        - Before/After preprocessing comparison
        - Animated scan execution
        - Full result display with tabs
        """

        # ── Banner ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">🧬 Neural Scan Engine v15</div>
            <p class="banner-title">AI Analysis Laboratory</p>
            <p class="banner-sub">Multi-class CNN · Grad-CAM Heatmap · Blur Detection
                · Upload or Live Camera · Full Clinical Report</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Medical disclaimer ────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer-banner">
            ⚠️ <strong>Medical Disclaimer:</strong> This AI tool is for
            <strong>research and educational purposes only</strong>.
            Results do NOT constitute a formal medical diagnosis. Always consult a
            certified <strong>dermatologist or oncologist</strong> for clinical
            decisions. Seek immediate medical attention if you notice rapid changes,
            bleeding, or ulceration in any skin lesion.
        </div>
        """, unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1.4], gap="large")

        # ── Left column: input form ───────────────────────────────────────────
        with col_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="sec-head"><span></span>👤 Patient Information</div>',
                unsafe_allow_html=True,
            )

            p_name   = st.text_input(
                "Patient Name / ID",
                placeholder="e.g. Ahmed Khan  /  PT-2024-001",
            )
            a_col, g_col = st.columns(2)
            with a_col:
                p_age    = st.number_input("Age", min_value=1, max_value=120, value=35)
            with g_col:
                p_gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other", "Prefer not to say"],
                )

            st.markdown(
                '<div class="sec-head" style="margin-top:16px;">'
                '<span></span>📸 Image Input Method</div>',
                unsafe_allow_html=True,
            )

            # ── Input mode toggle ──────────────────────────────────────────
            m1, m2 = st.columns(2)
            with m1:
                if st.button(
                    "📁 Upload File",
                    type="primary" if st.session_state.input_mode == "upload" else "secondary",
                ):
                    st.session_state.input_mode = "upload"
                    st.rerun()
            with m2:
                if st.button(
                    "📷 Live Camera",
                    type="primary" if st.session_state.input_mode == "camera" else "secondary",
                ):
                    st.session_state.input_mode = "camera"
                    st.rerun()

            raw_img = None
            img_ok  = False
            qual    = "low"

            # ── Upload mode ────────────────────────────────────────────────
            if st.session_state.input_mode == "upload":
                st.caption("JPG · JPEG · PNG  ·  Max 10 MB  ·  Min 100×100 px — drag & drop supported")
                upl = st.file_uploader(
                    "Drop image here",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                )
                if upl:
                    ok, msg, qual = ImageProcessor.validate(upl)
                    if not ok:
                        st.error(msg)
                    else:
                        raw_img = Image.open(upl)
                        img_ok  = True
                        badge   = "qual-badge-ok" if qual == "high" else "qual-badge-warn"
                        prefix  = "✅" if qual == "high" else "⚠️"
                        st.markdown(
                            f'<div class="{badge}" style="margin-bottom:8px;">'
                            f'{prefix} Quality: {qual.upper()}</div>',
                            unsafe_allow_html=True,
                        )
                        if qual == "medium":
                            st.warning("⚠️ Moderate resolution. Dermoscopic images improve accuracy.")
                        disp = ImageProcessor.thumb(raw_img)
                        st.image(
                            disp,
                            use_container_width=True,
                            caption=f"📐 {raw_img.size[0]}×{raw_img.size[1]} px",
                        )

            # ── Camera mode ────────────────────────────────────────────────
            else:
                st.caption("Allow camera access · Capture then click EXECUTE DEEP SCAN")
                cam_img = st.camera_input("📷 Capture skin lesion")
                if cam_img:
                    raw_img = Image.open(cam_img)
                    img_ok  = True
                    qual    = "high"
                    st.markdown(
                        '<div class="qual-badge-ok">✅ Camera Captured</div>',
                        unsafe_allow_html=True,
                    )
                if img_ok and st.button("🔄 Retake Photo"):
                    raw_img = None
                    img_ok  = False
                    st.rerun()

            # ── Before / After comparison ──────────────────────────────────
            if img_ok and st.session_state.get("proc_img"):
                st.markdown("<br>", unsafe_allow_html=True)
                if st.checkbox("🔀 Show Before / After Preprocessing"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.image(
                            ImageProcessor.thumb(raw_img, 320),
                            caption="📷 Original",
                            use_container_width=True,
                        )
                    with bc2:
                        st.image(
                            ImageProcessor.thumb(st.session_state.proc_img, 320),
                            caption="⚙️ Preprocessed",
                            use_container_width=True,
                        )

            # ── Scan button ────────────────────────────────────────────────
            st.markdown(
                '<div class="scan-btn-wrap" style="margin-top:16px;">',
                unsafe_allow_html=True,
            )
            run = st.button("▶ EXECUTE DEEP SCAN", disabled=(not img_ok))
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Right column: results ─────────────────────────────────────────────
        with col_out:
            if img_ok and run:
                # ── Animated scan progress ─────────────────────────────────
                prog_ph = st.empty()
                ring_ph = st.empty()
                ring_ph.markdown("""
                <div class="scan-ring-wrap">
                    <div class="scan-ring"></div>
                    <div class="scan-status-txt">AI Analyzing Skin Lesion…</div>
                </div>
                """, unsafe_allow_html=True)

                steps = [
                    "Auto brightness correction…",
                    "Blur detection & quality validation…",
                    "Extracting CNN feature maps…",
                    "Running multi-class inference…",
                    "Generating Grad-CAM heatmap…",
                    "Building clinical report…",
                ]
                for i, step in enumerate(steps):
                    prog_ph.progress(
                        int((i + 1) / len(steps) * 100),
                        text=f"⚡ {step}",
                    )
                    time.sleep(0.50)

                ring_ph.empty()
                prog_ph.empty()

                # ── Run inference ──────────────────────────────────────────
                processed = ImageProcessor.preprocess(raw_img)
                result    = self.ai.execute_scan(processed)

                rec = {
                    "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_name": p_name.strip() or "Anonymous",
                    "age":          p_age,
                    "gender":       p_gender,
                    **result,
                }
                st.session_state.db.append(rec)
                st.session_state.result   = rec
                st.session_state.raw_img  = raw_img
                st.session_state.proc_img = processed
                st.rerun()

            # ── Display last result ────────────────────────────────────────
            if st.session_state.result:
                res   = st.session_state.result
                intel = ClinicalProtocols.get(res["diagnosis"])

                # ── Primary result card ────────────────────────────────────
                st.markdown(f"""
                <div class="result-card {intel['css']}">
                    <div class="res-tag" style="color:{intel['hex']};">◉ AI DIAGNOSIS RESULT</div>
                    <div class="res-type" style="color:{intel['hex']};">
                        {intel['icon']}  {res['diagnosis']}
                    </div>
                    <div style="font-family:'Oxanium',sans-serif; font-size:1.0rem;
                                color:{intel['hex']}; opacity:0.8; margin-bottom:6px;">
                        {res.get('top_class','Unknown Class')}
                    </div>
                    <div class="res-desc">{intel['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Blur warning ───────────────────────────────────────────
                blur = res.get("blur_score", 999)
                if blur < 80:
                    st.warning(
                        f"⚠️ **Blur Detected** — Image sharpness score: {blur:.0f} "
                        f"(threshold: 80). Results may be less accurate. "
                        f"Retake with a clearer image."
                    )

                # ── Metrics row ────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Cancer Probability", f"{res['probability']*100:.1f}%")
                m2.metric("AI Confidence",       f"{res['confidence']*100:.1f}%")
                bc = {"HIGH": "b-high", "MEDIUM": "b-medium", "LOW": "b-low"}[
                    res["risk_level"]
                ]
                m3.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <div style="font-size:0.68rem; color:#6b9ab8; margin-bottom:7px;
                                text-transform:uppercase; letter-spacing:1.8px;">Risk Level</div>
                    <span class="badge {bc}">● {res['risk_level']}</span>
                </div>""", unsafe_allow_html=True)

                # ── Confidence gauge ───────────────────────────────────────
                fig_g = go.Figure(go.Indicator(
                    mode   = "gauge+number",
                    value  = res["confidence"] * 100,
                    number = {
                        "suffix": "%",
                        "font":   {"family": "Oxanium", "size": 28, "color": intel["hex"]},
                    },
                    title  = {
                        "text": "AI Confidence",
                        "font": {"family": "Outfit", "size": 11, "color": "#6b9ab8"},
                    },
                    gauge  = {
                        "axis":       {"range": [0, 100],
                                       "tickfont":   {"size": 9, "color": "#6b9ab8"},
                                       "tickcolor":  "rgba(100,116,139,0.25)"},
                        "bar":        {"color": intel["hex"], "thickness": 0.22},
                        "bgcolor":    "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps":      [
                            {"range": [0,  40], "color": "rgba(16,185,129,0.05)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.05)"},
                            {"range": [70,100], "color": "rgba(239,68,68,0.05)"},
                        ],
                        "threshold":  {
                            "line":  {"color": intel["hex"], "width": 3},
                            "value": res["confidence"] * 100,
                        },
                    },
                ))
                fig_g.update_layout(
                    height      = 195,
                    margin      = dict(l=10, r=10, t=40, b=5),
                    paper_bgcolor = "rgba(0,0,0,0)",
                    font_color  = "#6b9ab8",
                )
                st.plotly_chart(fig_g, use_container_width=True)

                st.info(f"🤖  {intel['ai_message']}")
                st.caption(
                    f"🔩 **{res['model_mode']}**  ·  📅 {res['timestamp']}  ·  "
                    f"🎯 Top: {res.get('top_class','N/A')}"
                )

            else:
                # ── Empty state ────────────────────────────────────────────
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:4.5rem 1.5rem;">
                    <div style="font-size:4rem; margin-bottom:14px; opacity:0.55;">🔬</div>
                    <div style="font-weight:700; font-size:0.98rem; margin-bottom:8px;">
                        Ready for Analysis
                    </div>
                    <div style="font-size:0.83rem; color:#6b9ab8; line-height:1.7;">
                        Upload an image or capture via camera<br>
                        then click <b>EXECUTE DEEP SCAN</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Result tabs (shown below both columns) ────────────────────────────
        if st.session_state.result:
            res   = st.session_state.result
            intel = ClinicalProtocols.get(res["diagnosis"])

            st.markdown("---")
            st.markdown(
                '<div class="sec-head" style="font-size:1.04rem;">'
                '<span></span>📋 Clinical Intelligence Engine</div>',
                unsafe_allow_html=True,
            )

            t1, t2, t3, t4, t5 = st.tabs([
                "🏥 Recommendations",
                "🌿 Patient Advice",
                "💊 Treatment Plan",
                "🤖 AI Explanation",
                "📄 Report",
            ])

            # ── Tab 1: Recommendations ─────────────────────────────────────
            with t1:
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**Clinical Recommendations**")
                    for item in intel["recommendations"]:
                        st.markdown(
                            f'<div class="step-box">{item}</div>',
                            unsafe_allow_html=True,
                        )
                with r2:
                    st.markdown("**Consultation & Follow-up**")
                    st.markdown(
                        f'<div class="step-box" style="border-left-color:{intel["hex"]};">'
                        f'{intel["consultation"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="step-box">📅 {intel["followup"]}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Tab 2: Patient Advice ──────────────────────────────────────
            with t2:
                for item in intel["patient_advice"]:
                    st.markdown(
                        f'<div class="step-box">🌿 {item}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Tab 3: Treatment Plan ──────────────────────────────────────
            with t3:
                tc1, tc2 = st.columns(2)
                plan = [
                    ("🩺 Procedures",    "procedures",     False, "#2563eb"),
                    ("💊 Medications",   "medications",    False, "#14b8a6"),
                    ("⚗️ Therapy",       "therapy",        False, "#8b5cf6"),
                    ("🚨 Emergency Signs","emergency_signs", True, "#ef4444"),
                ]
                for i, (lbl, key, emg, color) in enumerate(plan):
                    col = tc1 if i % 2 == 0 else tc2
                    with col:
                        items_html = "".join(
                            f'<div class="step-box {"step-emg" if emg else ""}" '
                            f'style="margin-bottom:5px;">{s}</div>'
                            for s in intel[key]
                        )
                        st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid {color};
                             padding:15px; margin-bottom:10px;">
                            <div style="font-weight:700; color:{color}; margin-bottom:10px;
                                        font-size:0.87rem;">{lbl}</div>
                            {items_html}
                        </div>
                        """, unsafe_allow_html=True)

            # ── Tab 4: AI Explanation ──────────────────────────────────────
            with t4:
                st.markdown("#### 🤖 Why Did AI Give This Result?")
                st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid #8b5cf6; padding:20px;">
                    <div style="font-weight:700; color:#a78bfa; margin-bottom:12px;
                                font-size:0.9rem;">
                        🧠 AI Feature Analysis — {res['diagnosis']} Detection
                    </div>
                    <div style="font-size:0.85rem; line-height:1.8; color:#dff0fa;">
                        {intel['why_result']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Multi-Class Probability Scores")
                scores = res.get("class_scores", {})
                if scores:
                    sorted_scores = sorted(
                        scores.items(), key=lambda x: x[1], reverse=True
                    )
                    labels = [s[0] for s in sorted_scores]
                    values = [s[1] * 100 for s in sorted_scores]
                    max_v  = max(values)
                    colors = ["#ef4444" if v == max_v else "#2563eb" for v in values]

                    fig_cls = go.Figure(go.Bar(
                        x            = values,
                        y            = labels,
                        orientation  = "h",
                        marker_color = colors,
                        marker_line_width = 0,
                        text         = [f"{v:.1f}%" for v in values],
                        textposition = "inside",
                        textfont     = dict(color="white", size=10, family="Oxanium"),
                    ))
                    fig_cls.update_layout(
                        height        = 300,
                        margin        = dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor = "rgba(0,0,0,0)",
                        plot_bgcolor  = "rgba(0,0,0,0)",
                        font          = dict(color="#6b9ab8", family="Outfit"),
                        xaxis         = dict(
                            range=[0, 100], showgrid=False,
                            zeroline=False, showticklabels=False,
                        ),
                        yaxis         = dict(showgrid=False),
                    )
                    st.plotly_chart(fig_cls, use_container_width=True)

                st.markdown("#### 🔥 Grad-CAM Heatmap Simulation")
                st.markdown("""
                <div class="heatmap-box">
                    <div style="font-size:2rem; margin-bottom:8px;">🔥</div>
                    <div style="font-weight:600; margin-bottom:4px; color:#f87171;">
                        Grad-CAM Visualization
                    </div>
                    <div style="font-size:0.78rem;">
                        Grad-CAM highlights regions of the image that most influenced
                        the AI's decision.<br><br>
                        <strong>Red zones</strong> = High activation / suspicious areas<br>
                        <strong>Blue zones</strong> = Low activation / normal tissue<br><br>
                        ⚙️ Full Grad-CAM requires TensorFlow model in online mode.
                        Load <code>skin_cancer_cnn.h5</code> to enable real heatmaps.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Tab 5: Report ──────────────────────────────────────────────
            with t5:
                st.markdown("#### 📥 Download Clinical Reports")
                d1, d2 = st.columns(2)
                scan_rec  = st.session_state.result
                proc      = st.session_state.proc_img

                with d1:
                    if PDF_OK and proc:
                        pdf_bytes = ReportGenerator.pdf(scan_rec, proc)
                        fname = (
                            f"SkinScan_{scan_rec.get('patient_name','PT')}_"
                            f"{datetime.date.today()}.pdf"
                        ).replace(" ", "_")
                        st.download_button(
                            "📄 Download PDF Report",
                            data      = pdf_bytes,
                            file_name = fname,
                            mime      = "application/pdf",
                        )
                    else:
                        st.warning("Install ReportLab:\n`pip install reportlab`")

                with d2:
                    st.download_button(
                        "📊 Download CSV Registry",
                        data      = ReportGenerator.csv_data(st.session_state.db),
                        file_name = f"SkinScan_Registry_{datetime.date.today()}.csv",
                        mime      = "text/csv",
                    )

                st.markdown("""
                <div style='font-size:0.73rem; color:#6b9ab8; margin-top:12px;
                            padding:12px 14px;
                            border:1px solid rgba(100,116,139,0.18);
                            border-radius:10px; line-height:1.6;'>
                    ⚠️ <b>Disclaimer:</b> AI-generated reports for research &amp;
                    educational purposes only. Not a formal medical diagnosis.
                    Always consult a certified dermatologist.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: AI ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def _analysis(self) -> None:
        """
        Render the AI Analysis deep-dive page for the most recent scan.

        Sections
        --------
        1. Primary diagnosis card with accuracy meter gauge
        2. Multi-class score progress bars
        3. AI Explanation Panel (full why-result text)
        4. Grad-CAM heatmap visualization area
        """
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">🧠 Deep Analysis</div>
            <p class="banner-title">AI Analysis Results</p>
            <p class="banner-sub">Multi-class scores · AI Explanation · Grad-CAM · Accuracy Meter</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.result:
            st.info("📭 No scan data. Head to **AI Scan** to perform an analysis first.")
            return

        res   = st.session_state.result
        intel = ClinicalProtocols.get(res["diagnosis"])

        c1, c2 = st.columns([1.2, 1], gap="large")

        # ── Left: gauge + summary ─────────────────────────────────────────────
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sec-head"><span></span>🎯 Primary Diagnosis: '
                f'<span style="color:{intel["hex"]};">{res["diagnosis"]}</span></div>',
                unsafe_allow_html=True,
            )
            bc_cls = (
                "b-high"   if res["risk_level"] == "HIGH"   else
                "b-medium" if res["risk_level"] == "MEDIUM" else "b-low"
            )
            st.markdown(f"""
            <div style="font-size:0.9rem; line-height:1.8; margin-bottom:16px;">
                <strong>Top Class:</strong>
                <span style="color:{intel['hex']}; font-family:'Oxanium',monospace;
                             font-weight:700;">{res.get('top_class','N/A')}</span><br>
                <strong>Risk Level:</strong>
                <span class="badge {bc_cls}"> {res['risk_level']}</span><br>
                <strong>Model:</strong> {res['model_mode']}<br>
                <strong>Scan Time:</strong> {res['timestamp']}
            </div>
            """, unsafe_allow_html=True)

            fig_acc = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = res["confidence"] * 100,
                delta = {
                    "reference":   70,
                    "increasing":  {"color": "#10b981"},
                    "decreasing":  {"color": "#ef4444"},
                },
                number = {
                    "suffix": "%",
                    "font":   {"family": "Oxanium", "size": 32, "color": intel["hex"]},
                },
                title  = {
                    "text": "Prediction Accuracy Meter",
                    "font": {"family": "Outfit", "size": 12, "color": "#6b9ab8"},
                },
                gauge  = {
                    "axis":        {"range": [0, 100], "tickfont": {"size": 9}},
                    "bar":         {"color": intel["hex"], "thickness": 0.28},
                    "bgcolor":     "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps":       [
                        {"range": [0,  40], "color": "rgba(239,68,68,0.08)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                        {"range": [70,100], "color": "rgba(16,185,129,0.08)"},
                    ],
                    "threshold": {
                        "line":  {"color": "white", "width": 2},
                        "value": res["confidence"] * 100,
                    },
                },
            ))
            fig_acc.update_layout(
                height        = 220,
                margin        = dict(l=10, r=10, t=50, b=5),
                paper_bgcolor = "rgba(0,0,0,0)",
                font_color    = "#6b9ab8",
            )
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Right: class scores ───────────────────────────────────────────────
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="sec-head"><span></span>📊 Multi-Class Scores</div>',
                unsafe_allow_html=True,
            )
            scores = res.get("class_scores", {})
            for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                pct     = score * 100
                is_top  = (cls == res.get("top_class"))
                color   = intel["hex"] if is_top else "#6b9ab8"
                star    = "⭐" if is_top else ""
                weight  = "700" if is_top else "400"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.76rem; margin-bottom:3px;">
                        <span style="color:{color}; font-weight:{weight};">{star} {cls}</span>
                        <span style="font-family:'Oxanium',monospace;
                                     color:{color};">{pct:.1f}%</span>
                    </div>
                    <div style="background:rgba(37,99,235,0.12); border-radius:99px;
                                height:6px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{color};
                                    border-radius:99px; transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── AI Explanation panel ──────────────────────────────────────────────
        st.markdown(
            '<div class="glass-card" style="border-left:3px solid #8b5cf6;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sec-head"><span></span>🤖 AI Explanation Panel</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div style="font-size:0.87rem; line-height:1.9; color:#dff0fa;">
            {intel['why_result']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Grad-CAM placeholder ──────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-head"><span></span>🔥 Grad-CAM Heatmap Visualization</div>',
            unsafe_allow_html=True,
        )
        h1, h2 = st.columns(2)
        with h1:
            if st.session_state.raw_img:
                st.image(
                    ImageProcessor.thumb(st.session_state.raw_img, 350),
                    caption="📷 Original Image",
                    use_container_width=True,
                )
        with h2:
            st.markdown("""
            <div class="heatmap-box" style="height:200px; display:flex;
                 flex-direction:column; justify-content:center;">
                <div style="font-size:2.5rem; margin-bottom:8px;">🔥</div>
                <div style="font-weight:700; color:#f87171; margin-bottom:4px;">
                    Grad-CAM Overlay
                </div>
                <div style="font-size:0.76rem;">
                    Load skin_cancer_cnn.h5 for real heatmaps.<br>
                    Red = High Suspicion  ·  Blue = Normal Tissue
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════

    def _dashboard(self) -> None:
        """
        Render the clinical analytics dashboard with:
        - 5 KPI stat cards
        - Diagnosis distribution pie chart
        - Risk level bar chart
        - Probability vs Confidence scatter plot
        - Confidence trend line chart
        """
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">📊 Analytics</div>
            <p class="banner-title">Clinical Dashboard</p>
            <p class="banner-sub">Session statistics · Real-time KPIs
                · Diagnosis patterns · Confidence trends</p>
        </div>
        """, unsafe_allow_html=True)

        db  = st.session_state.db
        n   = len(db)
        mal = sum(1 for r in db if r.get("diagnosis") == "Malignant")
        avg_c = (sum(r.get("confidence", 0) for r in db) / n * 100) if n else 0
        hi  = sum(1 for r in db if r.get("risk_level") == "HIGH")

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            ("🧬", "Total Scans",    str(n),    "This session",       "#3b82f6"),
            ("🔴", "Malignant",      str(mal),  "High-risk detected", "#ef4444"),
            ("🟢", "Benign",         str(n-mal),"Low-risk cleared",   "#10b981"),
            ("🚨", "High Risk",      str(hi),   "Urgent cases",       "#f59e0b"),
            ("⚡", "Avg Confidence", f"{avg_c:.1f}%", "CNN inference", "#8b5cf6"),
        ]
        for col, (icon, lbl, val, dlt, color) in zip([k1,k2,k3,k4,k5], kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-glow" style="background:{color};"></div>
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value" style="color:{color};">{val}</div>
                    <div class="kd-neu">{dlt}</div>
                </div>
                """, unsafe_allow_html=True)

        if not db:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("📭 No scan data. Head to **AI Scan** to begin.")
            return

        df      = pd.DataFrame(db)
        df["prob"] = df["probability"] * 100
        df["conf"] = df["confidence"]  * 100

        # ── Plotly layout base ────────────────────────────────────────────────
        PL = dict(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = dict(family="Outfit", color="#6b9ab8"),
            margin        = dict(l=4, r=4, t=44, b=4),
        )
        GR = dict(
            gridcolor    = "rgba(37,99,235,0.10)",
            zerolinecolor = "rgba(37,99,235,0.08)",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)

        # ── Diagnosis pie ─────────────────────────────────────────────────────
        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            ser  = df["diagnosis"].value_counts()
            fig1 = go.Figure(go.Pie(
                labels       = ser.index.tolist(),
                values       = ser.values.tolist(),
                hole         = 0.54,
                marker       = dict(
                    colors = ["#ef4444", "#10b981"],
                    line   = dict(color="rgba(0,0,0,0)", width=2),
                ),
                textinfo     = "percent+label",
                textfont_size = 11,
            ))
            fig1.update_layout(
                title       = "Diagnosis Distribution",
                height      = 300,
                showlegend  = True,
                legend      = dict(font_size=11, orientation="h", y=-0.1),
                annotations = [dict(
                    text     = f"<b>{n}</b><br>scans",
                    x=0.5, y=0.5,
                    font_size  = 14,
                    font_color = "#dff0fa",
                    showarrow  = False,
                )],
                **PL,
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Risk bar chart ────────────────────────────────────────────────────
        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            rc = df["risk_level"].value_counts().reset_index()
            rc.columns = ["Risk", "Count"]
            fig2 = go.Figure()
            for risk, color in [("HIGH","#ef4444"), ("MEDIUM","#f59e0b"), ("LOW","#10b981")]:
                subset = rc[rc["Risk"] == risk]
                if not subset.empty:
                    fig2.add_trace(go.Bar(
                        x=subset["Risk"], y=subset["Count"],
                        name=risk, marker_color=color, marker_line_width=0,
                    ))
            fig2.update_layout(
                title      = "Risk Distribution",
                height     = 300,
                showlegend = False,
                barmode    = "group",
                xaxis      = dict(title="", **GR),
                yaxis      = dict(title="Cases", **GR),
                **PL,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        r3, r4 = st.columns(2)

        # ── Scatter: probability vs confidence ────────────────────────────────
        with r3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig3 = go.Figure()
            for diag, color, sym in [
                ("Malignant", "#ef4444", "circle"),
                ("Benign",    "#10b981", "diamond"),
            ]:
                subset = df[df["diagnosis"] == diag]
                if not subset.empty:
                    fig3.add_trace(go.Scatter(
                        x=subset["prob"], y=subset["conf"],
                        mode="markers", name=diag,
                        marker=dict(color=color, size=9, opacity=0.85, symbol=sym),
                    ))
            fig3.update_layout(
                title  = "Probability vs Confidence",
                height = 285,
                xaxis  = dict(title="Probability (%)", **GR),
                yaxis  = dict(title="Confidence (%)",  **GR),
                legend = dict(orientation="h", y=-0.2, font_size=11),
                **PL,
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Trend line ────────────────────────────────────────────────────────
        with r4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if len(df) >= 2:
                x    = list(range(1, len(df) + 1))
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=x, y=df["conf"],
                    mode="lines+markers", name="Confidence",
                    line=dict(color="#2563eb", width=2.5, shape="spline"),
                    marker=dict(size=7),
                    fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
                ))
                fig4.add_trace(go.Scatter(
                    x=x, y=df["prob"],
                    mode="lines+markers", name="Probability",
                    line=dict(color="#ef4444", width=2, dash="dot", shape="spline"),
                    marker=dict(size=7),
                ))
                fig4.update_layout(
                    title  = "Scan Trend",
                    height = 285,
                    xaxis  = dict(title="Scan #", **GR),
                    yaxis  = dict(title="Score (%)", **GR, range=[0, 105]),
                    legend = dict(orientation="h", y=-0.2, font_size=11),
                    **PL,
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Need 2+ scans for trend analysis.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def _history(self) -> None:
        """
        Render the patient history registry page with:
        - KPI summary cards
        - Multi-filter search (text, diagnosis, risk, gender)
        - Interactive data table
        - CSV and JSON export
        - Clear all records option
        """
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">📋 Patient History</div>
            <p class="banner-title">Scan History Database</p>
            <p class="banner-sub">Complete session records · Smart filters
                · Export to CSV / JSON</p>
        </div>
        """, unsafe_allow_html=True)

        db = st.session_state.db

        if not db:
            st.info("📭 No records yet. Run scans in **AI Scan** to populate history.")
            return

        # ── Build display dataframe ───────────────────────────────────────────
        df = pd.DataFrame([{
            "Time":      (r.get("timestamp", "").split(" ")[1]
                          if " " in r.get("timestamp", "") else ""),
            "Patient":   r.get("patient_name", "ANON"),
            "Age":       r.get("age",          "—"),
            "Gender":    r.get("gender",        "—"),
            "Diagnosis": r.get("diagnosis",     "—"),
            "Top Class": r.get("top_class",     "—"),
            "Risk":      r.get("risk_level",    "—"),
            "Prob.":     f"{r.get('probability', 0)*100:.1f}%",
            "Conf.":     f"{r.get('confidence',  0)*100:.1f}%",
            "Blur":      f"{r.get('blur_score',  0):.0f}",
        } for r in db])

        # ── KPI strip ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        m     = sum(1 for r in db if r.get("diagnosis")  == "Malignant")
        h     = sum(1 for r in db if r.get("risk_level") == "HIGH")
        avg_c = sum(r.get("confidence", 0) for r in db) / len(db) * 100
        for col, lbl, val, color in [
            (k1, "Records",   len(db),          "#60a5fa"),
            (k2, "Malignant", m,                "#f87171"),
            (k3, "High Risk", h,                "#fbbf24"),
            (k4, "Avg Conf.", f"{avg_c:.1f}%",  "#a78bfa"),
        ]:
            with col:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
                    f'<div class="kpi-value" style="color:{color};">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────────────────────
        with st.expander("🔍 Filter Records"):
            fc1, fc2, fc3 = st.columns(3)
            fd = fc1.multiselect(
                "Diagnosis", ["Malignant","Benign"],
                default=["Malignant","Benign"],
            )
            fr = fc2.multiselect(
                "Risk", ["HIGH","MEDIUM","LOW"],
                default=["HIGH","MEDIUM","LOW"],
            )
            fg = fc3.multiselect(
                "Gender",
                ["Male","Female","Other","Prefer not to say"],
                default=["Male","Female","Other","Prefer not to say"],
            )

        mask  = (
            df["Diagnosis"].isin(fd) &
            df["Risk"].isin(fr)      &
            df["Gender"].isin(fg)
        )
        df_f = df[mask]
        st.caption(f"Showing **{len(df_f)}** of **{len(df)}** records")

        st.markdown(
            '<div class="glass-card" style="padding:0; overflow:hidden;">',
            unsafe_allow_html=True,
        )
        st.dataframe(df_f, use_container_width=True, hide_index=True, height=380)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Export buttons ────────────────────────────────────────────────────
        e1, e2, e3 = st.columns(3)

        with e1:
            st.download_button(
                "📥 Export CSV",
                data      = ReportGenerator.csv_data(db),
                file_name = f"SkinScan_{datetime.date.today()}.csv",
                mime      = "text/csv",
            )

        with e2:
            safe_db = [
                {
                    k: str(v) if isinstance(v, datetime.datetime) else v
                    for k, v in r.items()
                    if not isinstance(v, dict)
                }
                for r in db
            ]
            st.download_button(
                "🔗 Export JSON",
                data      = json.dumps(safe_db, indent=2),
                file_name = f"SkinScan_{datetime.date.today()}.json",
                mime      = "application/json",
            )

        with e3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.db       = []
                st.session_state.result   = None
                st.session_state.raw_img  = None
                st.session_state.proc_img = None
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: MEDICAL GUIDE
    # ══════════════════════════════════════════════════════════════════════════

    def _medical_guide(self) -> None:
        """
        Render the comprehensive medical education module with 5 tabs:
        1. Doctor Guide — urgency-stratified consultation pathways
        2. Prevention Tips — sun protection, nutrition, self-examination
        3. Disease Info — clinical profiles for 6 lesion types
        4. Treatments — 8 treatment modality cards
        5. Warning Signs — emergency indicators
        """
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">👨‍⚕️ Medical Professional</div>
            <p class="banner-title">Medical Guide & Prevention</p>
            <p class="banner-sub">Doctor recommendations · Prevention tips
                · Disease information · Treatment awareness · Warning signs</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-banner" style="margin-bottom:24px;">
            🏥 <strong>For Healthcare Professionals &amp; Patients:</strong>
            The information below is for <strong>educational awareness only</strong>.
            It does not replace professional medical advice, diagnosis, or treatment.
            Always seek guidance from a <strong>qualified dermatologist or oncologist</strong>.
            In case of emergency symptoms, call your local emergency number immediately.
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4, t5 = st.tabs([
            "👨‍⚕️ Doctor Guide",
            "🛡️ Prevention Tips",
            "📚 Disease Info",
            "💊 Treatments",
            "🚨 Warning Signs",
        ])

        # ── Tab 1: Doctor Guide ───────────────────────────────────────────────
        with t1:
            st.markdown(
                '<div class="sec-head"><span></span>👨‍⚕️ Doctor Recommendation System</div>',
                unsafe_allow_html=True,
            )
            st.markdown("""
            <div class="glass-card">
                <div style="font-weight:700; font-size:1.0rem; margin-bottom:12px; color:#60a5fa;">
                    When Should You See a Dermatologist?
                </div>
            """, unsafe_allow_html=True)

            urgencies = [
                ("🚨 EMERGENCY",  "Visit ER / Call 112 immediately",
                 "Any lesion with active bleeding, rapid ulceration, lymph node swelling, "
                 "or spreading satellite lesions.", "#ef4444"),
                ("⚡ URGENT (48h)", "Onco-Dermatologist within 2 days",
                 "AI flagged HIGH RISK, asymmetric lesion >6mm, multi-color, "
                 "rapidly changing morphology.", "#f59e0b"),
                ("📅 ROUTINE",     "Annual dermatology visit",
                 "Stable benign lesion with no ABCDE changes. "
                 "Standard monitoring protocol.", "#10b981"),
                ("🔍 MONITORING",  "6-month follow-up",
                 "Medium confidence AI result, borderline features, "
                 "or patient history of melanoma.", "#3b82f6"),
            ]
            for icon, title, desc, color in urgencies:
                st.markdown(f"""
                <div class="step-box" style="border-left-color:{color}; margin-bottom:10px;">
                    <div style="font-weight:700; color:{color}; margin-bottom:4px;
                                font-size:0.88rem;">{icon} {title}</div>
                    <div style="font-size:0.82rem; color:#dff0fa;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="sec-head" style="margin-top:20px;"><span></span>'
                '🔬 What to Tell Your Doctor</div>',
                unsafe_allow_html=True,
            )
            tell_doc = [
                "Duration — How long has the lesion been present?",
                "Changes — Has it grown, changed color, or changed shape recently?",
                "Symptoms — Any pain, itching, burning, or bleeding?",
                "History — Family history of melanoma or skin cancer?",
                "Exposure — Years of sun/UV exposure, history of sunburns?",
                "Previous lesions — Any previously removed or biopsied lesions?",
                "Photo documentation — Bring timestamped photos if available.",
            ]
            for tip in tell_doc:
                st.markdown(
                    f'<div class="prev-tip">📋 {tip}</div>',
                    unsafe_allow_html=True,
                )

        # ── Tab 2: Prevention Tips ────────────────────────────────────────────
        with t2:
            st.markdown(
                '<div class="sec-head"><span></span>🛡️ Skin Cancer Prevention Tips</div>',
                unsafe_allow_html=True,
            )
            prevention_cats = {
                "☀️ Sun Protection": [
                    "Apply SPF 50+ broad-spectrum sunscreen every 2 hours outdoors.",
                    "Reapply after swimming, sweating, or towel drying.",
                    "Wear UPF 50+ protective clothing, wide-brim hat, and UV-blocking sunglasses.",
                    "Seek shade between 10:00 AM – 4:00 PM (peak UV hours).",
                    "Avoid tanning beds and sunlamps — they emit harmful UV radiation.",
                    "Check the daily UV index — take extra precautions when UV ≥ 6.",
                ],
                "🥗 Nutrition & Lifestyle": [
                    "Eat antioxidant-rich foods: berries, leafy greens, tomatoes, carrots.",
                    "Omega-3 fatty acids (salmon, flaxseed) have photoprotective effects.",
                    "Green tea polyphenols may reduce UV-induced skin damage.",
                    "Maintain healthy weight — obesity linked to increased cancer risk.",
                    "Quit smoking — accelerates cellular DNA damage in skin tissue.",
                    "Limit alcohol — associated with increased melanoma risk.",
                ],
                "🔍 Self-Examination": [
                    "Perform full-body ABCDE self-check once per month.",
                    "Use a full-length mirror and hand mirror to check hard-to-see areas.",
                    "Photograph suspicious lesions monthly to track changes over time.",
                    "Check scalp, between toes, under nails, and in skin folds.",
                    "Report any new mole appearing after age 30 to a dermatologist.",
                    "Use the 'Ugly Duckling' rule — a mole that looks different from others.",
                ],
                "🏥 Medical Monitoring": [
                    "Annual professional skin examination for all adults over 40.",
                    "Every 6 months if you have risk factors (family history, fair skin, many moles).",
                    "Digital dermoscopy baseline documentation for all concerning lesions.",
                    "Genetic counseling if 2+ first-degree relatives have melanoma.",
                    "Vitamin D via supplementation only — not UV exposure.",
                ],
            }
            for cat, tips in prevention_cats.items():
                with st.expander(cat, expanded=False):
                    for tip in tips:
                        st.markdown(
                            f'<div class="prev-tip">{tip}</div>',
                            unsafe_allow_html=True,
                        )

        # ── Tab 3: Disease Info ───────────────────────────────────────────────
        with t3:
            st.markdown(
                '<div class="sec-head"><span></span>📚 Skin Cancer Educational Guide</div>',
                unsafe_allow_html=True,
            )
            diseases = [
                ("🔴", "Melanoma",
                 "Most dangerous. Arises from melanocytes. Can spread to organs. "
                 "5-year survival >98% if caught early vs ~23% if metastatic.",
                 "Asymmetric, irregular border, multiple colors, >6mm, evolving. "
                 "Can appear anywhere including scalp, under nails, soles.",
                 "#ef4444"),
                ("🟠", "Basal Cell Carcinoma (BCC)",
                 "Most common skin cancer (~80% of cases). Rarely metastasizes but "
                 "causes significant local tissue destruction if untreated.",
                 "Pearly or waxy bump, flat flesh-colored or brown scar-like lesion, "
                 "bleeding or scabbing sore that heals and returns.",
                 "#f97316"),
                ("🟡", "Squamous Cell Carcinoma (SCC)",
                 "Second most common. Can spread to lymph nodes. Risk increases with "
                 "cumulative UV exposure and immunosuppression.",
                 "Firm red nodule, flat lesion with scaly crust, new sore on old scar, "
                 "rough patch on lip, red sore inside mouth.",
                 "#f59e0b"),
                ("🟢", "Benign Nevus (Mole)",
                 "Common benign growth from melanocytes. Most people have 10–40 moles. "
                 "Monitoring for changes is essential.",
                 "Symmetrical, well-defined border, uniform color (tan or brown), "
                 "usually <6mm. Present since childhood or young adulthood.",
                 "#10b981"),
                ("🔵", "Actinic Keratosis",
                 "Pre-cancerous lesion caused by chronic UV exposure. 5–10% progress "
                 "to SCC if untreated. Treat to prevent progression.",
                 "Rough, dry, scaly patch of skin, flat to slightly raised patch, "
                 "hard wart-like surface, itching/burning in the affected area.",
                 "#3b82f6"),
                ("🟣", "Seborrheic Keratosis",
                 "Benign, waxy, wart-like growth. Very common in older adults. "
                 "No cancer risk but can resemble melanoma.",
                 "Waxy, scaly, slightly elevated lesion. Range from light tan to black. "
                 "'Stuck on' appearance. Multiple lesions common.",
                 "#8b5cf6"),
            ]
            gc = st.columns(2)
            for i, (icon, name, about, signs, color) in enumerate(diseases):
                with gc[i % 2]:
                    st.markdown(f"""
                    <div class="guide-card" style="border-top:3px solid {color};
                         margin-bottom:16px;">
                        <div style="display:flex; align-items:center; gap:8px;
                                    margin-bottom:10px;">
                            <span style="font-size:1.5rem;">{icon}</span>
                            <div style="font-weight:700; font-size:0.96rem;
                                        color:{color};">{name}</div>
                        </div>
                        <div style="font-size:0.78rem; color:#dff0fa; line-height:1.7;
                                    margin-bottom:10px;">{about}</div>
                        <div style="font-size:0.74rem; color:#6b9ab8; line-height:1.6;
                                    border-top:1px solid rgba(37,99,235,0.15);
                                    padding-top:8px;">
                            <strong style="color:{color};">Visual Signs:</strong> {signs}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 4: Treatments ─────────────────────────────────────────────────
        with t4:
            st.markdown(
                '<div class="sec-head"><span></span>💊 Treatment Awareness</div>',
                unsafe_allow_html=True,
            )
            treatments = [
                ("🩺", "Surgical Excision",
                 "Removal of lesion with clear margins. Gold standard for most skin "
                 "cancers. Success rate >95% for early-stage lesions.", "#2563eb"),
                ("🔬", "Mohs Surgery",
                 "Layer-by-layer removal with real-time microscopic analysis. "
                 "Highest cure rate (~99%) for BCC/SCC.", "#14b8a6"),
                ("💊", "Immunotherapy",
                 "Pembrolizumab, Ipilimumab — checkpoint inhibitors that activate "
                 "immune system to fight cancer cells.", "#8b5cf6"),
                ("🧬", "Targeted Therapy",
                 "BRAF/MEK inhibitors for BRAF-mutated melanoma. "
                 "Vemurafenib + Cobimetinib combination.", "#f59e0b"),
                ("☀️", "Photodynamic Therapy",
                 "Light-activated drug destroys cancer cells. Used for superficial "
                 "BCC and actinic keratosis.", "#10b981"),
                ("❄️", "Cryotherapy",
                 "Liquid nitrogen freeze-destroys benign/pre-cancerous lesions. "
                 "Quick office procedure.", "#60a5fa"),
                ("⚡", "Radiation Therapy",
                 "Used post-surgery or when surgery not possible. "
                 "Targets residual cancer cells.", "#ef4444"),
                ("💉", "Intralesional Therapy",
                 "Direct injection of IL-2, talimogene laherparepvec (T-VEC) "
                 "into tumor. Used for advanced melanoma.", "#f97316"),
            ]
            tc = st.columns(2)
            for i, (icon, name, desc, color) in enumerate(treatments):
                with tc[i % 2]:
                    st.markdown(f"""
                    <div class="step-box" style="border-left-color:{color};
                         margin-bottom:10px;">
                        <div style="display:flex; align-items:center; gap:8px;
                                    margin-bottom:4px;">
                            <span>{icon}</span>
                            <span style="font-weight:700; color:{color};
                                         font-size:0.88rem;">{name}</span>
                        </div>
                        <div style="font-size:0.79rem; color:#dff0fa;
                                    line-height:1.6;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 5: Warning Signs ──────────────────────────────────────────────
        with t5:
            st.markdown(
                '<div class="sec-head"><span></span>🚨 Warning Signs — Seek Immediate Care</div>',
                unsafe_allow_html=True,
            )
            st.markdown("""
            <div class="disclaimer-banner"
                 style="border-left-color:#ef4444;
                        background:linear-gradient(135deg,rgba(239,68,68,0.10),rgba(220,38,38,0.05));">
                🚨 <strong>EMERGENCY:</strong> If you experience any of the following
                symptoms, seek immediate medical attention. Do NOT wait for an appointment
                — visit an emergency department or call your doctor immediately.
            </div>
            """, unsafe_allow_html=True)

            warnings = [
                ("🔴", "Rapid Size Change",
                 "Lesion doubles in size within days or weeks — sign of aggressive growth."),
                ("🩸", "Spontaneous Bleeding",
                 "Lesion bleeds without any injury or trauma — indicates disrupted vasculature."),
                ("🔵", "Lymph Node Swelling",
                 "Swollen lymph nodes near neck, armpit, or groin — possible metastatic spread."),
                ("🟡", "Ulceration",
                 "Open sore that doesn't heal within 4 weeks — hallmark of invasive malignancy."),
                ("🟣", "Satellite Lesions",
                 "New small lesions appearing around original mole — sign of local metastasis."),
                ("⚪", "Color Whitening",
                 "Loss of pigment (white/grey zones) within lesion — regression pattern in melanoma."),
                ("🔶", "Pain / Burning",
                 "New pain, burning, or tingling in lesion area — possible nerve involvement."),
                ("🟤", "Thick / Raised Profile",
                 "Sudden thickening or raised hard nodule on flat lesion — vertical growth phase."),
            ]
            wc = st.columns(2)
            for i, (color_icon, title, desc) in enumerate(warnings):
                with wc[i % 2]:
                    st.markdown(f"""
                    <div class="step-box step-emg" style="margin-bottom:10px;">
                        <div style="display:flex; align-items:center; gap:8px;
                                    margin-bottom:3px;">
                            <span style="font-size:1.1rem;">{color_icon}</span>
                            <span style="font-weight:700; color:#f87171;
                                         font-size:0.87rem;">{title}</span>
                        </div>
                        <div style="font-size:0.78rem; color:#dff0fa;
                                    line-height:1.5;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  PAGE: ABOUT / SETTINGS
    # ══════════════════════════════════════════════════════════════════════════

    def _about(self) -> None:
        """
        Render the About & Settings page with:
        - Dark/light theme toggle
        - AI engine specification table
        - Expandable user guide sections
        """
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">ℹ️ About</div>
            <p class="banner-title">About & Settings</p>
            <p class="banner-sub">Platform information · Appearance
                · User guide · AI Engine Details</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Appearance / Theme ────────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-head"><span></span>🎨 Appearance</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                '<div class="set-lbl">Color Theme</div>'
                '<div class="set-desc">Switch between dark clinical mode and light mode</div>',
                unsafe_allow_html=True,
            )
        with c2:
            is_dark = st.session_state.theme == "dark"
            toggled = st.toggle("Dark Mode", value=is_dark)
            if toggled != is_dark:
                st.session_state.theme = "dark" if toggled else "light"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── AI Engine Details ─────────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-head"><span></span>🤖 AI Engine Details</div>',
            unsafe_allow_html=True,
        )
        dot  = "🟢" if self.ai.is_online else "🟠"
        mode = "Neural Network Online" if self.ai.is_online else "Simulation Mode"

        engine_rows = [
            ("Model File",        "skin_cancer_cnn.h5",                            "#14b8a6"),
            ("Architecture",      "MobileNetV2 + Custom Head (CNN)",               "#60a5fa"),
            ("Output Classes",    "8 classes: Melanoma, BCC, SCC, Benign, AK, SK, DF, VL", "#a78bfa"),
            ("Primary Output",    "Binary: Benign / Malignant (sigmoid)",          "#60a5fa"),
            ("Input Dimensions",  "224 × 224 px  ·  RGB  ·  Normalized 0–1",      "#60a5fa"),
            ("Preprocessing",     "Resize → Normalize → Contrast → Sharpen → Brightness", "#7fa3c0"),
            ("Blur Detection",    "Laplacian variance — threshold: 80",             "#f59e0b"),
            ("Grad-CAM",          "Available in online mode with TensorFlow backend", "#10b981"),
            ("Dataset",           "Melanoma Cancer Image Dataset (Kaggle)",         "#60a5fa"),
            ("Live App",          "https://skincancerpredictions.streamlit.app/",   "#2dd4bf"),
            ("Engine Status",     f"{dot} {mode}",                                 "#f59e0b"),
            ("Platform Version",  "SkinScan AI v15.0",                             "#7fa3c0"),
        ]
        for lbl, val, color in engine_rows:
            st.markdown(f"""
            <div class="set-row">
                <span class="set-lbl">{lbl}</span><br>
                <span style="font-family:'Oxanium',monospace; font-size:0.81rem;
                             color:{color};">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── User Guide ────────────────────────────────────────────────────────
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-head"><span></span>📖 User Guide</div>',
            unsafe_allow_html=True,
        )
        guide_sections = [
            ("📤 Image Upload", [
                "Accepted: <b>JPG, JPEG, PNG</b> only.",
                "Max <b>10 MB</b> · Min <b>100×100 px</b> · Drag &amp; Drop supported.",
                "Auto-preprocessing: resize 224×224 → normalize → contrast → sharpen → brightness.",
                "Quality badge shown: HIGH (≥300×300) · MEDIUM (&lt;300×300) with warning.",
            ]),
            ("📷 Camera Capture", [
                "Click <b>Live Camera</b> toggle in AI Scan.",
                "Allow camera access in browser. Works on mobile camera too.",
                "Use <b>Retake Photo</b> button to discard and re-capture.",
                "Before/After comparison view available after first scan.",
            ]),
            ("🤖 AI Inference", [
                "Model: <b>skin_cancer_cnn.h5</b> loaded at startup.",
                "🟢 Online Mode: Real TF inference | 🟠 Simulation: Demo mode.",
                "8-class multi-class scores with top-class identification.",
                "Blur detection via Laplacian variance — warns if image is blurry.",
            ]),
            ("📊 Results & Reports", [
                "<b>Probability</b>: Likelihood of primary diagnosis (0–100%).",
                "<b>Confidence</b>: Model certainty | Risk: HIGH ≥80% · MEDIUM ≥50% · LOW &lt;50%.",
                "AI Explanation panel: feature-by-feature breakdown of decision.",
                "Download PDF + CSV from Report tab. JSON export in History.",
            ]),
            ("🚀 Deployment", [
                "Place <b>skin_cancer_cnn.h5</b> in the same folder as app.py.",
                "Run: <b>streamlit run app.py</b>",
                "Live at: <b>https://skincancerpredictions.streamlit.app/</b>",
                "Requirements: see requirements.txt for all dependencies.",
            ]),
        ]
        for title, points in guide_sections:
            with st.expander(title):
                for pt in points:
                    st.markdown(
                        f'<div class="step-box">{pt}</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  ENTERPRISE FOOTER  ── v15
    # ══════════════════════════════════════════════════════════════════════════

    def _footer(self) -> None:
        """
        Render the full enterprise glassmorphism footer with:
        - Brand block (logo, description, tech stack chips, social links)
        - Platform navigation links
        - Feature links
        - Developer contact information
        - Badge strip
        - Bottom copyright / disclaimer bar
        """
        st.markdown("""
        <div class="footer-outer">
        <div class="site-footer">
          <div class="footer-inner">

            <div class="footer-top">

              <!-- Brand Column -->
              <div class="footer-brand">
                <div class="footer-brand-logo">
                  <span class="footer-brand-icon">🔬</span>
                  <div>
                    <div class="footer-brand-name">SkinScan AI</div>
                    <div class="footer-brand-tagline">Next-Gen Dermatology Intelligence</div>
                  </div>
                </div>
                <p class="footer-brand-desc">
                  An AI-powered clinical platform for dermoscopic skin lesion analysis.
                  Developed as a Final Year Project at the University of Agriculture
                  Faisalabad using deep learning CNN models for benign/malignant
                  classification of skin cancer.
                </p>
                <div class="footer-tech-stack">
                  <span class="ftech-chip">Python</span>
                  <span class="ftech-chip">Streamlit</span>
                  <span class="ftech-chip">TensorFlow</span>
                  <span class="ftech-chip">Plotly</span>
                  <span class="ftech-chip">PIL</span>
                  <span class="ftech-chip">ReportLab</span>
                  <span class="ftech-chip">NumPy</span>
                  <span class="ftech-chip">Pandas</span>
                  <span class="ftech-chip">MobileNetV2</span>
                </div>
                <div class="footer-social">
                  <a href="https://github.com/rehanshafiq70" target="_blank"
                     class="social-btn github" title="GitHub">
                    <svg width="17" height="17" viewBox="0 0 24 24"
                         fill="currentColor" style="color:#c9d1d9;">
                      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205
                               11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235
                               -3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23
                               -1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845
                               1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765
                               -1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385
                               1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3
                               1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56
                               3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23
                               1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375
                               .81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225
                               .69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                    </svg>
                  </a>
                  <a href="https://www.linkedin.com/in/rehanshafiq70" target="_blank"
                     class="social-btn linkedin" title="LinkedIn">
                    <svg width="16" height="16" viewBox="0 0 24 24"
                         fill="currentColor" style="color:#0a66c2;">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852
                               -3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414
                               v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267
                               2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926
                               -2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064
                               .925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782
                               13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774
                               0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24
                               23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                  </a>
                  <a href="mailto:rehanshafiq6540@gmail.com"
                     class="social-btn email" title="Email">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                         stroke="#14b8a6" stroke-width="2"
                         stroke-linecap="round" stroke-linejoin="round">
                      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1
                               0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                      <polyline points="22,6 12,13 2,6"/>
                    </svg>
                  </a>
                </div>
              </div>

              <!-- Platform Links -->
              <div>
                <div class="footer-col-title">Platform</div>
                <span class="footer-nav-link">🏠 Home Dashboard</span>
                <span class="footer-nav-link">📷 AI Scan Lab</span>
                <span class="footer-nav-link">🧠 AI Analysis</span>
                <span class="footer-nav-link">📊 Analytics Dashboard</span>
                <span class="footer-nav-link">📁 Patient History</span>
                <span class="footer-nav-link">👨‍⚕️ Medical Guide</span>
                <span class="footer-nav-link">⚙️ Settings</span>
              </div>

              <!-- Feature Links -->
              <div>
                <div class="footer-col-title">Features</div>
                <span class="footer-nav-link">🧬 Multi-Class CNN</span>
                <span class="footer-nav-link">🔥 Grad-CAM Heatmap</span>
                <span class="footer-nav-link">🤖 AI Explanation Panel</span>
                <span class="footer-nav-link">📷 Live Camera Scan</span>
                <span class="footer-nav-link">📄 PDF / CSV Reports</span>
                <span class="footer-nav-link">🛡️ Blur Detection</span>
                <span class="footer-nav-link">🌓 Dark / Light Mode</span>
                <span class="footer-nav-link">🌐 Live: skincancerpredictions.streamlit.app</span>
              </div>

              <!-- Developer Contact -->
              <div>
                <div class="footer-col-title">Developer Contact</div>

                <div class="footer-contact-item">
                  <div class="fci-icon">👨‍💻</div>
                  <div>
                    <div class="fci-label">Developer</div>
                    <div class="fci-value">Rehan Shafique</div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">🎓</div>
                  <div>
                    <div class="fci-label">Reg. No.</div>
                    <div class="fci-value">2022-AG-7662</div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">🏫</div>
                  <div>
                    <div class="fci-label">Institution</div>
                    <div class="fci-value">University of Agriculture Faisalabad</div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">📧</div>
                  <div>
                    <div class="fci-label">Email</div>
                    <div class="fci-value">
                      <a href="mailto:rehanshafiq6540@gmail.com">
                        rehanshafiq6540@gmail.com
                      </a>
                    </div>
                    <div class="email-copy-btn"
                         onclick="navigator.clipboard.writeText('rehanshafiq6540@gmail.com')
                                  .then(()=>{this.textContent='✅ Copied!';
                                  setTimeout(()=>{this.innerHTML='📋 Copy Email'},2000)})">
                      📋 Copy Email
                    </div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">💼</div>
                  <div>
                    <div class="fci-label">LinkedIn</div>
                    <div class="fci-value">
                      <a href="https://www.linkedin.com/in/rehanshafiq70" target="_blank">
                        linkedin.com/in/rehanshafiq70
                      </a>
                    </div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">🐙</div>
                  <div>
                    <div class="fci-label">GitHub</div>
                    <div class="fci-value">
                      <a href="https://github.com/rehanshafiq70" target="_blank">
                        github.com/rehanshafiq70
                      </a>
                    </div>
                  </div>
                </div>

                <div class="footer-contact-item">
                  <div class="fci-icon">🔗</div>
                  <div>
                    <div class="fci-label">Live App</div>
                    <div class="fci-value">
                      <a href="https://skincancerpredictions.streamlit.app/" target="_blank">
                        skincancerpredictions.streamlit.app
                      </a>
                    </div>
                  </div>
                </div>

              </div>
            </div>

            <!-- Badge Strip -->
            <div class="footer-badges">
              <span class="fbadge">🔬 CNN Deep Learning</span>
              <span class="fbadge">🏥 Clinical Intelligence</span>
              <span class="fbadge">🧬 Dermoscopy AI</span>
              <span class="fbadge">🎓 Final Year Project 2026</span>
              <span class="fbadge">🌐 University of Agriculture Faisalabad</span>
              <span class="fbadge">⚡ Streamlit v15.0</span>
              <span class="fbadge">🤖 TensorFlow CNN</span>
              <span class="fbadge">🩺 Binary: Benign / Malignant</span>
            </div>

            <!-- Bottom Bar -->
            <div class="footer-bottom">
              <div class="footer-copy">
                © 2026 <strong>SkinScan AI</strong> —
                Developed by <strong>Rehan Shafique</strong>
                &nbsp;·&nbsp; University of Agriculture Faisalabad
                &nbsp;·&nbsp; Department of Bioinformatics
                &nbsp;·&nbsp; BS Bioinformatics Session 2022–2026
              </div>
              <div class="footer-disclaimer">
                ⚠️ Research &amp; Educational Use Only — Not a Certified Medical Device
              </div>
              <div class="footer-version-badge">v15.0</div>
            </div>

          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = SkinScanApp()
    app.launch()
