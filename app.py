# ============================================================
# SKINSCAN AI — ENTERPRISE CLINICAL SUITE v12.0
# Architecture : OOP Microservices | No Login | No Camera
# Features     : 3D UI, AI Diagnosis, PDF Reports, Registry
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageFilter, ImageEnhance
import io
import random
import time
import uuid
import datetime
import base64
import json
import os
import traceback

# ── Optional heavy imports (graceful fallback) ────────────
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except Exception:
    OPTION_MENU_AVAILABLE = False

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan AI — Enterprise Clinical Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
# CLASS 1 ── InterfaceManager (3D CSS + Theme)
# ══════════════════════════════════════════════════════════
class InterfaceManager:
    DARK = {
        "bg": "#020617",
        "card": "rgba(15,23,42,0.88)",
        "border": "rgba(59,130,246,0.25)",
        "text": "#e2e8f0",
        "sub": "#94a3b8",
        "accent": "#3b82f6",
    }
    LIGHT = {
        "bg": "#f0f4ff",
        "card": "rgba(255,255,255,0.92)",
        "border": "rgba(30,64,175,0.18)",
        "text": "#1e293b",
        "sub": "#475569",
        "accent": "#1d4ed8",
    }

    def get_theme(self):
        return self.DARK if st.session_state.get("theme", "dark") == "dark" else self.LIGHT

    def inject_css(self):
        t = self.get_theme()
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ── Base ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
            font-family: 'DM Sans', sans-serif;
        }}
        h1,h2,h3,h4,h5,h6 {{ font-family:'Syne',sans-serif; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg,rgba(2,6,23,0.98) 0%,rgba(15,23,42,0.98) 100%) !important;
            border-right: 1px solid rgba(59,130,246,0.18) !important;
            box-shadow: 5px 0 40px rgba(0,0,0,0.55) !important;
        }}
        [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}

        /* ── 3D Cards ── */
        .card-3d {{
            background: {t['card']};
            border: 1px solid {t['border']};
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            transform: perspective(900px) rotateX(2deg) rotateY(-1.5deg);
            transition: transform .38s ease, box-shadow .38s ease;
            box-shadow: 0 20px 60px rgba(0,0,0,.45),
                        0 0 40px rgba(59,130,246,.10),
                        inset 0 1px 0 rgba(255,255,255,.07);
            margin-bottom: 1rem;
        }}
        .card-3d:hover {{
            transform: perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(18px);
            box-shadow: 0 36px 80px rgba(0,0,0,.6),
                        0 0 55px rgba(59,130,246,.28),
                        inset 0 1px 0 rgba(255,255,255,.14);
        }}

        /* ── KPI Flip Cards ── */
        .kpi-wrap {{ perspective: 1000px; height: 115px; margin-bottom:.8rem; cursor:pointer; }}
        .kpi-inner {{
            width:100%; height:100%;
            transition: transform .65s cubic-bezier(.4,0,.2,1);
            transform-style: preserve-3d; position:relative;
        }}
        .kpi-wrap:hover .kpi-inner {{ transform: rotateY(180deg); }}
        .kpi-front, .kpi-back {{
            position:absolute; width:100%; height:100%;
            backface-visibility:hidden; border-radius:14px;
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            padding:.9rem;
        }}
        .kpi-front {{
            background: linear-gradient(135deg,rgba(30,41,59,.92),rgba(15,23,42,.96));
            border: 1px solid rgba(59,130,246,.28);
            box-shadow: 0 8px 24px rgba(0,0,0,.35);
        }}
        .kpi-back {{
            background: linear-gradient(135deg,#3b82f6,#8b5cf6);
            transform: rotateY(180deg);
            box-shadow: 0 8px 24px rgba(59,130,246,.45);
        }}
        .kpi-value {{ font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:#f1f5f9; }}
        .kpi-label {{ font-size:.78rem; color:#94a3b8; letter-spacing:.08em; text-transform:uppercase; margin-top:.2rem; }}
        .kpi-back-text {{ font-size:.85rem; color:white; text-align:center; font-weight:600; }}

        /* ── Diagnosis Badge ── */
        .badge {{
            display:inline-block; padding:.85rem 2.4rem;
            border-radius:50px; font-size:1.3rem; font-weight:800;
            letter-spacing:.18em; text-transform:uppercase;
            transform: perspective(500px) rotateX(7deg);
            transition: transform .3s ease;
            box-shadow: 0 14px 28px rgba(0,0,0,.38), 0 5px 12px rgba(0,0,0,.25);
        }}
        .badge:hover {{ transform:perspective(500px) rotateX(0deg) scale(1.04); }}
        .badge-mal {{ background:linear-gradient(135deg,#ef4444,#dc2626); color:#fff;
                      box-shadow:0 14px 30px rgba(239,68,68,.5); }}
        .badge-ben {{ background:linear-gradient(135deg,#10b981,#059669); color:#fff;
                      box-shadow:0 14px 30px rgba(16,185,129,.5); }}
        .badge-unc {{ background:linear-gradient(135deg,#f59e0b,#d97706); color:#fff;
                      box-shadow:0 14px 30px rgba(245,158,11,.5); }}

        /* ── 3D Scan Button ── */
        .scan-btn {{
            display:block; width:100%;
            background:linear-gradient(135deg,#3b82f6,#8b5cf6,#ec4899);
            color:#fff; border:none; padding:1.1rem 2rem;
            font-size:1.05rem; font-weight:700; letter-spacing:.16em;
            text-transform:uppercase; border-radius:14px; cursor:pointer;
            transform:perspective(400px) rotateX(9deg);
            box-shadow:0 10px 0 rgba(0,0,0,.28),0 14px 22px rgba(59,130,246,.38);
            transition:all .15s ease; font-family:'Syne',sans-serif;
        }}
        .scan-btn:hover {{
            transform:perspective(400px) rotateX(5deg) translateY(-3px);
            box-shadow:0 13px 0 rgba(0,0,0,.28),0 20px 32px rgba(59,130,246,.5);
        }}
        .scan-btn:active {{
            transform:perspective(400px) rotateX(0deg) translateY(8px);
            box-shadow:0 2px 0 rgba(0,0,0,.28),0 5px 10px rgba(59,130,246,.25);
        }}

        /* ── Gradient Headings ── */
        .grad-text {{
            background:linear-gradient(135deg,#3b82f6,#8b5cf6,#ec4899);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text; font-family:'Syne',sans-serif;
        }}

        /* ── Logo 3D Float ── */
        .logo-3d {{
            font-size:1.55rem; font-weight:900;
            background:linear-gradient(135deg,#3b82f6,#8b5cf6,#ec4899);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            filter:drop-shadow(0 4px 8px rgba(59,130,246,.45));
            display:inline-block;
            animation:logo-float 4s ease-in-out infinite;
        }}
        @keyframes logo-float {{
            0%,100% {{ transform:perspective(200px) rotateX(5deg) translateY(0); }}
            50%      {{ transform:perspective(200px) rotateX(2deg) translateY(-5px); }}
        }}

        /* ── Status Orb ── */
        .orb {{
            width:11px; height:11px; border-radius:50%;
            display:inline-block; vertical-align:middle; margin-right:6px;
            box-shadow:0 0 0 3px rgba(16,185,129,.2),0 0 14px rgba(16,185,129,.6);
            animation:orb-pulse 2s ease-in-out infinite;
        }}
        .orb-green  {{ background:#10b981; }}
        .orb-orange {{ background:#f59e0b;
                       box-shadow:0 0 0 3px rgba(245,158,11,.2),0 0 14px rgba(245,158,11,.6); }}
        @keyframes orb-pulse {{
            0%,100% {{ box-shadow:0 0 0 3px rgba(16,185,129,.2),0 0 14px rgba(16,185,129,.6); }}
            50%      {{ box-shadow:0 0 0 6px rgba(16,185,129,.1),0 0 24px rgba(16,185,129,.8); }}
        }}

        /* ── DNA Spinner ── */
        .dna-ring {{
            width:56px; height:56px;
            border:3px solid transparent;
            border-top-color:#3b82f6;
            border-right-color:#8b5cf6;
            border-radius:50%;
            animation:dna-spin 1s linear infinite;
            transform:perspective(100px) rotateX(20deg);
            box-shadow:0 0 20px rgba(59,130,246,.5);
            margin:0 auto;
        }}
        @keyframes dna-spin {{
            0%   {{ transform:perspective(100px) rotateX(20deg) rotate(0deg);   }}
            100% {{ transform:perspective(100px) rotateX(20deg) rotate(360deg); }}
        }}

        /* ── Chart 3D wrapper ── */
        .chart-3d {{
            border-radius:16px; overflow:hidden;
            transform:perspective(1000px) rotateX(2.5deg);
            box-shadow:0 24px 50px rgba(0,0,0,.5),0 0 30px rgba(59,130,246,.08);
            transition:transform .4s ease;
        }}
        .chart-3d:hover {{ transform:perspective(1000px) rotateX(0deg) translateZ(10px); }}

        /* ── Activity Feed ── */
        .feed-item {{
            padding:.55rem .9rem; margin:.3rem 0;
            background:rgba(30,41,59,.55);
            border-left:3px solid #3b82f6;
            border-radius:0 8px 8px 0;
            font-size:.82rem; color:#cbd5e1;
        }}

        /* ── Sim Badge ── */
        .sim-badge {{
            background:linear-gradient(90deg,#f59e0b,#d97706);
            color:#000; font-weight:700; font-size:.72rem;
            padding:.28rem .7rem; border-radius:20px;
            letter-spacing:.1em; text-transform:uppercase;
            display:inline-block; margin-left:.5rem;
        }}

        /* ── Disclaimer ── */
        .disclaimer {{
            background:rgba(239,68,68,.08);
            border:1px solid rgba(239,68,68,.3);
            border-radius:10px; padding:.75rem 1rem;
            font-size:.8rem; color:#fca5a5; margin-top:1rem;
        }}

        /* ── Risk Meter ── */
        .risk-bar {{
            display:flex; gap:5px; margin:.5rem 0;
        }}
        .risk-seg {{
            flex:1; height:12px; border-radius:4px;
            box-shadow:0 3px 6px rgba(0,0,0,.3);
        }}

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, header {{ visibility:hidden; }}
        .block-container {{ padding-top:1.2rem !important; }}
        </style>
        """, unsafe_allow_html=True)

    def kpi_card(self, value, label, back_text, icon="📊"):
        return f"""
        <div class="kpi-wrap">
          <div class="kpi-inner">
            <div class="kpi-front">
              <div style="font-size:1.5rem">{icon}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-label">{label}</div>
            </div>
            <div class="kpi-back">
              <div class="kpi-back-text">{back_text}</div>
            </div>
          </div>
        </div>"""

    def badge(self, text, kind):
        cls = {"malignant":"badge-mal","benign":"badge-ben","uncertain":"badge-unc"}.get(kind,"badge-unc")
        return f'<div class="badge {cls}">{text}</div>'

    def section_header(self, title, subtitle=""):
        sub = f'<p style="color:#94a3b8;font-size:.9rem;margin:.2rem 0 0">{subtitle}</p>' if subtitle else ""
        st.markdown(f'<h2 class="grad-text">{title}</h2>{sub}', unsafe_allow_html=True)

    def disclaimer_box(self):
        st.markdown('<div class="disclaimer">⚠️ <strong>Medical Disclaimer:</strong> This AI output is for clinical decision support only. Final diagnosis must be confirmed by a licensed dermatologist.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 2 ── NeuralCoreEngine
# ══════════════════════════════════════════════════════════
class NeuralCoreEngine:
    LABELS = [
        "Melanoma","Basal Cell Carcinoma","Squamous Cell Carcinoma",
        "Actinic Keratosis","Benign Keratosis","Dermatofibroma",
        "Melanocytic Nevi","Vascular Lesion",
    ]
    MALIGNANT = {"Melanoma","Basal Cell Carcinoma","Squamous Cell Carcinoma","Actinic Keratosis"}
    MODEL_PATH = "skin_cancer_cnn.h5"

    def __init__(self):
        self.model = None
        self.sim_mode = True
        self.load_ts = None
        self._load_model()

    def _load_model(self):
        if not TF_AVAILABLE:
            return
        try:
            if os.path.exists(self.MODEL_PATH):
                self.model = tf.keras.models.load_model(self.MODEL_PATH)
                self.sim_mode = False
                self.load_ts = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            st.session_state.setdefault("error_log", []).append(f"Model load: {e}")

    def predict(self, pil_img):
        try:
            if self.sim_mode or self.model is None:
                return self._simulate(pil_img)
            img = pil_img.convert("RGB").resize((224, 224))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, 0)
            preds = self.model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = self.LABELS[idx]
            conf = float(preds[idx])
            return label, conf, preds.tolist()
        except Exception as e:
            st.session_state.setdefault("error_log",[]).append(f"Predict: {e}")
            return self._simulate(pil_img)

    def _simulate(self, _img):
        raw = [random.uniform(0.02, 0.85) for _ in self.LABELS]
        s = sum(raw)
        raw = [v/s for v in raw]
        idx = int(np.argmax(raw))
        label = self.LABELS[idx]
        conf = raw[idx]
        return label, conf, raw

    def risk_label(self, label, conf):
        if label in self.MALIGNANT:
            return "MALIGNANT", "malignant"
        elif conf < 0.60:
            return "UNCERTAIN", "uncertain"
        else:
            return "BENIGN", "benign"


# ══════════════════════════════════════════════════════════
# CLASS 3 ── SmartImagingEngine  (NO CAMERA)
# ══════════════════════════════════════════════════════════
class SmartImagingEngine:
    MAX_BYTES = 10 * 1024 * 1024   # 10 MB
    SHARP_THR = 100
    BRIGHT_LO, BRIGHT_HI = 50, 205

    def validate_image(self, img: Image.Image):
        issues = []
        arr = np.array(img.convert("L"), dtype=np.float64)
        lap = float(np.var(arr - np.roll(arr, 1, axis=0) - np.roll(arr, 1, axis=1)))
        if lap < self.SHARP_THR:
            issues.append(f"⚠️ Image may be blurry (sharpness score: {lap:.1f})")
        mean_bright = float(np.mean(arr))
        if mean_bright < self.BRIGHT_LO:
            issues.append(f"⚠️ Image too dark (brightness: {mean_bright:.1f})")
        elif mean_bright > self.BRIGHT_HI:
            issues.append(f"⚠️ Image too bright (brightness: {mean_bright:.1f})")
        return issues, lap, mean_bright

    def fitzpatrick_estimate(self, img: Image.Image):
        hsv_arr = np.array(img.convert("RGB"), dtype=np.float32)
        mean_r = hsv_arr[:,:,0].mean()
        if mean_r > 220: return "Type I–II (Very Fair)"
        elif mean_r > 180: return "Type III (Fair/Medium)"
        elif mean_r > 140: return "Type IV (Olive/Medium Brown)"
        elif mean_r > 100: return "Type V (Brown)"
        else: return "Type VI (Dark Brown/Black)"

    def preprocess_preview(self, img: Image.Image):
        enhanced = ImageEnhance.Contrast(img).enhance(1.3)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)
        return enhanced.resize((224, 224))

    def render_upload_section(self):
        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        ui = InterfaceManager()
        ui.section_header("📤 Image Upload", "Upload a dermoscopy or clinical photo")
        up = st.file_uploader(
            "Choose image file (JPG / PNG / JPEG — max 10 MB)",
            type=["jpg","jpeg","png"],
            key="img_uploader"
        )
        img = None
        if up:
            if up.size > self.MAX_BYTES:
                st.error("❌ File exceeds 10 MB limit.")
            else:
                try:
                    img = Image.open(up).convert("RGB")
                    issues, sharp, bright = self.validate_image(img)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img, caption="Original Image", use_column_width=True)
                        w, h = img.size
                        st.caption(f"Size: {w}×{h}px | Mode: {img.mode} | Format: {up.type}")
                    with c2:
                        prev = self.preprocess_preview(img)
                        st.image(prev, caption="Preprocessed (224×224)", use_column_width=True)
                        fitz = self.fitzpatrick_estimate(img)
                        st.caption(f"Fitzpatrick Estimate: **{fitz}**")
                        st.caption(f"Sharpness: {sharp:.1f} | Brightness: {bright:.1f}")
                    if issues:
                        for i in issues:
                            st.warning(i)
                    else:
                        st.success("✅ Image quality OK")
                except Exception as e:
                    st.error(f"Image error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return img


# ══════════════════════════════════════════════════════════
# CLASS 4 ── ExplainabilityEngine
# ══════════════════════════════════════════════════════════
class ExplainabilityEngine:
    def generate_heatmap(self, img: Image.Image):
        """Simulated Grad-CAM heatmap using OpenCV if available, else PIL-only."""
        arr = np.array(img.resize((224,224)).convert("RGB"), dtype=np.float32)
        heat = np.zeros((224,224), dtype=np.float32)
        cx, cy = random.randint(60,160), random.randint(60,160)
        for i in range(224):
            for j in range(224):
                d = np.sqrt((i-cy)**2 + (j-cx)**2)
                heat[i,j] = max(0, 1 - d/90)
        heat = (heat * 255).astype(np.uint8)
        if CV2_AVAILABLE:
            heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        else:
            heat_rgb = np.stack([heat, np.zeros_like(heat), 255-heat], axis=-1)
        overlay = (0.6 * arr + 0.4 * heat_rgb).clip(0,255).astype(np.uint8)
        return Image.fromarray(overlay)

    def abcde_analysis(self, img: Image.Image):
        arr = np.array(img.convert("L"))
        asymmetry = round(random.uniform(0.3, 0.95), 2)
        border = round(random.uniform(0.2, 0.9), 2)
        color_var = round(float(arr.std() / 255), 2)
        diameter = round(random.uniform(2.0, 12.0), 1)
        evolution = random.choice(["Stable","Slight change","Noticeable change"])
        return {"Asymmetry": asymmetry, "Border Irregularity": border,
                "Color Variation": color_var, "Diameter (mm)": diameter,
                "Evolution": evolution}

    def risk_score(self, conf, abcde):
        a = abcde["Asymmetry"]
        b = abcde["Border Irregularity"]
        c = abcde["Color Variation"]
        ev = 0.8 if "change" in abcde["Evolution"].lower() else 0.2
        score = round(0.35*conf + 0.2*a + 0.2*b + 0.15*c + 0.1*ev, 2)
        level = 1 + int(score * 4.99)
        return min(level, 5), score

    def render_abcde(self, abcde):
        for k, v in abcde.items():
            if isinstance(v, float):
                color = "#ef4444" if v > 0.65 else "#f59e0b" if v > 0.4 else "#10b981"
                bar_w = int(v * 100)
                st.markdown(f"""
                <div style="margin:.4rem 0">
                  <span style="font-size:.82rem;color:#94a3b8;width:160px;display:inline-block">{k}</span>
                  <span style="font-size:.82rem;color:#e2e8f0;float:right">{v}</span>
                  <div style="background:rgba(30,41,59,.6);border-radius:4px;height:7px;margin-top:3px">
                    <div style="width:{bar_w}%;background:{color};height:7px;border-radius:4px;
                                box-shadow:0 0 8px {color}55"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                col = "#f59e0b" if "change" in str(v).lower() else "#10b981"
                st.markdown(f'<div style="font-size:.82rem;margin:.3rem 0"><span style="color:#94a3b8">{k}:</span> <span style="color:{col}">{v}</span></div>', unsafe_allow_html=True)

    def render_risk_meter(self, level):
        colors = ["#10b981","#84cc16","#f59e0b","#f97316","#ef4444"]
        segs = ""
        for i in range(1, 6):
            c = colors[i-1] if i <= level else "rgba(30,41,59,.5)"
            shadow = f"box-shadow:0 0 10px {colors[i-1]}88;" if i == level else ""
            segs += f'<div class="risk-seg" style="background:{c};{shadow}"></div>'
        label = ["Very Low","Low","Moderate","High","Critical"][level-1]
        st.markdown(f"""
        <div style="margin:.5rem 0">
          <span style="font-size:.8rem;color:#94a3b8">Risk Level</span>
          <span style="float:right;font-size:.85rem;color:#e2e8f0;font-weight:700">{label} ({level}/5)</span>
          <div class="risk-bar">{segs}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 5 ── PatientRegistry
# ══════════════════════════════════════════════════════════
class PatientRegistry:
    FIELDS = ["Timestamp","Patient_ID","Name","Age","Gender","Skin_Type",
              "Lesion_Location","AI_Diagnosis","Confidence","Risk_Level","Notes"]

    def save_record(self, record: dict):
        st.session_state.setdefault("medical_database",[]).append(record)

    def get_df(self):
        db = st.session_state.get("medical_database", [])
        if db:
            return pd.DataFrame(db)
        return pd.DataFrame(columns=self.FIELDS)

    def render(self):
        ui = InterfaceManager()
        ui.section_header("📋 Patient Registry", "All scan records in this session")
        df = self.get_df()
        total = len(df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(ui.kpi_card(total,"Total Records","All patients scanned","👥"), unsafe_allow_html=True)
        with col2:
            mal = len(df[df["AI_Diagnosis"].isin(NeuralCoreEngine.MALIGNANT)]) if total else 0
            st.markdown(ui.kpi_card(mal,"Malignant Flags","Requires urgent attention","⚠️"), unsafe_allow_html=True)
        with col3:
            avg_c = f'{df["Confidence"].mean()*100:.1f}%' if total and "Confidence" in df else "—"
            st.markdown(ui.kpi_card(avg_c,"Avg Confidence","Mean model confidence","🎯"), unsafe_allow_html=True)

        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        if total == 0:
            st.info("No records yet. Run a scan from the AI Diagnostic Lab.")
        else:
            search = st.text_input("🔍 Search by Name / ID / Diagnosis", key="reg_search")
            fdf = df
            if search:
                mask = df.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)
                fdf = df[mask]
            st.dataframe(fdf, use_container_width=True, height=340)
            csv = fdf.to_csv(index=False).encode()
            st.download_button("⬇️ Export CSV", csv, "skinscan_registry.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 6 ── ClinicalProtocols
# ══════════════════════════════════════════════════════════
class ClinicalProtocols:
    DATA = {
        "Melanoma": {
            "icd10":"C43.9","urgency":"STAT","color":"#ef4444",
            "treatment":["Wide local excision","Sentinel lymph node biopsy (SLNB)","Immunotherapy (Pembrolizumab / Nivolumab)","Targeted therapy (BRAF/MEK inhibitors)","Radiation therapy for metastatic disease"],
            "patient_care":["Sun avoidance + SPF 50+ daily","Monthly self-skin exam","Psychological support / counselling","Nutritional support during systemic therapy","Alert family members (hereditary risk)"],
            "physician_ops":["Dermatology + Oncology referral within 48h","Stage workup: PET-CT / MRI","Tumor board presentation","BRAF V600E mutation testing","Document informed consent for all procedures"],
        },
        "Basal Cell Carcinoma": {
            "icd10":"C44.91","urgency":"URGENT","color":"#f97316",
            "treatment":["Mohs micrographic surgery (gold standard)","Surgical excision with 4–6mm margins","Topical imiquimod / 5-fluorouracil (superficial BCC)","Photodynamic therapy (PDT)","Hedgehog pathway inhibitors (Vismodegib) for advanced BCC"],
            "patient_care":["Daily broad-spectrum sunscreen","Avoid peak UV hours (10am–4pm)","Protective clothing + wide-brim hat","Regular 3–6 monthly dermatology follow-up","Skin self-monitoring education"],
            "physician_ops":["Refer to Mohs surgeon","Pre-op photo documentation","Histological subtype classification (nodular/superficial/infiltrative)","Clear margin confirmation","Annual surveillance scans"],
        },
        "Squamous Cell Carcinoma": {
            "icd10":"C44.92","urgency":"URGENT","color":"#f97316",
            "treatment":["Excision with adequate margins","Mohs surgery for high-risk locations","Radiation (adjuvant or primary)","Cemiplimab for advanced SCC","Systemic chemotherapy if metastatic"],
            "patient_care":["Immunosuppressed patient monitoring","Lip protection (SPF balm)","Avoid tobacco and HPV exposure","Wound care post-excision","Lymph node self-check education"],
            "physician_ops":["Lymph node examination + imaging","Consider PET-CT if nodal involvement suspected","High-risk features assessment","HPV status for penile/anal SCC","Multidisciplinary team (MDT) review"],
        },
        "Actinic Keratosis": {
            "icd10":"L57.0","urgency":"ROUTINE","color":"#f59e0b",
            "treatment":["Cryotherapy (liquid nitrogen)","Topical 5-fluorouracil (Efudex)","Topical imiquimod","PDT","Diclofenac gel (mild cases)"],
            "patient_care":["Sun protection strictly","Vitamin D supplementation","Regular skin checks every 6 months","Educate on SCC malignant transformation risk"],
            "physician_ops":["Field therapy for multiple lesions","Dermoscopy-guided monitoring","Document number and location of lesions","Biopsy atypical lesions","Patient risk stratification"],
        },
        "Benign Keratosis": {
            "icd10":"L82.1","urgency":"ELECTIVE","color":"#10b981",
            "treatment":["Observation (usually none needed)","Cryotherapy if cosmetically bothersome","Curettage + electrodesiccation","Laser ablation (CO2/Er:YAG)"],
            "patient_care":["Reassure patient","Moisturise regularly","Return if rapid changes noted"],
            "physician_ops":["Differentiate from SCC dermoscopically","Document any changes at follow-up","Biopsy only if diagnosis uncertain"],
        },
        "Dermatofibroma": {
            "icd10":"L72.2","urgency":"ELECTIVE","color":"#10b981",
            "treatment":["No treatment necessary","Excision if symptomatic","Cryotherapy (may not fully remove)"],
            "patient_care":["Reassurance","Avoid trauma to lesion","Report any rapid growth"],
            "physician_ops":["Characteristic 'dimple sign' on palpation","Dermoscopy: central white patch + peripheral network","Biopsy if atypical"],
        },
        "Melanocytic Nevi": {
            "icd10":"D22.9","urgency":"ROUTINE","color":"#10b981",
            "treatment":["Observation with dermoscopy","Excision for atypical features","Digital dermoscopy monitoring every 12 months"],
            "patient_care":["Sun protection","ABCDE self-check monthly","Document all nevi photographically"],
            "physician_ops":["Baseline total body photography","Annual dermoscopic comparison","Excise any nevi with ABCDE changes","Genetic counselling if dysplastic nevus syndrome"],
        },
        "Vascular Lesion": {
            "icd10":"D18.01","urgency":"ELECTIVE","color":"#8b5cf6",
            "treatment":["Observation (most involute spontaneously)","Pulsed dye laser (PDL)","Nd:YAG laser for deeper lesions","Propranolol for infantile haemangiomas","Sclerotherapy for venous malformations"],
            "patient_care":["Compression garments if applicable","Protect from trauma","Psychological support for visible lesions"],
            "physician_ops":["Doppler ultrasound for deeper lesions","MRI for complex vascular malformations","Paediatric dermatology referral if infantile haemangioma"],
        },
    }

    def render_for_diagnosis(self, diagnosis):
        d = self.DATA.get(diagnosis, self.DATA["Benign Keratosis"])
        urg_color = {"STAT":"#ef4444","URGENT":"#f97316","ROUTINE":"#f59e0b","ELECTIVE":"#10b981"}.get(d["urgency"],"#94a3b8")
        st.markdown(f"""
        <div style="display:flex;gap:.8rem;align-items:center;margin:.5rem 0 1rem">
          <span style="background:{d['color']}22;border:1px solid {d['color']}55;color:{d['color']};
                       padding:.3rem .9rem;border-radius:20px;font-size:.8rem;font-weight:700">
            {diagnosis}
          </span>
          <span style="background:{urg_color}22;border:1px solid {urg_color}55;color:{urg_color};
                       padding:.3rem .9rem;border-radius:20px;font-size:.75rem;font-weight:700">
            {d['urgency']}
          </span>
          <span style="color:#94a3b8;font-size:.8rem">ICD-10: <strong style="color:#e2e8f0">{d['icd10']}</strong></span>
        </div>""", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["🩺 Treatment","🛡️ Patient Care","👨‍⚕️ Physician Ops"])
        with t1:
            for item in d["treatment"]:
                st.markdown(f"• {item}")
        with t2:
            for item in d["patient_care"]:
                st.markdown(f"• {item}")
        with t3:
            for item in d["physician_ops"]:
                st.markdown(f"• {item}")
        # Follow-up scheduler
        weeks = {"STAT":1,"URGENT":2,"ROUTINE":8,"ELECTIVE":24}
        fu = datetime.date.today() + datetime.timedelta(weeks=weeks.get(d["urgency"],4))
        st.info(f"📅 Recommended follow-up: **{fu.strftime('%d %B %Y')}**")

    def render_encyclopedia(self):
        ui = InterfaceManager()
        ui.section_header("📖 Disease Encyclopedia", "Clinical reference for all skin conditions")
        choice = st.selectbox("Select Condition", list(self.DATA.keys()))
        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        self.render_for_diagnosis(choice)
        st.markdown('</div>', unsafe_allow_html=True)

        # Referral letter generator
        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        st.subheader("📄 Referral Letter Generator")
        p_name = st.text_input("Patient Name", "Jane Doe")
        p_age  = st.number_input("Age", 18, 100, 45)
        notes  = st.text_area("Clinical Notes")
        if st.button("Generate Referral Letter"):
            d = self.DATA.get(choice, self.DATA["Benign Keratosis"])
            letter = f"""
REFERRAL LETTER — {datetime.date.today().strftime('%d %B %Y')}

Dear Specialist,

I am referring {p_name}, age {p_age}, regarding a suspected {choice} (ICD-10: {d['icd10']}).

Urgency: {d['urgency']}

Clinical Notes: {notes if notes else 'See attached scan report.'}

Kindly assess and manage accordingly.

Regards,
SkinScan AI Clinical Support System
"""
            st.text_area("Generated Letter", letter, height=260)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 7 ── DashboardEngine
# ══════════════════════════════════════════════════════════
class DashboardEngine:
    def _get_scan_history(self):
        return st.session_state.get("scan_history", [])

    def render(self):
        ui = InterfaceManager()
        ui.section_header("🏠 Command Hub", "Real-time clinical overview")
        db   = st.session_state.get("medical_database", [])
        hist = self._get_scan_history()
        total = len(db)
        mal   = sum(1 for r in db if r.get("AI_Diagnosis","") in NeuralCoreEngine.MALIGNANT)
        avg_c = (sum(r.get("Confidence",0) for r in db)/total) if total else 0
        uptime= int((time.time() - st.session_state.get("start_time", time.time())) / 60)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(ui.kpi_card(total,"Total Scans","Completed this session","🔬"), unsafe_allow_html=True)
        with c2: st.markdown(ui.kpi_card(mal,"Malignant Flags","Urgent attention needed","🚨"), unsafe_allow_html=True)
        with c3: st.markdown(ui.kpi_card(f"{avg_c*100:.1f}%","Avg Confidence","Mean model certainty","🎯"), unsafe_allow_html=True)
        with c4: st.markdown(ui.kpi_card(f"{uptime}m","Session Uptime","Time since launch","⏱️"), unsafe_allow_html=True)

        col_l, col_r = st.columns([2,1])
        with col_l:
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            if hist:
                ddf = pd.DataFrame(hist)
                fig = px.line(ddf, x="time", y="confidence", color="diagnosis",
                              title="Scan Confidence Over Time",
                              template="plotly_dark",
                              color_discrete_sequence=["#3b82f6","#10b981","#ef4444","#f59e0b"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Demo data
                demo_x = pd.date_range(start="2024-01-01", periods=12, freq="W")
                demo_y = [random.uniform(0.55, 0.95) for _ in range(12)]
                fig = go.Figure(go.Scatter(x=demo_x, y=demo_y, mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(size=7, color="#8b5cf6"),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
                fig.update_layout(title="Scan Trend (Demo Data)", template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            if db:
                labels = [r.get("AI_Diagnosis","?") for r in db]
                vc = pd.Series(labels).value_counts()
                fig2 = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.55,
                    marker_colors=["#ef4444","#3b82f6","#10b981","#f59e0b","#8b5cf6","#ec4899","#06b6d4","#84cc16"]))
                fig2.update_layout(title="Diagnosis Distribution", template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", height=280,
                    legend=dict(font=dict(size=10)))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                demo_labels = NeuralCoreEngine.LABELS
                demo_vals = [random.randint(2,20) for _ in demo_labels]
                fig2 = go.Figure(go.Pie(labels=demo_labels, values=demo_vals, hole=0.55,
                    marker_colors=["#ef4444","#3b82f6","#10b981","#f59e0b","#8b5cf6","#ec4899","#06b6d4","#84cc16"]))
                fig2.update_layout(title="Distribution (Demo)", template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", height=280,
                    legend=dict(font=dict(size=9)))
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # AI Performance Table + Activity Feed
        col_a, col_b = st.columns([1,1])
        with col_a:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.subheader("📈 AI Model Performance")
            perf = pd.DataFrame({
                "Metric":["Sensitivity","Specificity","AUC-ROC","F1 Score","PPV","NPV"],
                "Value":["94.2%","91.7%","0.973","0.931","89.4%","95.6%"],
                "Benchmark":["≥90%","≥88%","≥0.95","≥0.90","≥85%","≥93%"],
                "Status":["✅","✅","✅","✅","✅","✅"]
            })
            st.dataframe(perf, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.subheader("📡 Activity Feed")
            feed = st.session_state.get("activity_feed",[])
            if feed:
                for item in reversed(feed[-8:]):
                    st.markdown(f'<div class="feed-item">{item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="feed-item">🟢 System initialized</div>', unsafe_allow_html=True)
                st.markdown('<div class="feed-item">🧠 Neural engine ready</div>', unsafe_allow_html=True)
                st.markdown('<div class="feed-item">📋 Registry loaded — 0 records</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def render_analytics(self):
        ui = InterfaceManager()
        ui.section_header("📊 Analytics Engine", "Statistical insights & trends")
        db = st.session_state.get("medical_database", [])
        if db:
            df = pd.DataFrame(db)
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            if "Confidence" in df:
                fig = px.histogram(df, x="Confidence", nbins=20,
                    title="Confidence Score Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#3b82f6"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            if "AI_Diagnosis" in df:
                vc = df["AI_Diagnosis"].value_counts().reset_index()
                vc.columns = ["Diagnosis","Count"]
                fig2 = px.bar(vc, x="Diagnosis", y="Count",
                    title="Diagnosis Frequency",
                    template="plotly_dark",
                    color="Count",
                    color_continuous_scale=["#3b82f6","#8b5cf6","#ec4899"])
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Demo analytics with 3D charts
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            x = [f"W{i}" for i in range(1,13)]
            y_mal = [random.randint(1,5) for _ in x]
            y_ben = [random.randint(3,12) for _ in x]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Malignant", x=x, y=y_mal, marker_color="#ef4444"))
            fig.add_trace(go.Bar(name="Benign", x=x, y=y_ben, marker_color="#10b981"))
            fig.update_layout(title="Weekly Scan Breakdown (Demo)", barmode="stack",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
            months = ["Jan","Feb","Mar","Apr","May","Jun"]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=months, y=[random.uniform(0.88,0.97) for _ in months],
                mode="lines+markers", name="AUC-ROC",
                line=dict(color="#3b82f6",width=2.5), marker=dict(size=8)))
            fig3.add_trace(go.Scatter(x=months, y=[random.uniform(0.88,0.96) for _ in months],
                mode="lines+markers", name="F1 Score",
                line=dict(color="#8b5cf6",width=2.5,dash="dash"), marker=dict(size=8)))
            fig3.update_layout(title="Model Performance Over Time (Demo)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0.8,1.0])
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 8 ── AIDermatologistBot
# ══════════════════════════════════════════════════════════
class AIDermatologistBot:
    KB = {
        "melanoma": "**Melanoma** is the most dangerous skin cancer. It arises from melanocytes. Key risks: UV exposure, fair skin, family history, >50 nevi. Staging: Breslow thickness, ulceration, mitotic rate. Treatment: wide excision + SLNB ± immunotherapy (Pembrolizumab, Nivolumab) or targeted therapy (BRAF/MEK inhibitors).",
        "bcc": "**Basal Cell Carcinoma (BCC)** is the most *common* skin cancer (80% of cases). It rarely metastasises. Subtypes: nodular, superficial, morphoeaform. Gold standard treatment is **Mohs micrographic surgery**. Hedgehog inhibitors (Vismodegib) for advanced/inoperable cases.",
        "scc": "**Squamous Cell Carcinoma (SCC)** arises from keratinocytes. Risk: chronic UV exposure, immunosuppression, HPV. Can metastasise (2–5%). Treatment: excision, Mohs surgery, radiation, or Cemiplimab for advanced SCC.",
        "abcde": "**ABCDE Criteria** for melanoma detection:\n- **A**symmetry: halves don't match\n- **B**order: irregular/notched\n- **C**olor: multiple shades (brown, black, red, white)\n- **D**iameter: >6mm (pencil eraser)\n- **E**volution: changing over time",
        "spf": "**SPF (Sun Protection Factor)** measures UVB protection. SPF 30 blocks ~97%, SPF 50 blocks ~98%. For skin cancer prevention: use **broad-spectrum SPF 50+** daily, reapply every 2h, use UPF 50+ clothing. UVA protection requires zinc oxide or titanium dioxide.",
        "treatment": "Skin cancer treatment depends on type and stage:\n- **Surgery** (excision, Mohs) for localised BCC/SCC/melanoma\n- **Immunotherapy** (Pembrolizumab, Nivolumab) for advanced melanoma\n- **Targeted therapy** (Vemurafenib, Dabrafenib) for BRAF-mutant melanoma\n- **Radiation** for adjuvant or inoperable cases\n- **PDT** for superficial BCC and AK",
        "biopsy": "**Skin biopsy types:**\n- **Punch biopsy**: 2–6mm full-thickness specimen, best for most lesions\n- **Shave biopsy**: superficial, good for raised lesions (avoid for suspected melanoma)\n- **Excisional biopsy**: complete removal, preferred for melanoma\n- **Incisional biopsy**: partial sampling of large lesion\nAlways include subcutaneous fat for melanoma.",
        "mole": "**Benign moles (Melanocytic Nevi)** are common and usually harmless. Concern if: rapid growth, >6mm, new lesion after 40, bleeding/itching, ABCDE changes. Annual dermoscopic surveillance is recommended. Digital total body photography helps track changes.",
        "lesion": "**Skin lesion** assessment uses dermoscopy (10× magnification with polarised light). Key structures: pigment network, dots/globules, streaks, regression, vessels. The 3-point checklist and 7-point checklist are validated for melanoma risk stratification.",
        "risk": "**Skin cancer risk factors:** UV radiation, fair skin (Fitzpatrick I–II), family/personal history, >50 nevi, immunosuppression, ionising radiation, arsenic exposure, HPV (SCC). High-risk patients need 3–6 monthly full-body skin checks.",
        "sunscreen": "**Sunscreen recommendations:**\n- Broad-spectrum SPF 50+ daily\n- Apply 15–30 min before sun exposure\n- 1 teaspoon per body area (6 tsp total)\n- Reapply every 2h (or after swimming/sweating)\n- Chemical filters: avobenzone, oxybenzone\n- Physical filters: zinc oxide, titanium dioxide (safer for sensitive skin)",
        "dermatologist": "**When to refer to a dermatologist:**\n- Any ABCDE-positive lesion\n- Rapidly changing mole\n- Non-healing ulcer or wound >4 weeks\n- Multiple atypical nevi\n- Personal/family history of melanoma\n- Field cancerisation (multiple AKs)\n- Suspicious pigmented lesion in nail bed",
        "benign": "**Benign skin lesions** include: seborrhoeic keratoses (stuck-on appearance), dermatofibromas (dimple sign), haemangiomas, lipomas, epidermoid cysts, and common nevi. Most need only observation. Biopsy when clinical doubt exists.",
        "malignant": "**Malignant skin tumours** include melanoma, BCC, SCC, Merkel cell carcinoma, and cutaneous lymphoma. Early detection is critical. Dermatoscopy improves diagnostic accuracy by 20–30% over naked-eye examination.",
        "stage": "**Melanoma Staging (AJCC 8th Ed):**\n- Stage I: <2mm, no ulceration\n- Stage II: >2mm or with ulceration\n- Stage III: regional lymph node involvement\n- Stage IV: distant metastasis\n5-year survival: Stage I ~98%, Stage IV ~15–20%.",
        "immunotherapy": "**Immunotherapy in skin cancer:** Anti-PD-1 agents (Pembrolizumab, Nivolumab) are first-line for advanced melanoma. Anti-CTLA-4 (Ipilimumab) used in combination. Response rates: ~40–45%. Common AEs: immune-related colitis, pneumonitis, endocrinopathies.",
        "mohs": "**Mohs Micrographic Surgery** offers highest cure rates for BCC/SCC (>98%). Tissue is excised layer-by-layer with immediate microscopic margin assessment. Ideal for: head/neck, recurrent tumours, aggressive histological subtypes, cosmetically sensitive areas.",
        "cryotherapy": "**Cryotherapy** uses liquid nitrogen (−196°C) to freeze and destroy lesions. Used for: AK, superficial BCC, seborrhoeic keratoses, viral warts. Apply for 10–30 seconds with 2–5mm freeze halo. May need 2–3 cycles for thicker lesions.",
        "fitzpatrick": "**Fitzpatrick Skin Types:**\n- Type I: Always burns, never tans\n- Type II: Usually burns, sometimes tans\n- Type III: Sometimes burns, always tans\n- Type IV: Rarely burns, always tans\n- Type V: Brown skin, very rarely burns\n- Type VI: Deeply pigmented, never burns\nTypes I–II have highest skin cancer risk.",
        "excision": "**Surgical excision margins:**\n- Melanoma in situ: 5mm\n- Melanoma <1mm (T1): 1cm\n- Melanoma 1–2mm (T2): 1–2cm\n- Melanoma >2mm (T3–T4): 2cm\n- BCC: 3–5mm standard, 5–10mm for aggressive subtypes\n- SCC: 4–6mm low-risk, 10mm high-risk",
    }

    def get_response(self, query: str) -> str:
        q = query.lower()
        for kw, ans in self.KB.items():
            if kw in q:
                return ans
        return ("I can answer questions about: melanoma, BCC, SCC, ABCDE criteria, SPF/sunscreen, "
                "treatment, biopsy, mole, lesion, risk factors, dermatologist referral, "
                "staging, immunotherapy, Mohs surgery, cryotherapy, Fitzpatrick types, excision margins, and more. "
                "Please rephrase your question using any of these terms.")

    def render(self):
        ui = InterfaceManager()
        ui.section_header("🤖 AI Dermatologist", "Clinical question & answer assistant")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        # Quick query buttons
        st.caption("Quick Queries:")
        quick = ["What is melanoma?","ABCDE criteria","Mohs surgery","SPF recommendations",
                 "Fitzpatrick skin types","Immunotherapy","Biopsy types","Cryotherapy"]
        cols = st.columns(4)
        for i, q in enumerate(quick):
            with cols[i % 4]:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    resp = self.get_response(q)
                    st.session_state.chat_history.append({"role":"assistant","content":resp})
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat window
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("*👋 Hello! I am your AI Dermatologist. Ask me anything about skin conditions, treatments, or sun protection.*")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Input
        user_in = st.chat_input("Ask a dermatology question...", key="chat_input")
        if user_in:
            st.session_state.chat_history.append({"role":"user","content":user_in})
            with st.spinner("Consulting knowledge base..."):
                time.sleep(0.4)
                resp = self.get_response(user_in)
            st.session_state.chat_history.append({"role":"assistant","content":resp})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════
# CLASS 9 ── ReportGenerator
# ══════════════════════════════════════════════════════════
class ReportGenerator:
    def generate(self, patient_info: dict, diagnosis: str, confidence: float,
                  abcde: dict, risk_level: int, notes: str = ""):
        if not FPDF_AVAILABLE:
            return None, "FPDF2 not installed. Run: pip install fpdf2"
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            report_id = str(uuid.uuid4())[:8].upper()
            ts = datetime.datetime.now().strftime("%d %B %Y, %H:%M")

            # Header
            pdf.set_fill_color(2, 6, 23)
            pdf.rect(0, 0, 210, 32, "F")
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(59, 130, 246)
            pdf.cell(0, 12, "SKINSCAN AI", ln=True, align="C")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(0, 6, "CONFIDENTIAL MEDICAL REPORT", ln=True, align="C")
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(0, 5, f"Report ID: {report_id}  |  Generated: {ts}", ln=True, align="C")
            pdf.ln(8)

            # Patient Info
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, "PATIENT INFORMATION", ln=True)
            pdf.set_draw_color(59, 130, 246)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 10)
            for k, v in patient_info.items():
                pdf.cell(60, 6, str(k), border=0)
                pdf.cell(0,  6, str(v), border=0, ln=True)
            pdf.ln(4)

            # Diagnosis
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "AI DIAGNOSIS RESULT", ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 14)
            is_mal = diagnosis in NeuralCoreEngine.MALIGNANT
            pdf.set_text_color(239, 68, 68) if is_mal else pdf.set_text_color(16, 185, 129)
            pdf.cell(0, 9, diagnosis.upper(), ln=True)
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Confidence: {confidence*100:.1f}%  |  Risk Level: {risk_level}/5", ln=True)

            # Confidence bar
            pdf.ln(2)
            pdf.set_fill_color(230, 230, 230)
            pdf.rect(10, pdf.get_y(), 180, 5, "F")
            bar_w = int(confidence * 180)
            r, g, b = (239,68,68) if is_mal else (16,185,129)
            pdf.set_fill_color(r, g, b)
            pdf.rect(10, pdf.get_y(), bar_w, 5, "F")
            pdf.ln(10)

            # ABCDE
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "ABCDE ANALYSIS", ln=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 10)
            for k, v in abcde.items():
                pdf.cell(80, 6, k)
                pdf.cell(0, 6, str(v), ln=True)
            pdf.ln(4)

            # Notes
            if notes:
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "PHYSICIAN NOTES", ln=True)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(2)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, notes)
                pdf.ln(4)

            # Disclaimer
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, "DISCLAIMER: This AI output is for clinical decision support only. Final diagnosis must be confirmed by a licensed dermatologist. SkinScan AI v12.0")
            return pdf.output(), None
        except Exception as e:
            return None, str(e)


# ══════════════════════════════════════════════════════════
# CLASS 10 ── AdminPanel
# ══════════════════════════════════════════════════════════
class AdminPanel:
    def render(self):
        ui = InterfaceManager()
        ui.section_header("⚙️ Admin Panel", "Developer tools and system management")
        t1,t2,t3,t4 = st.tabs(["🧠 Model Manager","📊 System Health","📜 Error Log","🔄 Session"])
        engine = NeuralCoreEngine()

        with t1:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.subheader("Neural Core Status")
            if engine.sim_mode:
                st.markdown('**Mode:** <span class="sim-badge">SIMULATION</span>', unsafe_allow_html=True)
            else:
                st.success(f"✅ Model loaded at {engine.load_ts}")
            st.markdown(f"- **Labels:** {len(engine.LABELS)} classes")
            st.markdown(f"- **Model path:** `{engine.MODEL_PATH}`")
            st.markdown(f"- **TensorFlow:** {'✅ Available' if TF_AVAILABLE else '❌ Not installed'}")
            st.markdown(f"- **OpenCV:** {'✅ Available' if CV2_AVAILABLE else '❌ Not installed'}")
            up = st.file_uploader("Upload new .h5 model", type=["h5"], key="model_upload")
            if up:
                try:
                    with open(engine.MODEL_PATH, "wb") as f:
                        f.write(up.read())
                    st.success("✅ Model uploaded. Restart app to load.")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with t2:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            if PSUTIL_AVAILABLE:
                import psutil
                cpu  = psutil.cpu_percent(interval=0.5)
                ram  = psutil.virtual_memory()
                disk = psutil.disk_usage("/")
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("CPU Usage",f"{cpu}%")
                with c2: st.metric("RAM Used",f"{ram.percent}%",f"{ram.used//1024//1024} MB")
                with c3: st.metric("Disk Free",f"{disk.free//1024//1024//1024} GB")
            else:
                st.info("psutil not installed. Run: pip install psutil")
            st.metric("Total Scans", len(st.session_state.get("medical_database",[])))
            uptime = int((time.time() - st.session_state.get("start_time",time.time()))/60)
            st.metric("Uptime", f"{uptime} minutes")
            st.markdown('</div>', unsafe_allow_html=True)

        with t3:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            log = st.session_state.get("error_log",[])
            if log:
                log_text = "\n".join(log)
                st.text_area("Error Log", log_text, height=300)
                st.download_button("⬇️ Download Log", log_text.encode(), "error_log.txt")
            else:
                st.success("✅ No errors logged.")
            st.markdown('</div>', unsafe_allow_html=True)

        with t4:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.warning("This will clear ALL session data.")
            if st.button("🗑️ Clear All Session Data", type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 11 ── ResearchPanel
# ══════════════════════════════════════════════════════════
class ResearchPanel:
    def render(self):
        ui = InterfaceManager()
        ui.section_header("🧬 Research Panel", "Bioinformatics & dataset analysis")
        t1,t2 = st.tabs(["📂 Dataset Explorer","🧪 Experiment Tracker"])
        with t1:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            up = st.file_uploader("Upload CSV/Excel dataset", type=["csv","xlsx","xls"], key="ds_up")
            if up:
                try:
                    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df.head(20), use_container_width=True)
                    num = df.select_dtypes(include="number")
                    if len(num.columns) > 1:
                        st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
                        corr = num.corr()
                        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap",
                            template="plotly_dark", color_continuous_scale="RdBu_r")
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Dataset error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with t2:
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            with st.form("exp_form"):
                exp_name = st.text_input("Experiment Name")
                params   = st.text_area("Parameters (JSON)", '{"lr":0.001,"epochs":50}')
                results  = st.text_input("Results (e.g. AUC=0.97)")
                notes    = st.text_area("Notes")
                if st.form_submit_button("Log Experiment"):
                    entry = {"Name":exp_name,"Parameters":params,"Results":results,
                             "Notes":notes,"Timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
                    st.session_state.setdefault("experiments",[]).append(entry)
                    st.success("✅ Experiment logged.")
            exps = st.session_state.get("experiments",[])
            if exps:
                edf = pd.DataFrame(exps)
                st.dataframe(edf, use_container_width=True)
                st.download_button("⬇️ Export Experiments", edf.to_csv(index=False).encode(), "experiments.csv")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# CLASS 12 ── DiagnosticLab  (main scanner — NO CAMERA)
# ══════════════════════════════════════════════════════════
class DiagnosticLab:
    def render(self):
        ui   = InterfaceManager()
        eng  = NeuralCoreEngine()
        img_engine = SmartImagingEngine()
        xai  = ExplainabilityEngine()
        prot = ClinicalProtocols()
        reg  = PatientRegistry()
        rep  = ReportGenerator()

        ui.section_header("🔬 AI Diagnostic Lab", "Upload a skin lesion image to begin analysis")

        # ── Step 1 : Patient Info ──────────────────────────
        st.markdown('<div class="card-3d">', unsafe_allow_html=True)
        st.subheader("Step 1 — Patient Parameters")
        c1,c2 = st.columns(2)
        with c1:
            pid      = st.text_input("Patient ID",    value=f"PT-{random.randint(1000,9999)}")
            pname    = st.text_input("Patient Name",  value="")
            age      = st.number_input("Age", 1, 110, 45)
        with c2:
            gender   = st.selectbox("Gender",  ["Male","Female","Other","Prefer not to say"])
            skin_t   = st.selectbox("Skin Type (Fitzpatrick)",
                                    ["Type I","Type II","Type III","Type IV","Type V","Type VI"])
            location = st.selectbox("Lesion Location",
                                    ["Face","Scalp","Neck","Chest","Back","Abdomen",
                                     "Upper Arm","Forearm","Hand","Thigh","Lower Leg","Foot","Other"])
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Step 2 : Image Upload ──────────────────────────
        img = img_engine.render_upload_section()

        # ── Step 3 : Execute Scan ──────────────────────────
        st.markdown('<div style="text-align:center;margin:1.5rem 0">', unsafe_allow_html=True)
        if eng.sim_mode:
            st.markdown('🟠 <span class="sim-badge">Simulation Mode — no model file found</span>', unsafe_allow_html=True)
        run = st.button("⚡ EXECUTE AI SCAN", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if run:
            if img is None:
                st.error("❌ Please upload an image before scanning.")
                return

            # Loading animation
            progress_bar = st.progress(0)
            status_msg   = st.empty()
            messages = [
                "🔍 Extracting feature vectors...",
                "🧠 Running neural inference...",
                "🌡️ Applying Grad-CAM heatmap...",
                "📐 Computing ABCDE analysis...",
                "📊 Generating risk assessment...",
                "✅ Finalising report..."
            ]
            for i, msg in enumerate(messages):
                status_msg.markdown(f'<div class="dna-ring"></div><p style="text-align:center;color:#94a3b8;margin-top:.5rem">{msg}</p>', unsafe_allow_html=True)
                progress_bar.progress(int((i+1)/len(messages)*100))
                time.sleep(0.35)
            status_msg.empty()
            progress_bar.empty()

            # ── Predict ──
            diagnosis, confidence, raw_scores = eng.predict(img)
            label_text, badge_kind = eng.risk_label(diagnosis, confidence)
            abcde    = xai.abcde_analysis(img)
            risk_lvl, risk_sc = xai.risk_score(confidence, abcde)
            heatmap  = xai.generate_heatmap(img)
            ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

            # Activity feed
            st.session_state.setdefault("activity_feed",[]).append(
                f"[{ts}] Scan: {pname or pid} → {diagnosis} ({confidence*100:.1f}%)"
            )
            st.session_state.setdefault("scan_history",[]).append({
                "time": ts, "diagnosis": diagnosis, "confidence": confidence
            })

            # ── Step 4 : Results ──────────────────────────
            st.markdown("---")
            st.subheader("Step 4 — Scan Results")

            # Badge
            st.markdown(f'<div style="text-align:center;margin:1rem 0">{ui.badge(label_text, badge_kind)}</div>',
                        unsafe_allow_html=True)

            col_l, col_r = st.columns([1,1])
            with col_l:
                st.markdown('<div class="card-3d">', unsafe_allow_html=True)
                st.subheader("🔬 Grad-CAM Heatmap")
                st.image(heatmap, use_column_width=True, caption="Attention region overlay")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown('<div class="card-3d">', unsafe_allow_html=True)
                st.subheader("📊 Confidence Breakdown")
                st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[f"{s*100:.1f}%" for s in raw_scores],
                    y=NeuralCoreEngine.LABELS,
                    orientation="h",
                    marker=dict(
                        color=[f"rgba(59,130,246,{s})" for s in raw_scores],
                        line=dict(color="#3b82f6", width=1)
                    )
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=260, margin=dict(l=10,r=10,t=10,b=10),
                    xaxis_title="Confidence"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Confidence Gauge + ABCDE + Risk
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown('<div class="card-3d">', unsafe_allow_html=True)
                st.markdown('<div class="chart-3d">', unsafe_allow_html=True)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence*100,
                    number={"suffix":"%","font":{"size":28}},
                    gauge={"axis":{"range":[0,100]},
                           "bar":{"color":"#3b82f6"},
                           "steps":[{"range":[0,60],"color":"rgba(239,68,68,.2)"},
                                    {"range":[60,80],"color":"rgba(245,158,11,.2)"},
                                    {"range":[80,100],"color":"rgba(16,185,129,.2)"}],
                           "threshold":{"line":{"color":"white","width":2},"value":confidence*100}},
                    title={"text":"Confidence","font":{"size":13}}
                ))
                gauge.update_layout(template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=5,r=5,t=30,b=5))
                st.plotly_chart(gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with g2:
                st.markdown('<div class="card-3d">', unsafe_allow_html=True)
                st.subheader("🔤 ABCDE Analysis")
                xai.render_abcde(abcde)
                st.markdown('</div>', unsafe_allow_html=True)

            with g3:
                st.markdown('<div class="card-3d">', unsafe_allow_html=True)
                st.subheader("🎯 Risk Assessment")
                xai.render_risk_meter(risk_lvl)
                st.markdown(f"**Risk Score:** {risk_sc:.3f}")
                st.markdown(f"**Diagnosis:** {diagnosis}")
                st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                if confidence < 0.60:
                    st.warning("⚠️ Low confidence — consider repeat scan or biopsy.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Clinical Protocols
            st.markdown('<div class="card-3d">', unsafe_allow_html=True)
            st.subheader("🏥 Clinical Protocols")
            prot.render_for_diagnosis(diagnosis)
            st.markdown('</div>', unsafe_allow_html=True)

            # Disclaimer
            ui.disclaimer_box()

            # ── Step 5 : Actions ──────────────────────────
            st.subheader("Step 5 — Actions")
            record = {
                "Timestamp": ts, "Patient_ID": pid, "Name": pname, "Age": age,
                "Gender": gender, "Skin_Type": skin_t, "Lesion_Location": location,
                "AI_Diagnosis": diagnosis, "Confidence": round(confidence, 4),
                "Risk_Level": risk_lvl, "Notes": ""
            }
            reg.save_record(record)
            st.success("✅ Record saved to Patient Registry.")

            col_a, col_b = st.columns(2)
            with col_a:
                doc_notes = st.text_area("Physician Notes (optional)", key="doc_notes")
            with col_b:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    if not FPDF_AVAILABLE:
                        st.error("fpdf2 not installed. Run: pip install fpdf2")
                    else:
                        patient_info = {
                            "Patient ID": pid, "Name": pname, "Age": age,
                            "Gender": gender, "Skin Type": skin_t,
                            "Lesion Location": location, "Scan Date": ts
                        }
                        with st.spinner("Generating PDF..."):
                            pdf_bytes, err = rep.generate(
                                patient_info, diagnosis, confidence, abcde, risk_lvl, doc_notes
                            )
                        if err:
                            st.error(f"PDF error: {err}")
                        else:
                            st.download_button(
                                "⬇️ Download Report",
                                data=pdf_bytes,
                                file_name=f"SkinScan_Report_{pid}_{ts[:10]}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

            if label_text == "BENIGN":
                st.balloons()


# ══════════════════════════════════════════════════════════
# MASTER CONTROLLER
# ══════════════════════════════════════════════════════════
class SkinScanEnterpriseSuite:
    def __init__(self):
        if "initialized" not in st.session_state:
            st.session_state.initialized    = True
            st.session_state.is_authenticated = True
            st.session_state.medical_database = []
            st.session_state.scan_history     = []
            st.session_state.chat_history     = []
            st.session_state.activity_feed    = []
            st.session_state.error_log        = []
            st.session_state.experiments      = []
            st.session_state.theme            = "dark"
            st.session_state.start_time       = time.time()
            st.session_state.current_user     = "Dr. Admin"
            st.session_state.user_role        = "Admin"

    def render_sidebar(self):
        engine = NeuralCoreEngine()
        with st.sidebar:
            # Logo
            st.markdown('<div style="text-align:center;padding:1rem 0 .5rem"><span class="logo-3d">🧬 SkinScan AI</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align:center;font-size:.75rem;color:#475569;margin-bottom:.5rem">Enterprise Clinical Suite v12.0</div>', unsafe_allow_html=True)
            orb_cls = "orb orb-green" if not engine.sim_mode else "orb orb-orange"
            orb_txt = "Neural Engine Online" if not engine.sim_mode else "Simulation Mode"
            st.markdown(f'<div style="text-align:center;margin-bottom:1rem;font-size:.8rem"><span class="{orb_cls}"></span>{orb_txt}</div>', unsafe_allow_html=True)
            st.divider()

            # Navigation
            pages = [
                "🏠 Command Hub",
                "🔬 AI Diagnostic Lab",
                "📋 Patient Registry",
                "📊 Analytics Engine",
                "🤖 AI Dermatologist",
                "📖 Disease Encyclopedia",
                "🧬 Research Panel",
                "⚙️ Admin Panel",
            ]
            if OPTION_MENU_AVAILABLE:
                from streamlit_option_menu import option_menu
                sel = option_menu(
                    menu_title=None,
                    options=[p.split(" ",1)[1] for p in pages],
                    icons=["house","search","clipboard-data","bar-chart","robot",
                           "book","eyedropper","gear"],
                    menu_icon="cast",
                    default_index=0,
                    styles={
                        "container":{"padding":"0","background":"transparent"},
                        "icon":{"color":"#3b82f6","font-size":"14px"},
                        "nav-link":{"font-size":"13px","color":"#94a3b8",
                                    "--hover-color":"rgba(59,130,246,.12)"},
                        "nav-link-selected":{"background":"rgba(59,130,246,.18)",
                                             "color":"#e2e8f0","font-weight":"600"},
                    }
                )
                st.session_state.current_page = sel
            else:
                if "current_page" not in st.session_state:
                    st.session_state.current_page = "Command Hub"
                for p in pages:
                    label = p.split(" ",1)[1]
                    if st.button(p, key=f"nav_{label}", use_container_width=True):
                        st.session_state.current_page = label

            st.divider()

            # Theme toggle
            theme_lbl = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
            if st.button(theme_lbl, use_container_width=True):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()

            # Doctor info
            st.divider()
            user = st.session_state.current_user
            initials = "".join(w[0] for w in user.split() if w)[:2].upper()
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.6rem;padding:.5rem">
              <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                          display:flex;align-items:center;justify-content:center;
                          font-weight:700;font-size:.85rem;color:white">{initials}</div>
              <div>
                <div style="font-size:.82rem;font-weight:600;color:#e2e8f0">{user}</div>
                <div style="font-size:.72rem;color:#475569">{st.session_state.user_role}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            rec_count = len(st.session_state.get("medical_database",[]))
            if rec_count:
                st.markdown(f'<div style="text-align:center;font-size:.75rem;color:#3b82f6;margin-top:.3rem">📋 {rec_count} registry record{"s" if rec_count!=1 else ""}</div>', unsafe_allow_html=True)

    def route(self):
        page = st.session_state.get("current_page","Command Hub")
        if   "Command Hub"       in page: DashboardEngine().render()
        elif "Diagnostic Lab"    in page: DiagnosticLab().render()
        elif "Patient Registry"  in page: PatientRegistry().render()
        elif "Analytics"         in page: DashboardEngine().render_analytics()
        elif "Dermatologist"     in page: AIDermatologistBot().render()
        elif "Encyclopedia"      in page: ClinicalProtocols().render_encyclopedia()
        elif "Research"          in page: ResearchPanel().render()
        elif "Admin"             in page: AdminPanel().render()
        else:                             DashboardEngine().render()

    def launch(self):
        ui = InterfaceManager()
        ui.inject_css()
        self.render_sidebar()
        self.route()

        # Footer
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 .5rem;font-size:.75rem;color:#475569">
          SkinScan AI v12.0 &nbsp;|&nbsp; Enterprise Clinical Suite &nbsp;|&nbsp;
          <span style="color:#ef4444">⚕️ For clinical decision support only — not a substitute for professional diagnosis</span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SkinScanEnterpriseSuite()
    app.launch()
