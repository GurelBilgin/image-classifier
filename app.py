# -*- coding: utf-8 -*-
"""
Streamlit tabanlı hayvan görüntü sınıflandırma uygulaması. 
Bu uygulama, PyTorch ile eğitilmiş ResNet18 modelini kullanarak 
kullanıcının yüklediği hayvan görsellerini sınıflandırır. 
Sonuçlar kullanıcı dostu bir web arayüzünde sunulur.
"""

import streamlit as st
from streamlit.components.v1 import html
import torch
from PIL import Image
import os
import json
import time
import base64
from io import BytesIO
from src.model_utils import load_model
from src.data_utils import prepare_image
from src.translate import translate

# -----------------------------
# SAYFA AYARI
# -----------------------------
st.set_page_config(
    page_title=" Yapay Zeka Destekli Hayvan Görüntü Sınıflandırıcı",
    page_icon="🐾",
    layout="centered"
)

# -----------------------------
# STIL AYARLARI
# -----------------------------
st.markdown(
    """
    <style>
    .stFileUploader > label {
        font-size:18px;
        font-weight:bold;
        color:#ffffff;
    }
    .expander-header {
        background-color:#111;
        color:white;
        padding:8px;
        border-radius:5px;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# BAŞLIK VE AÇIKLAMA
# -----------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h2 style="color:white; font-size:28px; margin:0;">🐾  Yapay Zeka Destekli Hayvan Görüntü Sınıflandırıcı</h2>
        <p style="color:white; font-size:18px; margin:0;">
            Bu uygulama, hayvan görselinizi yapay zekâ ile değerlendirerek tahmin ettiği sınıfı ve güven oranını ekrana getirir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.divider()

# -----------------------------
# NASIL ÇALIŞIR
# -----------------------------
st.markdown(
    """
    <div style="font-size:20px; line-height:1.5;">
    <b>Nasıl Çalışır?</b><br>
    1️⃣ Bilgisayarınızdan bir hayvan görseli yükleyin<br>
    2️⃣ <b>Tahmin Et</b> butonuna basın<br>
    3️⃣ Yapay zekânın tahminini ve güven oranını görüntüleyin
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# DESTEKLENEN HAYVANLAR (EXPANDER)
# -----------------------------
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
CLASSES_PATH = "trained_models/classes.json"
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    with st.expander("🎯 Desteklenen Hayvan Türleri", expanded=False):
        for c in classes:
            tr = translate.get(c, c)
            st.markdown(f"<p style='margin:1;font-size:18px;'>• {c} ({tr})</p>", unsafe_allow_html=True)
st.divider()

# -----------------------------
# MODEL YÜKLE
# -----------------------------
MODEL_PATH = "trained_models/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, len(classes), device)
    model.eval()
else:
    st.error("Model dosyası bulunamadı. Önce train.py çalıştırın.")
    st.stop()

# -----------------------------
# DOSYA YÜKLEME
# -----------------------------
st.markdown(
    "<p style='font-size:20px; color:white; font-weight:bold; margin:0;'>📤 Bir hayvan resmi yükleyin</p>",
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# -----------------------------
# TAHMİN
# -----------------------------
MIN_CONFIDENCE = 0.85
MAX_WIDTH = 500

def pil_to_bytes(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_b64 = pil_to_bytes(image)

    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:20px;">
            <img src="data:image/png;base64,{img_b64}" style="max-width:{MAX_WIDTH}px; width:100%; height:auto;">
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔍 Tahmin Et", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        status.text("🔄 Görsel hazırlanıyor...")
        time.sleep(0.3)
        progress.progress(30)

        input_tensor = prepare_image(image).to(device)

        status.text("🧠 Model tahmin yapıyor...")
        time.sleep(0.5)
        progress.progress(70)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        status.text("✅ Sonuç hazırlanıyor...")
        time.sleep(0.3)
        progress.progress(100)

        progress.empty()
        status.empty()

        class_name = classes[pred.item()]
        tr_name = translate.get(class_name, class_name)
        confidence = conf.item()

        if confidence >= MIN_CONFIDENCE:
            st.success(
                f"**Tahmin:** {class_name} ({tr_name})\n\n"
                f"**Tahmin Güveni:** %{confidence * 100:.2f}"
            )
        else:
            st.error(
                "❌ Model bu görsel için yeterince emin değil.\n"
                "Lütfen desteklenen hayvan türlerinden birini yükleyin."
            )

# -----------------------------
# MODEL BİLGİSİ & NASIL ÇALIŞIR
# -----------------------------
st.divider()
with st.expander("💡 Model Nasıl Çalışır?", expanded=False):
    st.markdown(
        """
        <div style="background-color:#111; color:white; padding:10px; border-radius:5px; font-size:18px;">
        Bu uygulama, PyTorch ile eğitilmiş <b>ResNet18</b> modelini kullanır. 
        Yüklediğiniz görsel önce normalize edilir ve modele uygun tensor'a dönüştürülür. 
        Model tahminini yaptıktan sonra softmax ile olasılıkları hesaplar ve en yüksek olasılıklı sınıf kullanıcıya gösterilir. 
        %90'dan düşük güven oranında tahmin gösterilmez.
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown(
    '<p style="color:white; font-size:18px; margin:0;">Model: ResNet18 | Framework: PyTorch | Arayüz: Streamlit</p>',
    unsafe_allow_html=True
)
