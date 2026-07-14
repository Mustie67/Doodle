# app.py
"""
Streamlit chalkboard doodle-guessing app.

- Uses streamlit-drawable-canvas for drawing
- Loads model from model/doodle_cnn.h5
- Predicts after each stroke (on mouse up)
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "doodle_cnn.h5"
CSS_PATH = ROOT / "style.css"
CATEGORIES = ["cat", "house", "tree", "car", "fish", "star", "umbrella", "banana", "bicycle", "clock"]

st.set_page_config(page_title="Chalkboard Doodle AI", layout="wide", initial_sidebar_state="auto")

# inject CSS
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# load model if available
model = None
try:
    if MODEL_PATH.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH))
    else:
        st.warning("Trained model not found at model/doodle_cnn.h5 — the app will show placeholder predictions. Run model/train_model.py to create the model.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# header with hand-drawn title
st.markdown('<div class="chalk-title">Doodle Chalk AI</div>', unsafe_allow_html=True)
st.markdown('<div class="chalk-sub">Draw on the board — predictions after each stroke</div>', unsafe_allow_html=True)

# layout
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # transparent fill
        stroke_width=8,
        stroke_color="#F5F3E7",  # chalk-white
        background_color="#1B2E23",  # board color
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True
    )

    erase = st.button("Erase")
    if erase:
        # create a simple wipe animation (progressive overlay)
        for i in range(10):
            st.session_state["canvas_data"] = None
            st.experimental_rerun()

with col2:
    st.markdown("### Top guesses")
    guesses_placeholder = st.empty()

# helper: convert canvas image to 28x28 grayscale
def image_to_model_input(pil_im):
    # pil_im: RGBA or RGB or L
    im = pil_im.convert("L")  # grayscale
    # invert: canvas background is dark, drawing is light; model trained on white strokes on black background
    im = ImageOps.invert(im)
    im = im.resize((28,28), Image.ANTIALIAS)
    arr = np.asarray(im).astype("float32") / 255.0
    arr = arr.reshape((1,28,28,1))
    return arr

def predict_top3(img_arr):
    if model is None:
        # placeholder random-ish predictions
        probs = np.random.rand(len(CATEGORIES))
        probs = probs / probs.sum()
    else:
        probs = model.predict(img_arr)[0]
    top_idx = probs.argsort()[::-1][:3]
    return [(CATEGORIES[i], float(probs[i])) for i in top_idx]

# react to canvas updates
if canvas_result.image_data is not None:
    # convert to PIL
    im = Image.fromarray((canvas_result.image_data).astype('uint8'), mode="RGBA")
    # find bounding box of strokes to crop (optional)
    gray = im.convert("L")
    bbox = gray.getbbox()
    if bbox:
        cropped = im.crop(bbox).convert("RGBA")
    else:
        cropped = im
    model_input = image_to_model_input(cropped)
    top3 = predict_top3(model_input)
else:
    top3 = []

# show guesses
if top3:
    with guesses_placeholder:
        for name, p in top3:
            pct = f"{p*100:5.1f}%"
            st.markdown(f"<div class='guess-row'><div class='guess-name'>{name}</div><div class='guess-bar'><div class='guess-fill' style='width:{p*100}%;'></div></div><div class='guess-pct'>{pct}</div></div>", unsafe_allow_html=True)
else:
    guesses_placeholder.markdown('<div class="chalk-prompt">Draw something (e.g., a car or a tree) — I\'ll guess as you draw.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Palette: board #1B2E23, chalk white #F5F3E7, dusty yellow #E8C468, soft coral #E88D67, sage #7A9B87")
