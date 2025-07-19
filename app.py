import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import shufflenet_v2_x0_5
import easyocr
from imutils.object_detection import non_max_suppression
import streamlit as st
import pandas as pd
import requests

# --- PAGE CONFIGURATION ----------------------------------------------------
st.set_page_config(

    page_title="House Number Augmentation",

    layout="wide",

)

# --- DEFAULTS & SESSION STATE ----------------------------------------------
DEFAULTS = {

    "min_conf": 0.6,

    "width_ths": 0.7,

    "link_threshold": 0.4,

    "pad_digits": 0,

    "pad_group": 0,

    "model_path": "BestV1.pth",

    "replace_text": "",

    "source": "Gallery",

    "gallery_choice": None,

}

GALLERY_URLS = [

    "https://hc-cdn.hel1.your-objectstorage.com/s/v3/940054f77057ff35f3121ab00e9dcc312417afbf_everbilt-house-numbers-30406-e1_600.png",
    
    "https://hc-cdn.hel1.your-objectstorage.com/s/v3/ccc04b4ab991100f313d9bec4de1745fb459cfd3_image-asset.png",
    
    "https://hc-cdn.hel1.your-objectstorage.com/s/v3/d848c3ba82aa7dba6fb6d4d5f1d9fc4cd239571a_image__3_.png",
    
    "https://hc-cdn.hel1.your-objectstorage.com/s/v3/d3b426345be184661bf9b905e8400f3097b3eeb4_image__5_.png",
    
    "https://hc-cdn.hel1.your-objectstorage.com/s/v3/e3d9497bff619aba6fb64c4c238759ce68defb43_image__7_.png",

]

def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# initialize session state
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- CACHING RESOURCES ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_reader():
    return easyocr.Reader(["en"], gpu=False)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device: str = "cpu"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = shufflenet_v2_x0_5(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

_norm_mean = [0.4377, 0.4438, 0.4728]
_norm_std  = [0.1980, 0.2010, 0.1970]
_val_tf = T.Compose([
    T.ToPILImage(),
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(_norm_mean, _norm_std),
])

@st.cache_resource(show_spinner=False)
def get_classifier(model_path: str):
    model = load_model(model_path)
    return model, _val_tf, next(model.parameters()).device

# --- OCR + DIGIT SPLITTING --------------------------------------------------
def detect_digit_regions(img, min_conf, width_ths, link_threshold, pad_digits):
    
    reader = load_reader()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = reader.readtext(
        rgb,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
        width_ths=width_ths,
        link_threshold=link_threshold,
    )
    boxes_text = []
    for bbox, text, conf in raw:
        if text.isdigit() and conf >= min_conf:
            pts = np.array(bbox).reshape(-1, 2)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            boxes_text.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), text))
    if not boxes_text:
        return []
    rects = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in boxes_text])
    pick = non_max_suppression(rects, overlapThresh=0.3)
    digit_boxes = []
    for x1, y1, x2, y2 in pick:
        w, h = x2 - x1, y2 - y1
        for (x, y, ww, hh, text) in boxes_text:
            if x == x1 and y == y1 and ww == w and hh == h:
                L = len(text)
                # split into single digits
                slices = []
                if L > 1:
                    cw = w / L
                    for i in range(L):
                        xi = int(x + i * cw)
                        wi = int(cw) if i < L - 1 else int(x + w - xi)
                        slices.append((xi, y, wi, h))
                else:
                    slices.append((x, y, w, h))
                # refine each
                for sx, sy, sw, sh in slices:
                    crop = img[sy:sy + sh, sx:sx + sw]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, bw = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    cnts, _ = cv2.findContours(
                        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        cx, cy, cw2, ch2 = cv2.boundingRect(c)
                        px1 = max(sx, sx + cx - pad_digits)
                        py1 = max(sy, sy + cy - pad_digits)
                        px2 = min(sx + sw, sx + cx + cw2 + pad_digits)
                        py2 = min(sy + sh, sy + cy + ch2 + pad_digits)
                        digit_boxes.append((px1, py1, px2 - px1, py2 - py1))
                    else:
                        digit_boxes.append((sx, sy, sw, sh))
                break
    return digit_boxes

# --- GROUP DETECTION --------------------------------------------------------
def detect_group_regions(img, min_conf, width_ths, link_threshold, pad_group):
    reader = load_reader()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = reader.readtext(
        rgb,
        allowlist="0123456789",
        detail=1,
        paragraph=False,
        width_ths=width_ths,
        link_threshold=link_threshold,
    )
    boxes = []
    for bbox, text, conf in raw:
        if text.isdigit() and conf >= min_conf:
            pts = np.array(bbox).reshape(-1, 2)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    if not boxes:
        return []
    rects = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])
    pick = non_max_suppression(rects, overlapThresh=0.3)
    H, W = img.shape[:2]
    group_boxes = []
    for x1, y1, x2, y2 in pick:
        ex1 = np.clip(x1 - pad_group, 0, W)
        ey1 = np.clip(y1 - pad_group, 0, H)
        ex2 = np.clip(x2 + pad_group, 0, W)
        ey2 = np.clip(y2 + pad_group, 0, H)
        group_boxes.append((ex1, ey1, ex2 - ex1, ey2 - ey1))
    return group_boxes

# --- CLASSIFICATION ---------------------------------------------------------
def classify_regions(img, boxes, model, tf, device):
    results = []
    with torch.no_grad():
        for idx, (x, y, w, h) in enumerate(boxes):
            crop = img[y : y + h, x : x + w]
            inp = tf(crop).unsqueeze(0).to(device)
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            results.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "digit": int(probs.argmax()),
                    "confidence": float(probs.max()),
                }
            )
    return results

# --- DRAWING UTILITIES ------------------------------------------------------
def draw_annotations(img, annos, color=(0, 255, 0)):
    out = img.copy()
    for a in annos:
        x, y, w, h = a["x"], a["y"], a["w"], a["h"]
        lbl = f'{a["digit"]} ({a["confidence"]:.2f})'
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(
            lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(out, (x, y - th - 4), (x + tw, y), color, -1)
        cv2.putText(
            out,
            lbl,
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
    return out

# --- MAIN APP ---------------------------------------------------------------
def main():
    st.title("🏠 🔢 House Number Augmentation")

    # instructions
    st.markdown(
        """
    <div style="background: #232526; border-radius: 12px; padding: 2rem 1.5rem; margin-bottom: 1.5rem; color: #fff; box-shadow: 0 4px 24px rgba(0,0,0,0.15);">
        <h2 style="font-family: 'Segoe UI',sans-serif; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.5em;">
            <span style="color:#fff;">How to Use This App</span>
        </h2>
        <ol style="font-size: 1.1em; line-height: 1.7;">
            <li><b>Pick an image</b> <span style="color:#00e6d3;"></span> — upload your own or grab one from the gallery.</li>
            <li><b>Tweak to your desire</b> <span style="color:#ffb347;"></span> — tweak OCR confidence, digit splitting, and padding.</li>
            <li><b>Change Model( Only 2; BestV1.pth & BestV2.pth )</b> <span style="color:#f67280;"></span> — make sure either <code>BestV1.pth</code> or <code>BestV2.pth</code> is selected in the sidebar.</li>
            <li><b>Detect &amp; classify</b> <span style="color:#6a89cc;"></span> — see each digit's bounding box and prediction label in real time.</li>
            <li><b>Augment it</b> <span style="color:#38ada9;"></span> — enter text to erase and overlay new digits on the image.</li>
            <li><b>Reset</b> <span style="color:#f8c291;"></span> — click <b>Reset all to defaults</b> to start fresh.</li>
        </ol>
       
    </div>
    """,
        unsafe_allow_html=True,
    )

    # sidebar controls

    with st.sidebar:
        if st.button("Reset all to defaults"):
            reset_defaults()

        source = st.radio("Image source", ["Upload", "Gallery"], key="source")
        min_conf = st.slider(
            "Min OCR confidence", 0.1, 0.99, key="min_conf", step=0.05
        )
        width_ths = st.slider(
            "OCR width_ths", 0.1, 1.0, key="width_ths", step=0.05
        )
        link_threshold = st.slider(
            "OCR link_threshold", 0.1, 1.0, key="link_threshold", step=0.05
        )
        pad_digits = st.slider(
            "Digit pad (px)", 0, 20, key="pad_digits", step=1
        )
        pad_group = st.slider(
            "Group pad (px)", -20, 20, key="pad_group", step=1
        )
        model_path = st.text_input(
            "Model weights path", key="model_path"
        )
        replace_text = st.text_input(
            "Replacement text", key="replace_text"
        )

        st.markdown("---")
        
        img = None
        if source == "Upload":
            uploaded = st.file_uploader(
                "Upload JPG/PNG", type=["jpg", "jpeg", "png"]
            )
            if uploaded:
                with st.spinner("Loading uploaded image…"):
                    data = np.frombuffer(uploaded.read(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            st.markdown("#### Select from gallery:")
            cols = st.columns(len(GALLERY_URLS))
            for i, url in enumerate(GALLERY_URLS):
                with cols[i]:
                    st.image(url, use_container_width=True)
                    if st.button("Select", key=f"gal_{i}"):
                        st.session_state.gallery_choice = url
            if st.session_state.gallery_choice:
                with st.spinner("Fetching gallery image…"):
                    r = requests.get(st.session_state.gallery_choice)
                    buf = np.frombuffer(r.content, np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img is None:
        st.info("Please upload or select an image to proceed.")
        return

    # show original
    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, )

    # detect digits
    with st.spinner("Detecting digits…"):
        digit_boxes = detect_digit_regions(
            img, min_conf, width_ths, link_threshold, pad_digits
        )

    # classify
    with st.spinner("Classifying digits…"):
        model, tf, device = get_classifier(model_path)
        annos = classify_regions(img, digit_boxes, model, tf, device)

    # draw results
    out = draw_annotations(img, annos)
    st.subheader(f"Detected & Classified ({len(annos)} digits)")
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)

    # show table
    st.subheader("Per-Digit Predictions")
    df = pd.DataFrame(annos)[["x", "y", "w", "h", "digit", "confidence"]]
    st.dataframe(df.style.format({"confidence": "{:.2f}"}), use_container_width=True)

    # replacement
    if replace_text:
        with st.spinner("Erasing original number groups…"):
            group_boxes = detect_group_regions(
                img, min_conf, width_ths, link_threshold, pad_group
            )
        with st.spinner("Overlaying replacement text…"):
            repl = img.copy()
            H, W = img.shape[:2]
            for x, y, w, h in group_boxes:
                bw = 3
                x0, y0 = max(0, x - bw), max(0, y - bw)
                x1, y1 = min(W, x + w + bw), min(H, y + h + bw)
                region = img[y0:y1, x0:x1]
                mask = np.zeros(region.shape[:2], bool)
                mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = True
                border = region[~mask]
                color = [255, 255, 255]
                if border.size:
                    color = np.median(
                        border.reshape(-1, 3), axis=0
                    ).astype(np.uint8).tolist()
                repl[y : y + h, x : x + w] = color

                # compute text placement
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw0, th0), _ = cv2.getTextSize(
                    replace_text, font, 1.0, 2
                )
                if tw0 and th0:
                    scale = min(w / tw0 * 0.8, h / th0 * 0.8)
                    thk = max(1, int(scale * 2))
                    (tw, th), _ = cv2.getTextSize(
                        replace_text, font, scale, thk
                    )
                    tx, ty = x + (w - tw) // 2, y + (h + th) // 2
                    cv2.putText(
                        repl,
                        replace_text,
                        (tx, ty),
                        font,
                        scale,
                        (255, 255, 255),
                        thk + 2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        repl,
                        replace_text,
                        (tx, ty),
                        font,
                        scale,
                        (0, 0, 0),
                        thk,
                        cv2.LINE_AA,
                    )

        st.subheader("With Replacement Overlay")
        st.image(cv2.cvtColor(repl, cv2.COLOR_BGR2RGB), use_container_width=True)


if __name__ == "__main__":
    main()
