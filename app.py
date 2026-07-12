"""
Satellite Image Land-Use Classifier & Temporal Change Detector
Streamlit Dashboard — runs fully offline after setup.

Tabs:
  1. Classify Single Image
  2. Compare T1 / T2 (Change Detection)
  3. GradCAM Visualization  [Bonus A]
  4. About

Usage:
    streamlit run app.py
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "src"
CKPT_DIR = BASE_DIR / "outputs" / "checkpoints"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Land-Use Classifier",
    page_icon="🛰️",
    layout="wide",
)

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    class_path = SRC_DIR / "class_names.json"
    thr_path   = SRC_DIR / "change_threshold.json"

    # Fallback class names (EuroSAT standard) if notebook 01 hasn't run yet
    default_classes = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]
    class_names = default_classes
    if class_path.exists():
        try:
            class_names = json.loads(class_path.read_text())
        except Exception:
            class_names = default_classes

    threshold = 0.70
    if thr_path.exists():
        try:
            threshold = json.loads(thr_path.read_text()).get("threshold", 0.70)
        except Exception:
            threshold = 0.70

    return class_names, threshold

CLASS_NAMES, DEFAULT_THRESHOLD = load_config()
NUM_CLASSES = len(CLASS_NAMES)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = models.resnet18(weights=None)
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base.fc.in_features, NUM_CLASSES)
    )
    ckpt = CKPT_DIR / "resnet18_best.pt"
    loaded = False
    if ckpt.exists():
        try:
            base.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            loaded = True
        except Exception as e:
            st.warning(f"Could not load checkpoint: {e}")
    base.eval()

    extractor = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
        base.avgpool
    )
    return base, extractor, loaded

CLASSIFIER, EXTRACTOR, MODEL_LOADED = load_model()

# ── Transform ─────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess(pil_img):
    return TF(pil_img.convert("RGB")).unsqueeze(0)

# ── Inference helpers ─────────────────────────────────────────────────────────
def classify(pil_img):
    with torch.no_grad():
        probs = F.softmax(CLASSIFIER(preprocess(pil_img)), dim=1)[0].numpy()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def get_embedding(pil_img):
    with torch.no_grad():
        return EXTRACTOR(preprocess(pil_img)).flatten().numpy()

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def patch_diff_heatmap(img1, img2, patch_size=8):
    a1 = np.array(img1.resize((64, 64))).astype(float) / 255.0
    a2 = np.array(img2.resize((64, 64))).astype(float) / 255.0
    diff = np.abs(a1 - a2).mean(axis=2)
    n = 64 // patch_size
    heat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            heat[i, j] = diff[i*patch_size:(i+1)*patch_size,
                               j*patch_size:(j+1)*patch_size].mean()
    return heat

def heatmap_overlay(base_img, heat, alpha=0.45):
    img_arr = np.array(base_img.resize((224, 224))).astype(float) / 255.0
    cam_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((224, 224))
    cam_arr = np.array(cam_img).astype(float) / 255.0
    colored = cm.hot(cam_arr)[:, :, :3]
    blended = (1 - alpha) * img_arr + alpha * colored
    return np.clip(blended, 0, 1)

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts  = None
        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def generate(self, tensor, class_idx=None):
        self.model.eval()
        t = tensor.requires_grad_(True)
        logits = self.model(t)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().detach().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx

    def remove(self):
        self._fh.remove()
        self._bh.remove()

def run_gradcam(pil_img, class_idx=None):
    """Run GradCAM and return (overlay_array, heatmap_array, predicted_class_idx)."""
    gc = GradCAM(CLASSIFIER, CLASSIFIER.layer4[-1])
    tensor = preprocess(pil_img)
    cam, pred_idx = gc.generate(tensor, class_idx)
    gc.remove()
    img_arr = np.array(pil_img.resize((224, 224))).astype(float) / 255.0
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
    ).astype(float) / 255.0
    colored = cm.jet(cam_resized)[:, :, :3]
    blended = np.clip(0.55 * img_arr + 0.45 * colored, 0, 1)
    return blended, cam, pred_idx

# ═════════════════════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════════════════════
st.title("🛰️ Satellite Land-Use Classifier & Change Detector")
st.caption("EuroSAT · ResNet-18 Transfer Learning · Cosine-Similarity Change Detection")

if not MODEL_LOADED:
    st.warning(
        "⚠️ **Model checkpoint not found** (`outputs/checkpoints/resnet18_best.pt`).  \n"
        "Run notebooks **01 → 02 → 03** in order first.  \n"
        "Currently using random weights for demo purposes."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
op_mode = st.sidebar.radio(
    "Change detection operating point",
    ["Custom", "High Recall (0.80)", "Balanced (0.70)", "High Precision (0.55)"],
)
if op_mode == "High Recall (0.80)":
    threshold = 0.80
elif op_mode == "Balanced (0.70)":
    threshold = 0.70
elif op_mode == "High Precision (0.55)":
    threshold = 0.55
else:
    threshold = st.sidebar.slider(
        "Custom threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01
    )

st.sidebar.markdown(f"**Active threshold:** `{threshold:.2f}`")
st.sidebar.markdown("---")
st.sidebar.markdown("**EuroSAT Classes**")
for c in CLASS_NAMES:
    st.sidebar.markdown(f"- {c}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏷️ Classify", "🔄 Change Detection", "🔥 GradCAM", "📊 About"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Single image classification
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Single Tile Classification")
    uploaded = st.file_uploader("Upload a satellite tile", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img.resize((256, 256)), caption="Uploaded tile")

        with col2:
            pred_class, confidence, all_probs = classify(img)
            st.metric("Predicted Class", pred_class)
            st.metric("Confidence", f"{confidence*100:.1f}%")

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["tomato" if c == pred_class else "steelblue" for c in CLASS_NAMES]
            ax.barh(CLASS_NAMES, all_probs * 100, color=colors)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Class Probability Distribution")
            ax.set_xlim(0, 100)
            for i, v in enumerate(all_probs):
                if v > 0.01:
                    ax.text(v*100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — T1 / T2 change detection
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Temporal Change Detection")
    st.markdown("Upload **before (T1)** and **after (T2)** satellite tiles.")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        up_t1 = st.file_uploader("📷 T1 — Before", type=["jpg","jpeg","png"], key="t1")
    with col_t2:
        up_t2 = st.file_uploader("📷 T2 — After",  type=["jpg","jpeg","png"], key="t2")

    if up_t1 and up_t2:
        img_t1 = Image.open(up_t1).convert("RGB")
        img_t2 = Image.open(up_t2).convert("RGB")

        cls_t1, conf_t1, probs_t1 = classify(img_t1)
        cls_t2, conf_t2, probs_t2 = classify(img_t2)
        emb_t1 = get_embedding(img_t1)
        emb_t2 = get_embedding(img_t2)
        sim     = cosine_similarity(emb_t1, emb_t2)
        changed = sim < threshold

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("T1 Prediction", cls_t1, f"{conf_t1*100:.1f}%")
        m2.metric("T2 Prediction", cls_t2, f"{conf_t2*100:.1f}%")
        m3.metric("Cosine Similarity", f"{sim:.4f}",
                  delta="CHANGED 🔴" if changed else "UNCHANGED 🟢",
                  delta_color="inverse" if changed else "normal")

        if changed:
            st.error(f"🔴 **CHANGE DETECTED** — similarity {sim:.4f} < threshold {threshold:.2f}")
        else:
            st.success(f"🟢 **NO CHANGE** — similarity {sim:.4f} ≥ threshold {threshold:.2f}")

        # Images + heatmap row
        heat    = patch_diff_heatmap(img_t1, img_t2)
        heat_n  = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        overlay = heatmap_overlay(img_t2, heat_n)

        c1, c2, c3 = st.columns(3)
        c1.image(img_t1.resize((256, 256)),
                 caption=f"T1 · {cls_t1} ({conf_t1*100:.1f}%)")
        c2.image(img_t2.resize((256, 256)),
                 caption=f"T2 · {cls_t2} ({conf_t2*100:.1f}%)")
        c3.image((overlay*255).astype(np.uint8),
                 caption="Change heatmap (red = high diff)")

        # Probability bars
        st.markdown("---")
        st.subheader("Class Probabilities")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
        ax1.barh(CLASS_NAMES, probs_t1*100,
                 color=["tomato" if c==cls_t1 else "steelblue" for c in CLASS_NAMES])
        ax1.set_title("T1 Probabilities")
        ax1.set_xlabel("Probability (%)")
        ax1.set_xlim(0, 100)
        ax2.barh(CLASS_NAMES, probs_t2*100,
                 color=["tomato" if c==cls_t2 else "steelblue" for c in CLASS_NAMES])
        ax2.set_title("T2 Probabilities")
        ax2.set_xlabel("Probability (%)")
        ax2.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Multi-threshold table (Bonus B)
        st.markdown("---")
        st.subheader("Multi-Threshold Analysis")
        import pandas as pd
        rows = []
        for name, thr in [("High Recall", 0.80), ("Balanced", 0.70), ("High Precision", 0.55)]:
            rows.append({
                "Operating Point": name,
                "Threshold": thr,
                "Similarity": f"{sim:.4f}",
                "Decision": "CHANGED 🔴" if sim < thr else "UNCHANGED 🟢"
            })
        st.table(pd.DataFrame(rows))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GradCAM  [Bonus A]
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🔥 GradCAM — What Did the Model Look At?")
    st.markdown(
        "Upload a satellite tile to see **which pixels drove the classification**. "
        "Red/yellow = high activation, blue = ignored."
    )

    gc_upload = st.file_uploader(
        "Upload tile for GradCAM", type=["jpg","jpeg","png"], key="gc"
    )

    if gc_upload:
        gc_img = Image.open(gc_upload).convert("RGB")

        with st.spinner("Running GradCAM..."):
            pred_cls, confidence, all_probs = classify(gc_img)
            overlay_arr, cam_raw, pred_idx  = run_gradcam(gc_img)

        c1, c2, c3 = st.columns(3)
        c1.image(gc_img.resize((224, 224)), caption="Original image")

        # Pure heatmap
        fig_h, ax_h = plt.subplots(figsize=(3, 3))
        ax_h.imshow(cam_raw, cmap="jet")
        ax_h.axis("off")
        ax_h.set_title("GradCAM heatmap", fontsize=9)
        plt.tight_layout(pad=0)
        c2.pyplot(fig_h)
        plt.close()

        c3.image((overlay_arr * 255).astype(np.uint8),
                 caption=f"Overlay — Pred: {pred_cls} ({confidence*100:.1f}%)")

        st.markdown("---")
        st.subheader("Try a different target class")
        target_cls = st.selectbox(
            "Generate GradCAM for class:", CLASS_NAMES, index=pred_idx
        )
        if target_cls != pred_cls:
            with st.spinner(f"Running GradCAM for '{target_cls}'..."):
                t_idx = CLASS_NAMES.index(target_cls)
                ov2, cam2, _ = run_gradcam(gc_img, class_idx=t_idx)

            cc1, cc2 = st.columns(2)
            fig_c2, ax_c2 = plt.subplots(figsize=(3, 3))
            ax_c2.imshow(cam2, cmap="jet")
            ax_c2.axis("off")
            plt.tight_layout(pad=0)
            cc1.pyplot(fig_c2)
            plt.close()
            cc2.image((ov2 * 255).astype(np.uint8),
                      caption=f"Overlay for '{target_cls}'")

        # Probability bar
        st.markdown("---")
        fig_p, ax_p = plt.subplots(figsize=(8, 3))
        colors_p = ["tomato" if c == pred_cls else "steelblue" for c in CLASS_NAMES]
        ax_p.barh(CLASS_NAMES, all_probs * 100, color=colors_p)
        ax_p.set_xlabel("Probability (%)")
        ax_p.set_xlim(0, 100)
        ax_p.set_title("Class Probabilities")
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close()

        st.info(
            "**Interpretation:** GradCAM highlights spatial regions the model considers "
            "most discriminative for the predicted class. "
            "For land-use classification, activations typically concentrate on "
            "texture patterns, field boundaries, road networks, or water edges."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("About This Project")
    st.markdown("""
    ### Satellite Image Land-Use Classifier & Temporal Change Detector

    | Component | Details |
    |-----------|---------|
    | **Dataset** | EuroSAT — 27,000 Sentinel-2 RGB tiles, 10 land-use classes |
    | **Backbone** | ResNet-18 pretrained on ImageNet (torchvision) |
    | **Fine-tuning** | Phase 1: FC head only, 3 epochs, LR=1e-3 · Phase 2: layer3+layer4+FC, 5 epochs, LR=1e-4 |
    | **Change detection** | 512-dim cosine similarity, threshold via Youden's J on ROC |
    | **Dashboard** | Streamlit — runs 100% locally, no internet after setup |
    | **API / Libraries** | PyTorch, torchvision, scikit-learn, Streamlit, matplotlib, Pillow |

    ---
    ### 🎁 Bonus Features in This App
    - **Bonus A** — GradCAM visualization (Tab 3): see which pixels the model focuses on
    - **Bonus B** — Multi-threshold toggle in Tab 2: High Recall / Balanced / High Precision
    - **Bonus C** — t-SNE & UMAP comparisons (Notebook 06)
    - **Bonus D** — Class imbalance experiment (Notebook 07)

    ---
    ### Notebooks
    | # | Notebook | Purpose |
    |---|----------|---------|
    | 01 | data_pipeline | EuroSAT download, spatial block split, visualization |
    | 02 | baseline_cnn | 3-layer scratch CNN — performance floor |
    | 03 | transfer_learning | ResNet-18 two-phase fine-tuning |
    | 04 | change_detection | Embeddings, ROC curve, change heatmaps |
    | 05 | bonus_gradcam | GradCAM on all 10 classes |
    | 06 | bonus_tsne_umap | t-SNE & UMAP embedding comparison |
    | 07 | bonus_imbalance | Class imbalance experiment & mitigation |

    ---
    ### Run Order
    ```
    01_data_pipeline.ipynb     ← Run FIRST (downloads & splits EuroSAT)
    02_baseline_cnn.ipynb      ← Run SECOND
    03_transfer_learning.ipynb ← Run THIRD (creates resnet18_best.pt)
    04_change_detection.ipynb  ← Needs checkpoint from 03
    05_bonus_gradcam.ipynb     ← Needs checkpoint from 03
    06_bonus_tsne_umap.ipynb   ← Needs checkpoints from 02 & 03
    07_bonus_imbalance.ipynb   ← Independent (uses pretrained weights)

    streamlit run app.py       ← Run dashboard (needs 01 + 03 done)
    ```
    """)

    col1, col2 = st.columns(2)
    col1.markdown("**Classes (EuroSAT)**")
    for c in CLASS_NAMES:
        col1.markdown(f"- {c}")
    col2.markdown("**Evaluation Metrics**")
    for m in ["Per-class F1", "Macro-F1", "Confusion Matrix",
              "ROC-AUC", "Cosine Similarity", "Silhouette Score"]:
        col2.markdown(f"- {m}")
