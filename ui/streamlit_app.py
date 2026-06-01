import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.onnx_predictor import ONNXEmotionPredictor, EMOTION_LABELS

st.set_page_config(
    page_title="DeepXpress — Emotion Recognition",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 DeepXpress — Facial Emotion Recognition")
st.markdown("Upload an image to detect facial emotions.")

predictor = ONNXEmotionPredictor()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing emotions..."):
            faces, emotions, confidences = predictor.predict_emotions(image)

        if not faces:
            st.warning("No faces detected.")
        else:
            st.success(f"Detected {len(faces)} face(s).")

            result_img = image_rgb.copy()
            for face, emotion, confidence in zip(faces, emotions, confidences):
                x1, y1, w, h = [int(v) for v in face[:4]]
                label = f"{emotion} ({confidence:.1%})"

                import matplotlib.patches as patches
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(result_img)
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 8, label,
                    fontsize=10, color="lime",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
                )
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(
                    f"- **{emotion}**: {confidence:.1%} confidence "
                    f"(bbox: ({x1}, {y1}), {w}×{h})"
                )
