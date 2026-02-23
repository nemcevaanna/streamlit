import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("Классификация изображений")
st.write("Вы можете загрузить изображение или нарисовать его.")

# --- Выбор способа ввода ---
mode = st.radio("Выберите способ ввода изображения:", ["Загрузить изображение", "Рисовать на холсте"])

image = None

if mode == "Загрузить изображение":
    uploaded_file = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)

elif mode == "Рисовать на холсте":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        st.image(image, caption="Нарисованное изображение", use_container_width=False)

# --- Отправка изображения на API ---
if image is not None:
    st.subheader("Отправка на классификацию")

    # Предобработка
    image = image.resize((224, 224))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    if st.button("Классифицировать"):
        with st.spinner("Отправка на сервер..."):
            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(API_URL, files=files)
        
        if response.ok:
            result = response.json()
            print(result)
            pred_class = result["predicted_class"]
            probabilities = result["probabilities"]

            st.success(f"Предсказанный класс: **{pred_class}**")

            # --- Визуализация вероятностей ---
            st.subheader("Распределение вероятностей")
            labels = list(probabilities.keys())
            values = list(probabilities.values())

            fig, ax = plt.subplots()
            ax.barh(labels, values, color="skyblue")
            ax.set_xlabel("Вероятность")
            st.pyplot(fig)
        else:
            st.error("Ошибка при обращении к API.")

