import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import CLASS_NAMES, MODEL_PATH, load_model, predict_image


@st.cache_resource
def get_model(device_name):
    device = torch.device(device_name)

    return load_model(device)


def load_image_from_url(image_url):
    request = Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(request, timeout=10) as response:
        image_bytes = response.read()

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def show_prediction(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(str(device))
    predicted_index, confidence, probabilities = predict_image(
        model,
        image,
        device,
    )

    st.subheader("Resultado")
    st.write(f"Classe prevista: **{CLASS_NAMES[predicted_index]}**")
    st.write(f"Confiança: **{confidence:.2%}**")

    st.write("Probabilidades:")
    st.write(f"Chihuahua: **{probabilities[0].item():.2%}**")
    st.write(f"Muffin: **{probabilities[1].item():.2%}**")


def main():
    st.set_page_config(
        page_title="Chihuahua or Muffin",
    )

    st.title("Chihuahua or Muffin")
    st.write(
        "Envie uma imagem para o modelo classificar se ela parece um "
        "chihuahua ou um muffin."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Modelo treinado não encontrado. Execute o treino primeiro para "
            "gerar `models/best_model.pth`."
        )
        return

    upload_tab, link_tab = st.tabs(["Upload", "Link"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagem enviada", use_container_width=True)
            show_prediction(image)

    with link_tab:
        image_url = st.text_input("Cole o link da imagem")

        if image_url:
            try:
                image = load_image_from_url(image_url)
                st.image(image, caption="Imagem do link", use_container_width=True)
                show_prediction(image)
            except (URLError, TimeoutError, UnidentifiedImageError, OSError):
                st.error(
                    "Não foi possível carregar a imagem pelo link. "
                    "Verifique se a URL aponta diretamente para uma imagem."
                )


if __name__ == "__main__":
    main()
