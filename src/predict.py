import argparse
from pathlib import Path

import torch
from PIL import Image

from config import CLASSES as CLASS_NAMES
from config import DEFAULT_ARCHITECTURE, MODEL_PATH
from dataset import create_transform
from device import get_device
from model import SUPPORTED_ARCHITECTURES, create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faz predição de uma imagem usando o modelo treinado.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Caminho da imagem para classificação.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Caminho do checkpoint do modelo. Padrão: {MODEL_PATH}",
    )
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help=f"Arquitetura usada pelo checkpoint. Padrão: {DEFAULT_ARCHITECTURE}",
    )

    return parser.parse_args()


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    return prepare_image(image)


def prepare_image(image):
    transform = create_transform()
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def load_model(device, model_path=MODEL_PATH, architecture=DEFAULT_ARCHITECTURE):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}. "
            "Treine um modelo ou informe --model-path com um checkpoint válido."
        )

    model = create_model(pretrained=False, architecture=architecture)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = probabilities.max(dim=1)

    return predicted_index.item(), confidence.item(), probabilities[0].cpu()


def predict_image(model, image, device):
    image_tensor = prepare_image(image)

    return predict(model, image_tensor, device)


def print_prediction(image_path, predicted_index, confidence, probabilities):
    print(f"Caminho da imagem: {image_path}")
    print(f"Classe prevista: {CLASS_NAMES[predicted_index]}")
    print(f"Confiança: {confidence:.4f}")
    print("\nProbabilidade por classe:")

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name}: {probability.item():.4f}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    model_path = args.model_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    device = get_device()

    image_tensor = load_image(image_path)
    try:
        model = load_model(device, model_path, args.architecture)
    except FileNotFoundError as error:
        raise SystemExit(f"Erro: {error}") from None
    predicted_index, confidence, probabilities = predict(
        model,
        image_tensor,
        device,
    )

    print_prediction(image_path, predicted_index, confidence, probabilities)


if __name__ == "__main__":
    main()
