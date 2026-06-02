import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    CLASSES,
    HARD_CASES_DATA_DIR,
    IMAGE_EXTENSIONS,
    MODEL_HISTORY_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SPLITS,
)
from device import get_device
from predict import CLASS_NAMES, load_model, predict_image


PAGES = [
    "Início",
    "Inferência",
    "Dataset",
    "Dashboard de Avaliação",
    "Histórico de Avaliações",
    "Histórico de Modelos",
    "Casos Ambíguos",
]


def apply_sidebar_styles():
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div.stButton > button {
            background: transparent;
            border: 0;
            border-radius: 6px;
            color: inherit;
            font-size: 1.08rem;
            font-weight: 600;
            justify-content: flex-start;
            margin: 0.18rem 0;
            padding: 0.7rem 0.85rem;
            text-align: left;
            width: 100%;
        }

        div[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255, 75, 75, 0.12);
            color: rgb(255, 75, 75);
        }

        div[data-testid="stSidebar"] div.stButton > button:focus {
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation():
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = PAGES[0]

    with st.sidebar:
        for page in PAGES:
            label = page

            if page == st.session_state.selected_page:
                label = f"▸ {page}"

            if st.button(label, key=f"nav_{page}"):
                st.session_state.selected_page = page
                st.rerun()

    return st.session_state.selected_page


@st.cache_resource
def get_model(model_path, device_name):
    device = torch.device(device_name)

    return load_model(device, Path(model_path))


def list_model_files():
    if not MODELS_DIR.exists():
        return []

    model_files = [
        path
        for path in MODELS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".pth", ".pt"}
    ]

    return sorted(model_files)


def get_default_model_index(model_files):
    if MODEL_PATH in model_files:
        return model_files.index(MODEL_PATH)

    return 0


def list_evaluation_dirs():
    evaluations_dir = REPORTS_DIR / "evaluations"

    if not evaluations_dir.exists():
        return []

    return sorted(
        path
        for path in evaluations_dir.iterdir()
        if path.is_dir() and path.name != "by_model"
    )


def read_csv_if_exists(path):
    if not path.exists():
        return None

    return pd.read_csv(path)


def read_text_if_exists(path):
    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def read_metrics_if_exists(evaluation_dir):
    metrics_path = evaluation_dir / "metrics.json"

    if not metrics_path.exists():
        return None

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_dataset_status():
    rows = []

    for split in SPLITS:
        split_dir = PROCESSED_DATA_DIR / split

        if not split_dir.exists():
            rows.append({
                "split": split,
                "classe": "-",
                "imagens": 0,
                "status": "não encontrado",
            })
            continue

        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())

        if not class_dirs:
            rows.append({
                "split": split,
                "classe": "-",
                "imagens": 0,
                "status": "sem classes",
            })
            continue

        for class_dir in class_dirs:
            rows.append({
                "split": split,
                "classe": class_dir.name,
                "imagens": count_images(class_dir),
                "status": "ok",
            })

    return pd.DataFrame(rows)


def has_processed_dataset(dataset_status):
    expected_splits = set(SPLITS)
    detected_splits = set(dataset_status.loc[dataset_status["imagens"] > 0, "split"])

    return expected_splits.issubset(detected_splits)


def load_image_from_url(image_url):
    request = Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(request, timeout=10) as response:
        image_bytes = response.read()

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def classify_uncertainty(confidence, probabilities):
    probability_values = [probability.item() for probability in probabilities]
    sorted_probabilities = sorted(probability_values, reverse=True)
    margin = sorted_probabilities[0] - sorted_probabilities[1]

    if confidence < 0.60 or margin < 0.20:
        return {
            "level": "alta",
            "message": (
                "Predição incerta. As probabilidades estão próximas, então "
                "esta é uma região onde o modelo tende a errar mais."
            ),
            "color": "warning",
        }

    if confidence < 0.75 or margin < 0.50:
        return {
            "level": "moderada",
            "message": (
                "Predição razoável, mas ainda ambígua. Vale revisar a imagem "
                "antes de tratar o resultado como definitivo."
            ),
            "color": "info",
        }

    return {
        "level": "baixa",
        "message": (
            "Predição mais estável para o modelo atual. Ainda assim, a saída "
            "é uma estimativa probabilística, não uma garantia."
        ),
        "color": "success",
    }


def add_inference_history(row):
    if "inference_history" not in st.session_state:
        st.session_state.inference_history = []

    existing_keys = {
        history_row["history_key"]
        for history_row in st.session_state.inference_history
    }

    if row["history_key"] in existing_keys:
        return

    st.session_state.inference_history.insert(0, row)


def show_colored_prediction_status(predicted_label, true_label):
    if true_label == "-":
        st.info("Sem classe real informada. O app mostra apenas a predição.")
        return None

    if predicted_label == true_label:
        st.success("Resultado correto para a classe real informada.")
        return True

    st.error("Resultado incorreto para a classe real informada.")
    return False


def show_uncertainty_message(uncertainty):
    if uncertainty["color"] == "success":
        st.success(uncertainty["message"])
    elif uncertainty["color"] == "warning":
        st.warning(uncertainty["message"])
    else:
        st.info(uncertainty["message"])


def show_prediction(image, model_path, source, true_label="-"):
    device = get_device()
    model = get_model(str(model_path), str(device))
    predicted_index, confidence, probabilities = predict_image(
        model,
        image,
        device,
    )
    predicted_label = CLASS_NAMES[predicted_index]
    uncertainty = classify_uncertainty(confidence, probabilities)
    correct = show_colored_prediction_status(predicted_label, true_label)

    st.subheader("Resultado")
    st.write(f"Classe prevista: **{predicted_label}**")
    st.write(f"Confiança: **{confidence:.2%}**")

    st.write(f"Grau de incerteza: **{uncertainty['level']}**")
    show_uncertainty_message(uncertainty)

    st.write("Probabilidades:")
    st.write(f"Chihuahua: **{probabilities[0].item():.2%}**")
    st.write(f"Muffin: **{probabilities[1].item():.2%}**")

    add_inference_history({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "model": Path(model_path).name,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "uncertainty": uncertainty["level"],
        "correct": correct,
        "chihuahua_probability": probabilities[0].item(),
        "muffin_probability": probabilities[1].item(),
        "history_key": (
            f"{Path(model_path).name}|{source}|{true_label}|"
            f"{predicted_label}|{confidence:.6f}"
        ),
    })


def is_supported_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def relative_path(path):
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_hard_case_images(split):
    split_dir = HARD_CASES_DATA_DIR / split
    samples = []

    if not split_dir.exists():
        return samples

    for class_name in CLASSES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            continue

        samples.extend(
            (image_path, class_name)
            for image_path in sorted(class_dir.iterdir())
            if is_supported_image(image_path)
        )

    return samples


def run_hard_case_predictions(samples, model_path):
    device = get_device()
    model = get_model(str(model_path), str(device))
    rows = []

    for image_path, true_label in samples:
        image = Image.open(image_path).convert("RGB")
        predicted_index, confidence, probabilities = predict_image(
            model,
            image,
            device,
        )
        predicted_label = CLASS_NAMES[predicted_index]

        rows.append({
            "image_path": relative_path(image_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "chihuahua_probability": probabilities[0].item(),
            "muffin_probability": probabilities[1].item(),
            "correct": predicted_label == true_label,
        })

    return pd.DataFrame(rows)


def show_hard_case_summary(results_df):
    total_images = len(results_df)
    correct_images = int(results_df["correct"].sum())
    accuracy = correct_images / total_images if total_images else 0.0
    errors_df = results_df[results_df["correct"] == False]

    col_total, col_correct, col_errors, col_accuracy = st.columns(4)
    col_total.metric("Imagens", total_images)
    col_correct.metric("Acertos", correct_images)
    col_errors.metric("Erros", len(errors_df))
    col_accuracy.metric("Acurácia", f"{accuracy:.2%}")

    if not errors_df.empty:
        errors_by_class = (
            errors_df.groupby("true_label")
            .size()
            .reset_index(name="errors")
        )
        st.subheader("Erros por classe")
        st.dataframe(errors_by_class, use_container_width=True)

    confusion_df = pd.crosstab(
        results_df["true_label"],
        results_df["predicted_label"],
        rownames=["Classe real"],
        colnames=["Predição"],
        dropna=False,
    ).reindex(index=CLASSES, columns=CLASSES, fill_value=0)

    st.subheader("Matriz de confusão")
    st.dataframe(confusion_df, use_container_width=True)


def show_hard_case_cards(results_df):
    st.subheader("Resultados por imagem")

    for row_index, row in enumerate(results_df.to_dict("records")):
        image_path = PROJECT_ROOT / row["image_path"]
        status = "correto" if row["correct"] else "erro"
        border_color = "#1f8f4d" if row["correct"] else "#b3261e"

        with st.container(border=True):
            columns = st.columns([1, 2])

            with columns[0]:
                st.image(str(image_path), use_container_width=True)

            with columns[1]:
                st.markdown(
                    f"<strong style='color:{border_color}'>{status}</strong>",
                    unsafe_allow_html=True,
                )
                st.write(f"Arquivo: `{row['image_path']}`")
                st.write(f"Classe real: **{row['true_label']}**")
                st.write(f"Predição: **{row['predicted_label']}**")
                st.write(f"Confiança: **{row['confidence']:.2%}**")
                st.write(
                    "Probabilidades: "
                    f"chihuahua {row['chihuahua_probability']:.2%}, "
                    f"muffin {row['muffin_probability']:.2%}"
                )

        if row_index >= 99:
            st.info("Exibindo as primeiras 100 imagens para manter a interface leve.")
            break


def show_inference_history():
    history = st.session_state.get("inference_history", [])

    st.subheader("Histórico de inferências")

    if not history:
        st.info("Nenhuma inferência foi executada nesta sessão.")
        return

    history_df = pd.DataFrame(history).drop(columns=["history_key"])
    st.dataframe(history_df, use_container_width=True)

    if st.button("Limpar histórico de inferências"):
        st.session_state.inference_history = []
        st.rerun()


def show_home_page():
    st.title("Chihuahua or Muffin")
    st.write(
        "Projeto acadêmico de Visão Computacional para classificar imagens "
        "entre duas classes: chihuahua e muffin."
    )

    st.subheader("Pipeline")
    st.write(
        "O fluxo do projeto passa por preparação do dataset, treinamento do "
        "modelo, avaliação e inferência em imagens novas."
    )

    dataset_status = get_dataset_status()
    model_files = list_model_files()
    evaluation_dirs = list_evaluation_dirs()

    st.subheader("Status atual")
    col_dataset, col_models, col_evaluations = st.columns(3)
    col_dataset.metric(
        "Dataset processado",
        "detectado" if has_processed_dataset(dataset_status) else "pendente",
    )
    col_models.metric("Modelos disponíveis", len(model_files))
    col_evaluations.metric("Avaliações salvas", len(evaluation_dirs))

    total_images = int(dataset_status["imagens"].sum())
    st.metric("Imagens processadas", total_images)

    if model_files:
        st.info("Há modelos disponíveis. Use a página Inferência para testar imagens.")
    else:
        st.warning(
            "Nenhum modelo foi encontrado. Execute `python src\\train.py` "
            "para treinar um checkpoint."
        )


def show_inference_page():
    st.title("Inferência")
    st.write("Selecione um checkpoint e envie uma imagem para classificação.")

    model_files = list_model_files()

    if not model_files:
        st.warning(
            "Nenhum modelo treinado foi encontrado. Execute `python "
            "src\\train.py` para gerar um checkpoint em `models/`."
        )
        return

    selected_model = st.selectbox(
        "Modelo",
        model_files,
        index=get_default_model_index(model_files),
        format_func=lambda path: path.name,
    )
    true_label = st.selectbox(
        "Classe real, se conhecida",
        ["-", *CLASS_NAMES],
        help=(
            "Use este campo para o app indicar acerto ou erro. "
            "Se a classe real não for conhecida, mantenha '-'."
        ),
    )

    upload_tab, link_tab = st.tabs(["Upload", "Link"])

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Escolha uma imagem",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagem enviada", use_container_width=True)
            show_prediction(
                image,
                selected_model,
                source=f"upload:{uploaded_file.name}",
                true_label=true_label,
            )

    with link_tab:
        image_url = st.text_input("Cole o link da imagem")

        if image_url:
            try:
                image = load_image_from_url(image_url)
                st.image(image, caption="Imagem do link", use_container_width=True)
                show_prediction(
                    image,
                    selected_model,
                    source=f"link:{image_url}",
                    true_label=true_label,
                )
            except (URLError, TimeoutError, UnidentifiedImageError, OSError):
                st.error(
                    "Não foi possível carregar a imagem pelo link. "
                    "Verifique se a URL aponta diretamente para uma imagem."
                )


    show_inference_history()


def show_dataset_page():
    st.title("Dataset")
    st.write("Status local do dataset processado usado pelo PyTorch.")

    dataset_status = get_dataset_status()

    if has_processed_dataset(dataset_status):
        st.success("Dataset processado detectado.")
    else:
        st.warning("Dataset processado incompleto ou não encontrado.")

    st.dataframe(dataset_status, use_container_width=True)

    st.subheader("Comandos úteis")
    st.code("python src\\prepare_dataset.py", language="powershell")
    st.code(
        "python src\\prepare_dataset.py --raw-dir path\\to\\raw_dataset",
        language="powershell",
    )


def get_evaluation_options():
    history_path = REPORTS_DIR / "evaluation_history.csv"
    history_df = read_csv_if_exists(history_path)

    if history_df is not None and not history_df.empty:
        return history_df

    evaluation_dirs = list_evaluation_dirs()

    if not evaluation_dirs:
        return pd.DataFrame()

    return pd.DataFrame({
        "evaluation_dir": [path.relative_to(PROJECT_ROOT).as_posix() for path in evaluation_dirs],
        "model_name": [path.name for path in evaluation_dirs],
    })


def show_metrics(metrics):
    col_acc, col_precision, col_recall, col_f1 = st.columns(4)
    col_acc.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col_precision.metric("Precision", f"{metrics['precision']:.4f}")
    col_recall.metric("Recall", f"{metrics['recall']:.4f}")
    col_f1.metric("F1-score", f"{metrics['f1_score']:.4f}")


def show_confusion_matrix(evaluation_dir, metrics):
    image_path = evaluation_dir / "confusion_matrix.png"

    if image_path.exists():
        st.image(str(image_path), caption="Matriz de confusão")
        return

    if metrics and "confusion_matrix" in metrics:
        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=metrics.get("class_names"),
            columns=metrics.get("class_names"),
        )
        st.dataframe(matrix, use_container_width=True)
        return

    st.warning("Matriz de confusão não encontrada para esta avaliação.")


def show_evaluation_details(evaluation_dir):
    metrics = read_metrics_if_exists(evaluation_dir)

    if metrics:
        show_metrics(metrics)
    else:
        st.warning("Arquivo metrics.json não encontrado para esta avaliação.")

    st.subheader("Matriz de confusão")
    show_confusion_matrix(evaluation_dir, metrics)

    report = read_text_if_exists(evaluation_dir / "classification_report.txt")

    if report:
        with st.expander("Relatório de classificação", expanded=True):
            st.text(report)
    else:
        st.warning("Relatório de classificação não encontrado.")

    predictions = read_csv_if_exists(evaluation_dir / "predictions.csv")

    if predictions is not None:
        with st.expander("Predições", expanded=False):
            st.dataframe(predictions, use_container_width=True)


def show_evaluation_dashboard_page():
    st.title("Dashboard de Avaliação")

    evaluations_df = get_evaluation_options()

    if evaluations_df.empty:
        st.warning(
            "Nenhuma avaliação foi encontrada. Gere uma avaliação com "
            "`python src\\evaluate.py --model-path models\\best_model.pth`."
        )
        return

    evaluation_labels = evaluations_df["evaluation_dir"].tolist()
    selected_label = st.selectbox("Avaliação", evaluation_labels)
    evaluation_dir = PROJECT_ROOT / selected_label

    show_evaluation_details(evaluation_dir)


def show_evaluation_history_page():
    st.title("Histórico de Avaliações")

    history_path = REPORTS_DIR / "evaluation_history.csv"
    history_df = read_csv_if_exists(history_path)

    if history_df is None or history_df.empty:
        st.warning("Histórico de avaliações não encontrado.")
        st.code("python src\\evaluate.py", language="powershell")
        return

    st.dataframe(history_df, use_container_width=True)
    selected_evaluation = st.selectbox(
        "Ver detalhes da avaliação",
        history_df["evaluation_dir"].tolist(),
    )

    if selected_evaluation:
        with st.expander("Detalhes", expanded=False):
            show_evaluation_details(PROJECT_ROOT / selected_evaluation)


def show_model_history_page():
    st.title("Histórico de Modelos")

    model_files = list_model_files()

    if model_files:
        rows = []
        for model_file in model_files:
            rows.append({
                "modelo": model_file.name,
                "caminho": model_file.relative_to(PROJECT_ROOT).as_posix(),
                "padrão": model_file == MODEL_PATH,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Nenhum checkpoint `.pth` ou `.pt` foi encontrado em `models/`.")

    history_df = read_csv_if_exists(MODEL_HISTORY_PATH)

    if history_df is not None and not history_df.empty:
        st.subheader("model_history.csv")
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("`models/model_history.csv` ainda não existe.")
        st.code("python src\\train.py", language="powershell")


def show_hard_cases_page():
    st.title("Casos Ambíguos")
    st.write(
        "Execute inferência em lote nas imagens separadas em "
        "`data/hard_cases/` sem misturá-las automaticamente ao dataset principal."
    )

    model_files = list_model_files()

    if not model_files:
        st.warning(
            "Nenhum modelo treinado foi encontrado. Execute `python "
            "src\\train.py` para gerar um checkpoint em `models/`."
        )
        return

    selected_model = st.selectbox(
        "Modelo",
        model_files,
        index=get_default_model_index(model_files),
        format_func=lambda path: path.name,
    )
    selected_split = st.selectbox(
        "Split de hard cases",
        ["test", "val", "train"],
        help=(
            "`test` deve ser usado para diagnóstico final. "
            "`train` e `val` ajudam a inspecionar exemplos usados no ajuste."
        ),
    )
    samples = collect_hard_case_images(selected_split)

    st.write(
        f"Imagens encontradas em `data/hard_cases/{selected_split}`: "
        f"**{len(samples)}**"
    )

    if not samples:
        st.warning(
            "Nenhuma imagem foi encontrada para este split. Use a estrutura "
            "`data/hard_cases/<split>/<classe>/`."
        )
        return

    if st.button("Executar inferência em lote"):
        with st.spinner("Rodando predições nos hard cases..."):
            st.session_state.hard_cases_results = run_hard_case_predictions(
                samples,
                selected_model,
            )
            st.session_state.hard_cases_model = selected_model.name
            st.session_state.hard_cases_split = selected_split

    results_df = st.session_state.get("hard_cases_results")

    if results_df is None:
        st.info("Clique no botão para executar a inferência em lote.")
        return

    st.caption(
        "Resultados atuais: "
        f"{st.session_state.get('hard_cases_model')} | "
        f"split {st.session_state.get('hard_cases_split')}"
    )
    show_hard_case_summary(results_df)

    show_only_errors = st.checkbox("Mostrar apenas erros", value=False)
    display_df = results_df

    if show_only_errors:
        display_df = results_df[results_df["correct"] == False]

    st.subheader("Tabela de resultados")
    st.dataframe(display_df, use_container_width=True)

    show_hard_case_cards(display_df)


def main():
    st.set_page_config(
        page_title="Chihuahua or Muffin",
    )
    apply_sidebar_styles()

    page = render_sidebar_navigation()

    if page == "Início":
        show_home_page()
    elif page == "Inferência":
        show_inference_page()
    elif page == "Dataset":
        show_dataset_page()
    elif page == "Dashboard de Avaliação":
        show_evaluation_dashboard_page()
    elif page == "Histórico de Avaliações":
        show_evaluation_history_page()
    elif page == "Histórico de Modelos":
        show_model_history_page()
    elif page == "Casos Ambíguos":
        show_hard_cases_page()


if __name__ == "__main__":
    main()
