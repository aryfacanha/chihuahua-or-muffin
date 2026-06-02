import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa o pipeline padrão sem inferência.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Caminho opcional para o dataset bruto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Caminho opcional para o dataset processado.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Checkpoint opcional para avaliação.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Pula a etapa de treinamento.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Pula a etapa de avaliação.",
    )

    return parser.parse_args()


def run_step(name, command):
    print(f"\n=== {name} ===")
    print(" ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_prepare_command(args):
    command = [sys.executable, "src/prepare_dataset.py"]

    if args.raw_dir is not None:
        command.extend(["--raw-dir", str(args.raw_dir)])

    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])

    return command


def build_evaluate_command(args):
    command = [sys.executable, "src/evaluate.py"]

    if args.model_path is not None:
        command.extend(["--model-path", str(args.model_path)])

    return command


def main():
    args = parse_args()

    run_step("Preparação do dataset", build_prepare_command(args))
    run_step("Validação do carregamento do dataset", [sys.executable, "src/dataset.py"])
    run_step("Validação da arquitetura do modelo", [sys.executable, "src/model.py"])

    if not args.skip_train:
        run_step("Treinamento", [sys.executable, "src/train.py"])
    else:
        print("\n=== Treinamento ===")
        print("Etapa pulada por --skip-train.")

    if not args.skip_evaluate:
        run_step("Avaliação", build_evaluate_command(args))
    else:
        print("\n=== Avaliação ===")
        print("Etapa pulada por --skip-evaluate.")

    print("\nPipeline concluído.")


if __name__ == "__main__":
    main()
