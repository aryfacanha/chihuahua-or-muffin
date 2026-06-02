# Chihuahua or Muffin

Projeto acadêmico de Visão Computacional para classificar imagens entre duas classes:

- `chihuahua`
- `muffin`

A proposta é demonstrar um pipeline completo de aprendizado profundo com PyTorch: preparação do dataset, treinamento com transfer learning, avaliação, inferência por CLI e visualização local pelo Streamlit.

## Tecnologias

- Python
- PyTorch
- Torchvision
- Pillow
- Scikit-Learn
- Matplotlib
- Seaborn
- Pandas
- Streamlit

## Instalação

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

O `requirements.txt` mantém `torch` e `torchvision` genéricos para facilitar a instalação em CPU. Para usar GPU NVIDIA com CUDA, instale o PyTorch com a versão compatível com sua máquina seguindo o comando oficial em:

```text
https://pytorch.org/get-started/locally/
```

## Estrutura do Projeto

```text
app/                 Interface Streamlit local
data/raw/kaggle/     Dataset bruto local
data/processed/      Dataset separado em train, val e test
docs/                Documentação acadêmica
models/              Checkpoints treinados locais
reports/             Relatórios e histórico de avaliações locais
src/                 Scripts principais do pipeline
```

## Preparação do Dataset

Por padrão, o projeto espera o dataset bruto em `data/raw/kaggle/`:

```powershell
python src\prepare_dataset.py
```

Também é possível informar outro caminho:

```powershell
python src\prepare_dataset.py --raw-dir path\to\raw_dataset
```

O script separa os dados em `train`, `val` e `test`, usando a divisão padrão `70/15/15`. As proporções podem ser configuradas:

```powershell
python src\prepare_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

## Treinamento por CLI

O treinamento ainda é feito pela linha de comando:

```powershell
python src\train.py
```

O modelo usa transfer learning com `mobilenet_v2`. Ao final, o melhor checkpoint é salvo com o padrão:

```text
models/chihuahua_muffin_mobilenet_v2_YYYYMMDD_HHMMSS_valacc_SCORE.pth
```

Para manter compatibilidade com o fluxo simples, o mesmo checkpoint também é copiado para:

```text
models/best_model.pth
```

O arquivo local `models/model_history.csv` registra informações dos treinamentos, como caminho do checkpoint, arquitetura, data de criação, épocas, melhor acurácia de validação e observações.

## Avaliação por CLI

Para avaliar o modelo padrão:

```powershell
python src\evaluate.py
```

Para avaliar um checkpoint específico:

```powershell
python src\evaluate.py --model-path models\algum_modelo.pth
```

Cada avaliação gera uma pasta própria em:

```text
reports/evaluations/{model_stem}_{YYYYMMDD_HHMMSS}/
```

Dentro dela são salvos:

```text
classification_report.txt
confusion_matrix.png
predictions.csv
metrics.json
```

O arquivo `reports/evaluation_history.csv` funciona como índice global das avaliações. Além disso, o projeto mantém um histórico por modelo em:

```text
reports/evaluations/by_model/{model_stem}/evaluation_history.csv
```

Por compatibilidade, os últimos resultados também são copiados para:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/predictions.csv
```

## Inferência por CLI

Para classificar uma imagem usando o modelo padrão:

```powershell
python src\predict.py --image caminho\para\imagem.jpg
```

Para escolher um checkpoint específico:

```powershell
python src\predict.py --image caminho\para\imagem.jpg --model-path models\algum_modelo.pth
```

A saída mostra classe prevista, confiança e probabilidade por classe.

## Streamlit

O Streamlit é uma interface local e acadêmica para inferência, status do projeto e visualização de avaliações. Ele não executa treinamento e não prepara dataset por botão nesta versão.

Para abrir:

```powershell
streamlit run app\streamlit_app.py
```

Telas disponíveis:

- `Início`: resumo do projeto, status do dataset, modelos e avaliações.
- `Inferência`: seleção de checkpoint, upload ou link de imagem e predição.
- `Dataset`: status dos splits e contagem de imagens por classe.
- `Dashboard de Avaliação`: métricas, matriz de confusão, relatório e predições de uma avaliação já gerada.
- `Histórico de Avaliações`: tabela com avaliações salvas em `reports/evaluation_history.csv`.
- `Histórico de Modelos`: modelos encontrados em `models/` e conteúdo de `models/model_history.csv`, se existir.

## Arquivos Ignorados pelo Git

O repositório ignora artefatos locais e arquivos pesados:

- `.venv/`
- `__pycache__/`
- arquivos `.pyc`
- `models/*.pth`
- `models/*.pt`
- `models/model_history.csv`
- imagens em `data/raw/`
- imagens em `data/processed/`
- arquivos em `reports/`

A estrutura de algumas pastas é mantida com `.gitkeep`.

## Fluxo Recomendado em uma Máquina Nova

1. Clone o repositório.
2. Crie e ative o ambiente virtual.
3. Instale as dependências com `pip install -r requirements.txt`.
4. Baixe o dataset "Muffin vs Chihuahua" do Kaggle.
5. Organize o dataset bruto em `data/raw/kaggle/`.
6. Execute `python src\prepare_dataset.py`.
7. Execute `python src\train.py`.
8. Execute `python src\evaluate.py`.
9. Teste inferência por CLI com `python src\predict.py --image caminho\para\imagem.jpg`.
10. Abra a interface com `streamlit run app\streamlit_app.py`.

## Limitações

- O modelo só distingue as classes treinadas: `chihuahua` e `muffin`.
- Imagens fora desse domínio ainda recebem uma das duas classes.
- A qualidade do resultado depende da diversidade e qualidade do dataset.
- Treinamento em CPU pode demorar.
- O Streamlit ainda não executa treinamento nem preparação de dataset pela interface.
