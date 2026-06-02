# Chihuahua or Muffin

Projeto acadêmico de Visão Computacional para classificar imagens entre duas classes:

- `chihuahua`
- `muffin`

A ideia é construir um pipeline simples e reproduzível com PyTorch: preparar o dataset, treinar um modelo com transfer learning, avaliar resultados, fazer inferência por terminal e visualizar o estado do projeto em uma interface Streamlit local.

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
- KaggleHub

## Estrutura do Projeto

```text
app/                 Interface Streamlit local
data/raw/kaggle/     Dataset bruto local
data/processed/      Dataset separado em train, val e test
data/hard_cases/     Imagens ambíguas separadas para diagnóstico
docs/                Documentação acadêmica
models/              Checkpoints treinados locais
reports/             Relatórios e histórico de avaliações locais
src/                 Scripts principais do pipeline
```

Os dados, checkpoints e relatórios gerados localmente são ignorados pelo Git para manter o repositório leve. As pastas principais são preservadas com `.gitkeep` quando necessário.

## Instalação Rápida

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

O projeto usa CPU por padrão quando CUDA não está disponível. Para usar GPU NVIDIA, instale uma versão do PyTorch compatível com seu driver e sistema seguindo o guia oficial:

```text
https://pytorch.org/get-started/locally/
```

## Fluxo Rápido de Uso

Execute os comandos abaixo a partir da raiz do projeto.

1. Preparar o dataset:

```powershell
python src\prepare_dataset.py
```

Se `data/raw/kaggle/` estiver vazio, o script baixa automaticamente o dataset "Muffin vs Chihuahua" com `kagglehub`, copia os arquivos brutos para `data/raw/kaggle/` e gera o dataset processado em `data/processed/`.

2. Validar o carregamento do dataset:

```powershell
python src\dataset.py
```

3. Treinar o modelo:

```powershell
python src\train.py
```

4. Avaliar o modelo:

```powershell
python src\evaluate.py
```

5. Fazer inferência por terminal:

```powershell
python src\predict.py --image caminho\para\imagem.jpg
```

6. Abrir a interface Streamlit:

```powershell
streamlit run app\streamlit_app.py
```

Também existe um comando para executar o pipeline principal sem inferência:

```powershell
python src\run_pipeline.py
```

## Dataset

O dataset padrão é baixado do KaggleHub:

```text
samuelcortinhas/muffin-vs-chihuahua-image-classification
```

Por padrão, o projeto usa:

```text
data/raw/kaggle/
```

Para usar um dataset próprio:

```powershell
python src\prepare_dataset.py --raw-dir path\to\raw_dataset
```

O diretório informado precisa conter pastas chamadas exatamente `chihuahua` e `muffin`, em qualquer nível da árvore:

```text
path/to/raw_dataset/
├── train/
│   ├── chihuahua/
│   └── muffin/
└── test/
    ├── chihuahua/
    └── muffin/
```

O script procura essas pastas recursivamente, junta as imagens por classe e cria uma nova divisão em:

```text
data/processed/train/
data/processed/val/
data/processed/test/
```

A divisão padrão é `70%` treino, `15%` validação e `15%` teste.

## Casos Ambíguos

O projeto também prevê uma área separada para imagens ambíguas ou difíceis, que servem para diagnóstico do comportamento do modelo. Esses arquivos não fazem parte automaticamente do dataset principal processado em `data/processed/`.

A estrutura esperada é:

```text
data/hard_cases/
├── train/
│   ├── chihuahua/
│   └── muffin/
├── val/
│   ├── chihuahua/
│   └── muffin/
└── test/
    ├── chihuahua/
    └── muffin/
```

Casos de uso:

- inspecionar imagens reais em que chihuahua e muffin são visualmente parecidos;
- testar se o modelo está muito confiante em exemplos ambíguos;
- comparar resultados entre checkpoints diferentes;
- analisar erros antes de decidir alterar dataset, treinamento ou hiperparâmetros;
- usar `data/hard_cases/test/` como conjunto separado para avaliação final de exemplos difíceis.

O diagnóstico por terminal pode ser executado com:

```powershell
python src\evaluate_hard_cases.py --model-path models\best_model.pth
```

Os resultados são salvos em:

```text
reports/hard_cases/
```

## Modelo e Checkpoints

O projeto usa transfer learning com `mobilenet_v2`.

Durante o treinamento, as camadas extratoras de características ficam congeladas e a camada final é ajustada para duas classes.

Ao final do treino, o melhor checkpoint é salvo com nome versionado:

```text
models/chihuahua_muffin_mobilenet_v2_YYYYMMDD_HHMMSS_valacc_SCORE.pth
```

O mesmo checkpoint também é copiado para:

```text
models/best_model.pth
```

Esse arquivo funciona como padrão para avaliação, inferência e Streamlit.

O histórico local de modelos fica em:

```text
models/model_history.csv
```

## Avaliação e Relatórios

Para avaliar o modelo padrão:

```powershell
python src\evaluate.py
```

Para avaliar um checkpoint específico:

```powershell
python src\evaluate.py --model-path models\algum_modelo.pth
```

Cada avaliação gera uma pasta única em:

```text
reports/evaluations/{model_stem}_{YYYYMMDD_HHMMSS}/
```

Arquivos gerados:

```text
classification_report.txt
confusion_matrix.png
predictions.csv
metrics.json
```

Também são mantidos:

```text
reports/evaluation_history.csv
reports/evaluations/by_model/{model_stem}/evaluation_history.csv
```

Por compatibilidade, os últimos resultados também são copiados para:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/predictions.csv
```

## Interface Streamlit

O Streamlit é uma interface local e acadêmica para acompanhar o projeto. Ele não treina modelos e não prepara dataset por botão nesta versão.

Telas disponíveis:

- `Início`: resumo do projeto e status geral.
- `Inferência`: seleção de checkpoint e classificação de imagem.
- `Dataset`: status dos splits e contagem de imagens por classe.
- `Dashboard de Avaliação`: métricas, matriz de confusão, relatório e predições.
- `Histórico de Avaliações`: tabela com avaliações salvas.
- `Histórico de Modelos`: checkpoints disponíveis e histórico local de modelos.
- `Casos Ambíguos`: inferência em lote nas imagens de `data/hard_cases/`.

Para abrir:

```powershell
streamlit run app\streamlit_app.py
```

## Arquivos Ignorados pelo Git

O `.gitignore` ignora artefatos locais e arquivos pesados:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `models/*.pth`
- `models/*.pt`
- `models/model_history.csv`
- imagens em `data/raw/`
- imagens em `data/processed/`
- imagens em `data/hard_cases/`
- arquivos em `reports/`

## Limitações

- O modelo distingue apenas `chihuahua` e `muffin`.
- Imagens fora desse domínio ainda recebem uma das duas classes.
- A qualidade depende da diversidade e qualidade do dataset.
- Treinamento em CPU pode demorar.
- A interface Streamlit ainda não executa treinamento nem preparação de dataset.
