# Explicação técnica

## Objetivo

O projeto "Chihuahua or Muffin" é uma classificação binária de imagens.

Classes:

- `chihuahua`
- `muffin`

A solução cobre o fluxo completo: preparação do dataset, carregamento com PyTorch, criação do modelo, treinamento, avaliação, predição individual e interface Streamlit.

## Preparação do dataset

O arquivo `src/prepare_dataset.py` lê as imagens brutas em `data/raw/kaggle/`.

Esse caminho é o padrão do projeto, mas pode ser alterado pela CLI:

```powershell
python src\prepare_dataset.py --raw-dir path\to\raw_dataset
```

Ele procura recursivamente as classes `chihuahua` e `muffin`, embaralha as imagens com seed fixa e divide os dados em:

```text
70% train
15% val
15% test
```

As proporções também podem ser configuradas:

```powershell
python src\prepare_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

Antes de copiar os arquivos, o script valida o diretório bruto, as pastas das classes, a existência de imagens suportadas e a soma das proporções. Ao final, imprime um resumo com os caminhos usados, classes e contagens por split.

As imagens são copiadas para `data/processed/`, mantendo a estrutura esperada pelo PyTorch:

```text
data/processed/train/chihuahua
data/processed/train/muffin
data/processed/val/chihuahua
data/processed/val/muffin
data/processed/test/chihuahua
data/processed/test/muffin
```

## Carregamento com PyTorch

O arquivo `src/dataset.py` usa `torchvision.datasets.ImageFolder` para carregar os três splits.

As imagens passam pelas seguintes transformações:

- redimensionamento para `224x224`;
- conversão para tensor;
- normalização com média e desvio padrão do ImageNet.

Os `DataLoaders` usam `batch_size = 32`. O conjunto de treino é embaralhado; validação e teste não são.

## Modelo

O arquivo `src/model.py` cria uma MobileNetV2 com transfer learning.

A estratégia é:

- carregar a arquitetura MobileNetV2;
- usar pesos pré-treinados do ImageNet durante criação para treino/teste estrutural;
- congelar as camadas de extração de características;
- substituir a camada final por uma camada linear com duas saídas.

Na avaliação e na inferência, a arquitetura é criada sem baixar pesos pré-treinados e recebe os pesos locais de `models/best_model.pth`.

## Treinamento

O arquivo `src/train.py` executa o treino por `5` épocas usando:

- `CrossEntropyLoss`;
- otimizador `Adam`;
- taxa de aprendizado `0.001`;
- `cuda`, se disponível, ou `cpu`.

Ao final de cada época, o modelo é validado. Sempre que a acurácia de validação melhora, o `state_dict` é salvo em:

```text
models/best_model.pth
```

## Avaliação

O arquivo `src/evaluate.py` carrega o melhor modelo salvo e avalia o conjunto de teste.

Ele calcula:

- accuracy;
- precision ponderada;
- recall ponderado;
- F1-score ponderado;
- relatório de classificação;
- matriz de confusão.

Os resultados são salvos em:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/predictions.csv
```

## Predição

O arquivo `src/predict.py` faz inferência em uma única imagem:

```powershell
python src\predict.py --image caminho\para\imagem.jpg
```

A saída mostra:

- caminho da imagem;
- classe prevista;
- confiança;
- probabilidade por classe.

## Interface Streamlit

O arquivo `app/streamlit_app.py` disponibiliza uma interface simples para:

- upload de imagem local;
- predição a partir de URL direta de imagem;
- exibição da imagem enviada;
- exibição da classe prevista e das probabilidades.

## Como testar o fluxo

```powershell
python src\prepare_dataset.py
python src\dataset.py
python src\model.py
python src\train.py
python src\evaluate.py
python src\predict.py --image caminho\para\imagem.jpg
streamlit run app\streamlit_app.py
```
