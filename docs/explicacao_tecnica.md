# Explicação técnica

## Objetivo

O projeto "Chihuahua or Muffin" é uma classificação binária de imagens.

Classes:

- `chihuahua`
- `muffin`

A solução cobre o fluxo completo: preparação do dataset, carregamento com PyTorch, criação do modelo, treinamento, avaliação, predição individual e interface Streamlit.

## Preparação do dataset

O arquivo `src/prepare_dataset.py` lê as imagens brutas em `data/raw/kaggle/`.

Ao clonar o repositório, essa pasta existe apenas como estrutura vazia mantida por `.gitkeep`. Se o dataset padrão ainda não estiver disponível nesse caminho, o script baixa o dataset "Muffin vs Chihuahua" automaticamente com `kagglehub`, copia os arquivos brutos para `data/raw/kaggle/` e usa essa pasta como origem.

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

Por enquanto, a única arquitetura suportada explicitamente é `mobilenet_v2`. O código recebe o parâmetro `architecture` para facilitar a inclusão de outras arquiteturas futuramente, mas não adiciona novos modelos nesta versão.

## Treinamento

O arquivo `src/train.py` executa o treino por `5` épocas usando:

- `CrossEntropyLoss`;
- otimizador `Adam`;
- taxa de aprendizado `0.001`;
- `cuda`, se disponível, ou `cpu`.

Ao final de cada época, o modelo é validado. Sempre que a acurácia de validação melhora, o melhor `state_dict` é mantido. Ao final do treinamento, o checkpoint é salvo com o padrão:

```text
models/chihuahua_muffin_mobilenet_v2_YYYYMMDD_HHMMSS_valacc_SCORE.pth
```

O mesmo checkpoint é copiado para `models/best_model.pth` para manter compatibilidade com o fluxo antigo.

O arquivo `models/model_history.csv` registra um histórico local com:

- caminho do checkpoint;
- arquitetura;
- data de criação;
- número de épocas;
- melhor acurácia de validação;
- caminho do dataset de treino;
- observações.

Esse histórico não é versionado porque representa execuções locais e aponta para arquivos de modelo também locais.

## Avaliação

O arquivo `src/evaluate.py` carrega um checkpoint salvo e avalia o conjunto de teste. Por padrão, usa `models/best_model.pth`, mas permite selecionar outro arquivo:

```powershell
python src\evaluate.py --model-path models\algum_modelo.pth
```

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

Também é possível escolher um checkpoint específico:

```powershell
python src\predict.py --image caminho\para\imagem.jpg --model-path models\algum_modelo.pth
```

A saída mostra:

- caminho da imagem;
- classe prevista;
- confiança;
- probabilidade por classe.

## Interface Streamlit

O arquivo `app/streamlit_app.py` disponibiliza uma interface simples para:

- seleção de checkpoint em `models/`;
- upload de imagem local;
- predição a partir de URL direta de imagem;
- exibição da imagem enviada;
- exibição da classe prevista e das probabilidades.

O app não treina modelos e não depende do dataset de teste. Ele apenas carrega um checkpoint já existente e aplica a inferência na imagem enviada.

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
