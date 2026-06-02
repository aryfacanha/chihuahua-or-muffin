# Explicação Técnica

## Visão Geral do Pipeline

O projeto "Chihuahua or Muffin" implementa um pipeline de classificação binária de imagens. As classes são:

- `chihuahua`
- `muffin`

O fluxo principal é:

```text
dataset bruto -> preparação -> DataLoader -> modelo -> treino -> avaliação -> inferência -> dashboard local
```

Os scripts em `src/` continuam sendo a base do projeto. A interface Streamlit atua como painel local para visualizar status, fazer inferência e consultar avaliações já geradas.

## Preparação do Dataset

O arquivo `src/prepare_dataset.py` lê as imagens brutas em `data/raw/kaggle/`.

Ao clonar o repositório, essa pasta existe apenas como estrutura vazia mantida por `.gitkeep`. Se o dataset padrão ainda não estiver disponível nesse caminho, o script baixa o dataset "Muffin vs Chihuahua" automaticamente com `kagglehub`, copia os arquivos brutos para `data/raw/kaggle/` e usa essa pasta como origem.

Esse caminho é o padrão do projeto, mas pode ser alterado pela CLI:

```text
data/raw/kaggle/
```

Ele procura recursivamente pastas chamadas `chihuahua` e `muffin`, embaralha as imagens com seed fixa e cria a divisão:

```text
70% train
15% val
15% test
```

Uso padrão:

```powershell
python src\prepare_dataset.py
```

Uso com dataset bruto customizado:

```powershell
python src\prepare_dataset.py --raw-dir path\to\raw_dataset
```

O dataset processado fica em:

```text
data/processed/train/chihuahua
data/processed/train/muffin
data/processed/val/chihuahua
data/processed/val/muffin
data/processed/test/chihuahua
data/processed/test/muffin
```

Essa estrutura é compatível com `torchvision.datasets.ImageFolder`.

## Treinamento com Transfer Learning

O arquivo `src/model.py` cria uma MobileNetV2 com transfer learning.

A estratégia é:

- usar MobileNetV2 pré-treinada no ImageNet;
- congelar as camadas de extração de características;
- substituir a camada final por uma camada linear com duas saídas;
- treinar apenas os parâmetros necessários para a classificação binária.

O treinamento é executado por CLI:

```powershell
python src\train.py
```

O arquivo `src/train.py` usa:

- `CrossEntropyLoss`;
- otimizador `Adam`;
- validação ao final de cada época;
- salvamento do melhor checkpoint com base na acurácia de validação;
- GPU com CUDA, se disponível e funcional, ou CPU caso contrário.

A arquitetura atualmente suportada é `mobilenet_v2`.

## Checkpoints de Modelo

Ao final do treinamento, o melhor modelo é salvo com nome versionado:

```text
models/chihuahua_muffin_mobilenet_v2_YYYYMMDD_HHMMSS_valacc_SCORE.pth
```

O mesmo checkpoint também é copiado para:

```text
models/best_model.pth
```

Esse arquivo funciona como fallback e mantém o fluxo simples para avaliação, inferência e Streamlit.

O histórico local de modelos é salvo em:

```text
models/model_history.csv
```

Esse arquivo não é versionado, pois descreve execuções locais e aponta para checkpoints também locais.

## Avaliação e Reports Versionados

O arquivo `src/evaluate.py` avalia um checkpoint no conjunto de teste.

Uso padrão:

```powershell
python src\evaluate.py
```

Uso com checkpoint específico:

```powershell
python src\evaluate.py --model-path models\algum_modelo.pth
```

Cada avaliação cria uma pasta única:

```text
reports/evaluations/{model_stem}_{YYYYMMDD_HHMMSS}/
```

Dentro dessa pasta ficam:

```text
classification_report.txt
confusion_matrix.png
predictions.csv
metrics.json
```

O arquivo `metrics.json` relaciona a avaliação ao modelo usado e registra:

- caminho do modelo;
- nome do modelo;
- diretório da avaliação;
- data e hora da avaliação;
- caminho do dataset de teste;
- accuracy;
- precision;
- recall;
- F1-score;
- matriz de confusão;
- nomes das classes;
- caminhos dos artefatos gerados.

O histórico global de avaliações fica em:

```text
reports/evaluation_history.csv
```

Também existe um histórico por modelo em:

```text
reports/evaluations/by_model/{model_stem}/evaluation_history.csv
```

Essa organização permite que a interface Streamlit liste avaliações antigas e relacione cada dashboard ao checkpoint avaliado.

## Relação entre Checkpoint e Avaliação

Um checkpoint representa o estado treinado de um modelo. Uma avaliação representa a execução desse checkpoint sobre o conjunto de teste.

O mesmo checkpoint pode ser avaliado mais de uma vez. Cada execução gera uma nova pasta em `reports/evaluations/`, preservando o histórico e evitando sobrescrever resultados antigos.

Para compatibilidade com versões anteriores do projeto, a última avaliação também atualiza:

```text
reports/classification_report.txt
reports/confusion_matrix.png
reports/predictions.csv
```

Esses arquivos representam apenas os últimos resultados. O histórico principal fica em `reports/evaluations/`.

## Inferência

O arquivo `src/predict.py` classifica uma imagem individual:

```powershell
python src\predict.py --image caminho\para\imagem.jpg
```

Também é possível selecionar um checkpoint:

```powershell
python src\predict.py --image caminho\para\imagem.jpg --model-path models\algum_modelo.pth
```

A inferência usa o mesmo pré-processamento do treino: redimensionamento para `224x224`, conversão para tensor e normalização no padrão ImageNet.

## Streamlit como Dashboard Local

O arquivo `app/streamlit_app.py` fornece uma interface local/acadêmica para consultar o estado do projeto e usar o modelo treinado.

O Streamlit possui as telas:

- `Início`: explicação do projeto e status de dataset, modelos e avaliações.
- `Inferência`: seleção de checkpoint e classificação de imagem por upload ou URL.
- `Dataset`: status dos splits e quantidade de imagens por classe.
- `Dashboard de Avaliação`: métricas, matriz de confusão, relatório de classificação e predições de uma avaliação salva.
- `Histórico de Avaliações`: tabela com avaliações registradas.
- `Histórico de Modelos`: checkpoints disponíveis e histórico local de modelos.

O Streamlit não treina modelos e não prepara dataset pela interface nesta versão. Essas etapas continuam sendo feitas por CLI.

Para abrir:

```powershell
streamlit run app\streamlit_app.py
```

## Limitações

- O treinamento ainda não é feito pela interface Streamlit.
- A preparação do dataset ainda é feita por CLI.
- Treinamento em CPU pode demorar.
- A qualidade do modelo depende da qualidade e diversidade do dataset.
- O modelo foi treinado apenas para distinguir `chihuahua` e `muffin`.
- Imagens fora desse domínio ainda recebem uma das duas classes.
- A confiança exibida representa a probabilidade estimada entre as classes conhecidas, não uma certeza absoluta.

## Comandos Principais

```powershell
python src\prepare_dataset.py
python src\train.py
python src\evaluate.py
python src\predict.py --image caminho\para\imagem.jpg
streamlit run app\streamlit_app.py
```
