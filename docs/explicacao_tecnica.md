# Explicação técnica

## Objetivo

O projeto "Chihuahua or Muffin" é uma classificação binária de imagens.

Classes:
- `chihuahua`
- `muffin`

Nesta etapa, o projeto já prepara o dataset, carrega os dados, cria o modelo e executa um treino inicial. Ainda não há avaliação final no teste nem interface.

## Preparação do dataset

O arquivo `src/prepare_dataset.py` lê as imagens brutas em `data/raw/kaggle/`.

Ele procura as classes `chihuahua` e `muffin`, embaralha as imagens com seed fixa e divide os dados em:

```text
70% train
15% val
15% test
```

As imagens são copiadas para `data/processed/`, ficando organizadas assim:

```text
data/processed/train/chihuahua
data/processed/train/muffin
data/processed/val/chihuahua
data/processed/val/muffin
data/processed/test/chihuahua
data/processed/test/muffin
```

Ao final, o script mostra a quantidade de imagens por split e por classe.

## Carregamento com PyTorch

O arquivo `src/dataset.py` usa `torchvision.datasets.ImageFolder` para carregar:

```text
data/processed/train
data/processed/val
data/processed/test
```

As imagens passam por transforms simples: resize para `224x224`, conversão para tensor e normalização no padrão ImageNet.

Depois, são criados `DataLoaders` para `train`, `val` e `test`, usando `batch_size = 32`.

O resultado esperado inclui as classes detectadas, o `class_to_idx`, as quantidades por split e os shapes dos batches.

Exemplo de shape das imagens:

```text
torch.Size([32, 3, 224, 224])
```

Isso significa 32 imagens RGB com tamanho 224x224.

## Modelo e treino

O arquivo `src/model.py` cria uma MobileNetV2 pré-treinada com transfer learning. As camadas de features ficam congeladas, e a última camada é ajustada para duas classes.

O arquivo `src/train.py` treina o modelo por 5 épocas usando:

- `CrossEntropyLoss`
- otimizador `Adam`
- `cuda`, se disponível, ou `cpu`

Durante o treino, o script calcula loss e accuracy em treino e validação. O melhor modelo é salvo em:

```text
models/best_model.pth
```

## Como testar

Ative o ambiente virtual do projeto e execute:

```powershell
python src\prepare_dataset.py
python src\dataset.py
python src\model.py
python src\train.py
```

Se o treino finalizar e `models/best_model.pth` for criado, a fase de treino está funcionando.
