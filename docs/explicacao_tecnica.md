# Explicacao tecnica

## Objetivo

O projeto "Chihuahua or Muffin" e uma classificacao binaria de imagens.

Classes:
- `chihuahua`
- `muffin`

Nesta etapa, o projeto apenas prepara e carrega o dataset. Ainda nao ha treino, modelo ou interface.

## Preparacao do dataset

O arquivo `src/prepare_dataset.py` le as imagens brutas em `data/raw/kaggle/`.

Ele procura as classes `chihuahua` e `muffin`, embaralha as imagens com seed fixa e divide os dados em:

```text
70% train
15% val
15% test
```

As imagens sao copiadas para `data/processed/`, ficando organizadas assim:

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

As imagens passam por transforms simples: resize para `224x224`, conversao para tensor e normalizacao no padrao ImageNet.

Depois, sao criados `DataLoaders` para `train`, `val` e `test`, usando `batch_size = 32`.

O resultado esperado inclui as classes detectadas, o `class_to_idx`, as quantidades por split e os shapes dos batches.

Exemplo de shape das imagens:

```text
torch.Size([32, 3, 224, 224])
```

Isso significa 32 imagens RGB com tamanho 224x224.

## Como testar

Ative o ambiente virtual do projeto e execute:

```powershell
python src\prepare_dataset.py
python src\dataset.py
```

Se as classes, quantidades e shapes aparecerem corretamente, o dataset esta pronto para a proxima fase.
