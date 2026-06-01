# Dataset

## Fonte dos dados

O projeto utiliza o dataset "Muffin vs Chihuahua", disponível no Kaggle.

O dataset contém imagens de duas classes visualmente parecidas em alguns contextos:

- `chihuahua`
- `muffin`

## Organização bruta esperada

O script de preparação procura recursivamente pastas chamadas `chihuahua` e `muffin` dentro de:

```text
data/raw/kaggle/
```

Na máquina atual, a estrutura bruta local está organizada assim:

```text
data/raw/kaggle/test/chihuahua
data/raw/kaggle/test/muffin
data/raw/kaggle/train/chihuahua
data/raw/kaggle/train/muffin
```

## Preparação

O arquivo `src/prepare_dataset.py` combina as imagens encontradas por classe, embaralha com seed fixa e cria uma nova divisão:

```text
70% train
15% val
15% test
```

## Organização processada

Após a preparação, o dataset fica em:

```text
data/processed/train/chihuahua
data/processed/train/muffin
data/processed/val/chihuahua
data/processed/val/muffin
data/processed/test/chihuahua
data/processed/test/muffin
```

Essa organização é compatível com `torchvision.datasets.ImageFolder`.

## Contagens locais atuais

```text
train/chihuahua: 2239
train/muffin:    1902
val/chihuahua:    479
val/muffin:       407
test/chihuahua:   481
test/muffin:      409
```

Total processado: `5917` imagens.
