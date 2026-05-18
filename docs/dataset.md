# Dataset

## Fonte dos dados

O projeto utiliza o dataset "Muffin vs chihuahua", disponível no Kaggle.

O dataset contém imagens organizadas para um problema de classificação binária entre muffins e chihuahuas/cães pequenos.

## Classes utilizadas

- `chihuahua`
- `muffin`

## Organização esperada

As imagens serão organizadas na seguinte estrutura:

```text
data/processed/
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