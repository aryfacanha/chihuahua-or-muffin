# Chihuahua or Muffin

Projeto acadêmico desenvolvido para a disciplina de Tópicos Especiais em Computação.

## Objetivo

Criar uma prova de conceito de classificação binária de imagens capaz de distinguir entre imagens de chihuahuas e muffins.

## Problema

Chihuahuas e muffins podem ter padrões visuais semelhantes em determinadas imagens, especialmente quando olhos, nariz, manchas e textura aparecem de forma parecida. O projeto utiliza Visão Computacional e Aprendizado Profundo para treinar um classificador capaz de diferenciar as duas classes.

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

## Estratégia

O projeto utiliza transfer learning com MobileNetV2 pré-treinada no ImageNet. As camadas extratoras de características são congeladas e a camada final é substituída por uma camada linear com duas saídas:

- `chihuahua`
- `muffin`

## Estrutura

```text
app/                 Interface Streamlit
data/raw/kaggle/     Dataset bruto local do Kaggle
data/processed/      Dataset organizado em train, val e test
docs/                Documentação acadêmica
models/              Pesos treinados locais
reports/             Relatórios de avaliação
src/                 Código principal do pipeline
```

Os dados e modelos treinados são ignorados pelo Git para evitar versionar arquivos pesados ou dependentes da máquina local.

## Fluxo principal

```powershell
python src\prepare_dataset.py
python src\dataset.py
python src\model.py
python src\train.py
python src\evaluate.py
python src\predict.py --image caminho\para\imagem.jpg
streamlit run app\streamlit_app.py
```

## Status

Etapas concluídas:

- Preparação do dataset em `train`, `val` e `test`.
- Carregamento do dataset com PyTorch.
- Modelo com transfer learning usando MobileNetV2.
- Treinamento com validação e salvamento do melhor modelo.
- Avaliação no conjunto de teste.
- Geração de relatório de classificação e matriz de confusão.
- Predição individual por terminal.
- Interface Streamlit com upload de imagem e predição por URL.

## Resultados atuais

A avaliação salva em `reports/classification_report.txt` indica aproximadamente `0.99` de accuracy, precision, recall e F1-score no conjunto de teste local.

## Limitações

- O modelo depende da qualidade e diversidade do dataset.
- O classificador foi treinado apenas para distinguir chihuahuas e muffins.
- Imagens fora desse domínio ainda recebem uma das duas classes.
- A confiança do modelo representa a probabilidade estimada entre as classes conhecidas, não uma certeza absoluta.

## Setup em máquina nova

1. Crie e ative um ambiente virtual.
2. Instale as dependências com `pip install -r requirements.txt`.
3. Baixe o dataset "Muffin vs Chihuahua" do Kaggle.
4. Organize o dataset bruto em `data/raw/kaggle/`.
5. Execute `python src\prepare_dataset.py`.
6. Execute `python src\train.py` para gerar `models/best_model.pth`.
7. Execute `python src\evaluate.py` para gerar os relatórios.

Para usar GPU com CUDA, instale a versão de PyTorch compatível com a máquina seguindo a documentação oficial do PyTorch.
