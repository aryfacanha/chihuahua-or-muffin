# Chihuahua or Muffin

Projeto acadêmico desenvolvido para a disciplina de Tópicos Especiais em Computação.

## Objetivo

Criar uma prova de conceito de classificação binária de imagens capaz de distinguir entre imagens de chihuahuas e muffins.

## Problema

Chihuahuas e muffins podem ter padrões visuais semelhantes em determinadas imagens, especialmente quando olhos, nariz, manchas e textura aparecem de forma parecida. O projeto utiliza Visão Computacional e Aprendizado Profundo para treinar um classificador capaz de diferenciar as duas classes.

## Tecnologias previstas

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit

## Estratégia

O projeto utilizará transfer learning com um modelo pré-treinado, adaptando sua camada final para classificar duas classes:

- Chihuahua
- Muffin

## Escopo do MVP

- Organizar dataset em treino, validação e teste.
- Treinar um modelo de classificação binária.
- Avaliar o modelo com métricas.
- Gerar matriz de confusão.
- Criar interface simples para upload de imagem.
- Exibir classe prevista e confiança da predição.

## Limitações previstas

- O modelo depende da qualidade e diversidade do dataset.
- O classificador será treinado apenas para distinguir chihuahuas e muffins.
- Imagens fora desse domínio podem gerar predições incorretas.
- A confiança do modelo não deve ser interpretada como certeza absoluta.

## Status

Projeto em fase inicial.