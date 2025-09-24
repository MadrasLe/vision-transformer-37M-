# Vision Transformer (37.9M) - Treinado do Zero em CIFAR-100

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-red.svg)](https://pytorch.org/)

Este repositório contém a implementação e o treinamento **do zero** de um **Vision Transformer (ViT)** com **37.9 milhões de parâmetros**, utilizando o dataset **CIFAR-100** e a biblioteca PyTorch.

O objetivo deste projeto é demonstrar a capacidade de implementar e treinar uma arquitetura Transformer moderna para tarefas de visão computacional, seguindo as melhores práticas de engenharia de machine learning.

---

## Resultados Principais

* **Dataset:** CIFAR-100
* **Acurácia de Validação (melhor):** **54.84%** (na época 92)
* **Tempo de Treino:** 100 épocas em GPU NVIDIA L4 (~1h30min)

### Configuração do Modelo
* **Arquitetura:** Vision Transformer (ViT)
* **Parâmetros Treináveis:** 37.94 Milhões
* **Dimensão do Embedding (`dim`):** 384
* **Profundidade (`depth`):** 8 blocos Transformer
* **Cabeças de Atenção (`heads`):** 12
* **Tamanho do Patch (`patch`):** 4x4 pixels

---

### 1. Instalação
Primeiro, instale as dependências necessárias. É recomendado criar um ambiente virtual.

pip install -r requirements.txt

### 2. Executando o Treinamento
Você pode iniciar o treinamento usando o script `train_vit.py`.

**Para replicar o resultado do CIFAR-100 (37.9M):**
bash
!python train_vit.py --dataset cifar100 \(dataset, pode ser trocado por cifar10 ou imagenet1k)
                    --img-size 32 \
                    --patch 4 \
                    --model-dim 384 \(um multiplo de head)
                    --depth 8 \
                    --heads 12 \
                    --mlp-ratio 4 \
                    --batch-size 256 \(controlar batch size em caso de OOM)
                    --epochs 100 \(controlar tempo de treino)
                    --amp \
                    --outdir runs/vit_cifar100_30m



**Para treinar um modelo menor no CIFAR-10:**
```bash
python train_vit.py \
    --dataset cifar10 \
    --depth 6 \
    --heads 8 \
    --model-dim 256 \
    --epochs 50 \
    --amp \
    --outdir runs/vit_cifar10_small
```

---

###  Análise do Treinamento

O modelo foi treinado por 100 épocas e atingiu uma **acurácia máxima de 54.84%** no conjunto de validação do CIFAR-100.

Observa-se um leve overfitting a partir da época 60, aproximadamente. A acurácia de treino continuou a subir, atingindo 80.57%, enquanto a de validação estagnou em torno de 54%. Esta diferença indica que o modelo se ajustou excessivamente aos dados de treino, perdendo levemente sua capacidade de generalização.

**Possíveis Próximos Passos:**
* **Regularização:** Aumentar o `dropout` ou o `weight_decay` para mitigar o overfitting.
* **Early Stopping:** Interromper o treinamento quando a acurácia de validação parar de melhorar para economizar tempo e evitar o superajuste.
* **Dataset Maior:** Treinar em um dataset mais robusto, como o ImageNet-1k, provavelmente levaria a uma generalização muito melhor, dado o tamanho do modelo.

---

## Log de Treinamento Completo
<details>
100% 169M/169M [00:16<00:00, 10.1MB/s]
📦 Parâmetros treináveis: 37.94M
/content/train_vit.py:368: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
/content/train_vit.py:246: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=args.amp):
/content/train_vit.py:273: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=args.amp):
Epoch 1/100 | train: loss 4.4511 acc 3.47% | val: loss 4.0536 acc 8.37% | best 8.37% | 65.9s
Epoch 2/100 | train: loss 4.0691 acc 8.23% | val: loss 3.6347 acc 15.29% | best 15.29% | 64.7s
Epoch 3/100 | train: loss 3.7963 acc 12.11% | val: loss 3.3292 acc 19.45% | best 19.45% | 65.6s
Epoch 4/100 | train: loss 3.5929 acc 15.16% | val: loss 3.1697 acc 21.88% | best 21.88% | 66.6s
Epoch 5/100 | train: loss 3.4769 acc 17.44% | val: loss 3.0837 acc 23.74% | best 23.74% | 67.5s
Epoch 6/100 | train: loss 3.3821 acc 18.73% | val: loss 2.9403 acc 26.39% | best 26.39% | 67.3s
Epoch 7/100 | train: loss 3.2591 acc 20.87% | val: loss 2.8524 acc 28.25% | best 28.25% | 66.9s
Epoch 8/100 | train: loss 3.1933 acc 22.07% | val: loss 2.7035 acc 31.12% | best 31.12% | 67.4s
Epoch 9/100 | train: loss 3.1231 acc 23.54% | val: loss 2.6668 acc 31.43% | best 31.43% | 66.8s
Epoch 10/100 | train: loss 3.0553 acc 24.65% | val: loss 2.6068 acc 33.34% | best 33.34% | 67.4s
Epoch 11/100 | train: loss 2.9949 acc 26.00% | val: loss 2.5427 acc 34.06% | best 34.06% | 67.6s
Epoch 12/100 | train: loss 2.9319 acc 27.28% | val: loss 2.5364 acc 34.92% | best 34.92% | 66.8s
Epoch 13/100 | train: loss 2.8944 acc 28.04% | val: loss 2.4877 acc 35.45% | best 35.45% | 67.1s
Epoch 14/100 | train: loss 2.8455 acc 29.08% | val: loss 2.4332 acc 36.94% | best 36.94% | 67.4s
Epoch 15/100 | train: loss 2.8042 acc 29.89% | val: loss 2.3757 acc 38.19% | best 38.19% | 66.8s
Epoch 16/100 | train: loss 2.7627 acc 30.50% | val: loss 2.3456 acc 38.61% | best 38.61% | 66.6s
Epoch 17/100 | train: loss 2.7292 acc 31.15% | val: loss 2.2959 acc 39.79% | best 39.79% | 67.5s
Epoch 18/100 | train: loss 2.6793 acc 32.24% | val: loss 2.2796 acc 40.16% | best 40.16% | 67.4s
Epoch 19/100 | train: loss 2.6518 acc 32.76% | val: loss 2.2330 acc 41.64% | best 41.64% | 67.1s
Epoch 20/100 | train: loss 2.6006 acc 33.74% | val: loss 2.2246 acc 41.48% | best 41.64% | 66.8s
Epoch 21/100 | train: loss 2.5626 acc 34.61% | val: loss 2.2171 acc 40.91% | best 41.64% | 66.1s
Epoch 22/100 | train: loss 2.5225 acc 35.31% | val: loss 2.2482 acc 41.00% | best 41.64% | 66.1s
Epoch 23/100 | train: loss 2.4949 acc 35.89% | val: loss 2.1600 acc 43.32% | best 43.32% | 67.5s
Epoch 24/100 | train: loss 2.4532 acc 36.97% | val: loss 2.1345 acc 42.93% | best 43.32% | 66.1s
Epoch 25/100 | train: loss 2.4138 acc 37.64% | val: loss 2.0860 acc 44.76% | best 44.76% | 67.5s
Epoch 26/100 | train: loss 2.3873 acc 38.23% | val: loss 2.0982 acc 44.34% | best 44.76% | 66.9s
Epoch 27/100 | train: loss 2.3601 acc 38.99% | val: loss 2.0683 acc 45.06% | best 45.06% | 66.7s
Epoch 28/100 | train: loss 2.3053 acc 40.27% | val: loss 2.0506 acc 45.64% | best 45.64% | 67.5s
Epoch 29/100 | train: loss 2.2738 acc 40.93% | val: loss 2.0669 acc 45.62% | best 45.64% | 66.1s
Epoch 30/100 | train: loss 2.2272 acc 41.96% | val: loss 2.0115 acc 46.78% | best 46.78% | 67.6s
Epoch 31/100 | train: loss 2.1868 acc 42.79% | val: loss 2.0055 acc 46.78% | best 46.78% | 66.9s
Epoch 32/100 | train: loss 2.1479 acc 43.53% | val: loss 1.9527 acc 48.16% | best 48.16% | 67.5s
Epoch 33/100 | train: loss 2.1117 acc 44.69% | val: loss 1.9356 acc 49.01% | best 49.01% | 67.5s
Epoch 34/100 | train: loss 2.0836 acc 45.33% | val: loss 1.9641 acc 47.79% | best 49.01% | 66.2s
Epoch 35/100 | train: loss 2.0284 acc 46.05% | val: loss 1.9581 acc 48.69% | best 49.01% | 66.3s
Epoch 36/100 | train: loss 1.9989 acc 47.31% | val: loss 1.9611 acc 48.05% | best 49.01% | 66.1s
Epoch 37/100 | train: loss 1.9442 acc 48.53% | val: loss 1.9307 acc 49.35% | best 49.35% | 67.0s
Epoch 38/100 | train: loss 1.9119 acc 49.48% | val: loss 1.8868 acc 50.71% | best 50.71% | 67.6s
Epoch 39/100 | train: loss 1.8709 acc 50.06% | val: loss 1.8825 acc 50.12% | best 50.71% | 66.2s
Epoch 40/100 | train: loss 1.8258 acc 51.16% | val: loss 1.9492 acc 49.89% | best 50.71% | 66.5s
Epoch 41/100 | train: loss 1.7839 acc 52.08% | val: loss 1.9511 acc 49.30% | best 50.71% | 66.1s
Epoch 42/100 | train: loss 1.7571 acc 52.85% | val: loss 1.9190 acc 50.34% | best 50.71% | 66.2s
Epoch 43/100 | train: loss 1.7030 acc 54.54% | val: loss 1.9807 acc 48.95% | best 50.71% | 66.2s
Epoch 44/100 | train: loss 1.6737 acc 54.96% | val: loss 1.9423 acc 50.06% | best 50.71% | 66.2s
Epoch 45/100 | train: loss 1.6444 acc 55.89% | val: loss 1.9149 acc 51.20% | best 51.20% | 66.9s
Epoch 46/100 | train: loss 1.5865 acc 57.36% | val: loss 1.9098 acc 51.58% | best 51.58% | 67.7s
Epoch 47/100 | train: loss 1.5498 acc 58.41% | val: loss 1.8981 acc 51.95% | best 51.95% | 66.8s
Epoch 48/100 | train: loss 1.5176 acc 59.15% | val: loss 1.9327 acc 51.25% | best 51.95% | 66.2s
Epoch 49/100 | train: loss 1.4758 acc 60.46% | val: loss 1.9508 acc 51.54% | best 51.95% | 67.0s
Epoch 50/100 | train: loss 1.4432 acc 61.31% | val: loss 1.9663 acc 51.15% | best 51.95% | 66.9s
Epoch 51/100 | train: loss 1.4116 acc 62.17% | val: loss 1.9241 acc 52.26% | best 52.26% | 66.8s
Epoch 52/100 | train: loss 1.3877 acc 62.76% | val: loss 1.9381 acc 51.76% | best 52.26% | 67.0s
Epoch 53/100 | train: loss 1.3463 acc 63.71% | val: loss 1.9494 acc 51.84% | best 52.26% | 66.1s
Epoch 54/100 | train: loss 1.3299 acc 63.92% | val: loss 1.9561 acc 52.24% | best 52.26% | 66.9s
Epoch 55/100 | train: loss 1.2923 acc 65.19% | val: loss 1.9775 acc 52.32% | best 52.32% | 67.3s
Epoch 56/100 | train: loss 1.2702 acc 65.87% | val: loss 2.0061 acc 51.80% | best 52.32% | 66.8s
Epoch 57/100 | train: loss 1.2230 acc 66.90% | val: loss 1.9725 acc 52.45% | best 52.45% | 67.1s
Epoch 58/100 | train: loss 1.1953 acc 67.72% | val: loss 1.9850 acc 52.73% | best 52.73% | 66.8s
Epoch 59/100 | train: loss 1.1723 acc 68.38% | val: loss 1.9862 acc 52.91% | best 52.91% | 66.8s
Epoch 60/100 | train: loss 1.1467 acc 69.13% | val: loss 1.9906 acc 53.37% | best 53.37% | 67.6s
Epoch 61/100 | train: loss 1.1466 acc 69.11% | val: loss 1.9999 acc 53.47% | best 53.47% | 67.6s
Epoch 62/100 | train: loss 1.0985 acc 70.28% | val: loss 2.0487 acc 52.43% | best 53.47% | 66.2s
Epoch 63/100 | train: loss 1.0898 acc 70.67% | val: loss 2.0113 acc 53.57% | best 53.57% | 67.6s
Epoch 64/100 | train: loss 1.0638 acc 71.54% | val: loss 2.0034 acc 54.12% | best 54.12% | 67.5s
Epoch 65/100 | train: loss 1.0441 acc 71.88% | val: loss 2.0489 acc 53.18% | best 54.12% | 67.2s
Epoch 66/100 | train: loss 1.0204 acc 72.67% | val: loss 2.0429 acc 53.64% | best 54.12% | 67.0s
Epoch 67/100 | train: loss 1.0048 acc 72.91% | val: loss 2.0648 acc 53.42% | best 54.12% | 67.1s
Epoch 68/100 | train: loss 0.9843 acc 73.65% | val: loss 2.0717 acc 53.19% | best 54.12% | 66.8s
Epoch 69/100 | train: loss 0.9671 acc 74.01% | val: loss 2.0494 acc 53.66% | best 54.12% | 66.2s
Epoch 70/100 | train: loss 0.9468 acc 74.61% | val: loss 2.0375 acc 53.46% | best 54.12% | 66.1s
Epoch 71/100 | train: loss 0.9337 acc 74.98% | val: loss 2.0363 acc 54.13% | best 54.13% | 67.0s
Epoch 72/100 | train: loss 0.9260 acc 75.33% | val: loss 2.0416 acc 54.06% | best 54.13% | 66.1s
Epoch 73/100 | train: loss 0.9092 acc 75.33% | val: loss 2.0537 acc 53.72% | best 54.13% | 66.1s
Epoch 74/100 | train: loss 0.8954 acc 76.04% | val: loss 2.0412 acc 54.05% | best 54.13% | 66.2s
Epoch 75/100 | train: loss 0.8689 acc 76.68% | val: loss 2.0599 acc 53.70% | best 54.13% | 67.1s
Epoch 76/100 | train: loss 0.8846 acc 76.24% | val: loss 2.0713 acc 54.36% | best 54.36% | 68.0s
Epoch 77/100 | train: loss 0.8450 acc 77.34% | val: loss 2.0785 acc 54.08% | best 54.36% | 66.1s
Epoch 78/100 | train: loss 0.8522 acc 77.21% | val: loss 2.0563 acc 54.23% | best 54.36% | 66.2s
Epoch 79/100 | train: loss 0.8334 acc 77.47% | val: loss 2.0549 acc 54.50% | best 54.50% | 66.8s
Epoch 80/100 | train: loss 0.8269 acc 77.82% | val: loss 2.0682 acc 54.50% | best 54.50% | 66.1s
Epoch 81/100 | train: loss 0.8215 acc 77.85% | val: loss 2.0767 acc 54.61% | best 54.61% | 67.4s
Epoch 82/100 | train: loss 0.7978 acc 78.58% | val: loss 2.0970 acc 54.00% | best 54.61% | 66.9s
Epoch 83/100 | train: loss 0.8012 acc 78.49% | val: loss 2.0761 acc 54.67% | best 54.67% | 67.7s
Epoch 84/100 | train: loss 0.7813 acc 79.05% | val: loss 2.0718 acc 54.42% | best 54.67% | 66.1s
Epoch 85/100 | train: loss 0.7824 acc 79.01% | val: loss 2.0834 acc 54.54% | best 54.67% | 67.0s
Epoch 86/100 | train: loss 0.7788 acc 79.24% | val: loss 2.0765 acc 54.58% | best 54.67% | 67.0s
Epoch 87/100 | train: loss 0.7658 acc 79.64% | val: loss 2.0930 acc 54.52% | best 54.67% | 66.1s
Epoch 88/100 | train: loss 0.7521 acc 79.59% | val: loss 2.0801 acc 54.80% | best 54.80% | 67.5s
Epoch 89/100 | train: loss 0.7459 acc 80.10% | val: loss 2.0858 acc 54.58% | best 54.80% | 66.3s
Epoch 90/100 | train: loss 0.7542 acc 79.82% | val: loss 2.0869 acc 54.53% | best 54.80% | 66.1s
Epoch 91/100 | train: loss 0.7512 acc 79.85% | val: loss 2.0935 acc 54.75% | best 54.80% | 66.1s
Epoch 92/100 | train: loss 0.7444 acc 80.06% | val: loss 2.0922 acc 54.84% | best 54.84% | 67.8s
Epoch 93/100 | train: loss 0.7315 acc 80.47% | val: loss 2.0898 acc 54.54% | best 54.84% | 66.4s
Epoch 94/100 | train: loss 0.7241 acc 80.54% | val: loss 2.0865 acc 54.61% | best 54.84% | 66.2s
Epoch 95/100 | train: loss 0.7267 acc 80.43% | val: loss 2.0844 acc 54.65% | best 54.84% | 66.1s
Epoch 96/100 | train: loss 0.7384 acc 80.04% | val: loss 2.0915 acc 54.47% | best 54.84% | 66.2s
Epoch 97/100 | train: loss 0.7274 acc 80.47% | val: loss 2.0859 acc 54.55% | best 54.84% | 66.9s
Epoch 98/100 | train: loss 0.7276 acc 80.45% | val: loss 2.0876 acc 54.57% | best 54.84% | 66.4s
Epoch 99/100 | train: loss 0.7231 acc 80.47% | val: loss 2.0874 acc 54.61% | best 54.84% | 66.9s
Epoch 100/100 | train: loss 0.7291 acc 80.57% | val: loss 2.0890 acc 54.61% | best 54.84% | 66.1s

  
LINK DO MODELO TREINADO: https://huggingface.co/Madras1/VTLM37m
---

## Licença
Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
