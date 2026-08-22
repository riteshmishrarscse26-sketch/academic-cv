---
title: "Recreating a Model Inversion Attack on Handwritten Digits"
date: 2026-08-22
authors:
  - me
tags:
  - Model Inversion
  - Privacy
  - Machine Learning
  - MNIST
  - PyTorch
  - Trustworthy AI
categories:
  - Machine Learning Privacy
summary: "A hands-on reproduction of Fredrikson et al.'s model inversion attack using a linear softmax classifier trained on MNIST."
image:
  caption: "Model inversion results on handwritten digits"
  focal_point: "Center"
---

# Recreating a Model Inversion Attack on Handwritten Digits

*A hands-on reproduction of Fredrikson et al.'s model inversion attack (CCS 2015), using a scikit-learn-style digit classification setup with a PyTorch implementation.*

## Background

Machine learning models do not simply learn a mapping from inputs to labels. Their parameters can also encode statistical information about the data used during training.

Fredrikson, Jha, and Ristenpart's 2015 paper, *Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*, demonstrated that an attacker with access to a trained classifier's confidence scores can perform optimization directly on the input to reconstruct a representative example of a target class.

The important idea is simple:

> Instead of changing the model parameters, the attacker changes the input.

The trained model remains frozen while the attacker searches for an input that maximizes the model's confidence for a chosen target class.

This creates an important privacy question:

**How much information about the training data is actually encoded in a model's decision boundary?**

---

## What I Changed

Instead of reproducing the exact MNIST + CNN setup from existing implementations, I used a much simpler model.

### Dataset

I used the real **MNIST handwritten digit dataset**:

- 60,000 training images
- 10,000 test images
- Image resolution: 28 × 28 pixels
- 10 classes: digits 0–9
- Each image contains 784 pixel values

### Model

The classifier is a simple linear softmax model:

```text
784 input pixels
       ↓
Linear layer
nn.Linear(784, 10)
       ↓
10 class logits
       ↓
Softmax
       ↓
Digit probabilities
