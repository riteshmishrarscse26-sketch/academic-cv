---
title: "Recreating a Model Inversion Attack on Handwritten Digits"
date: 2026-08-22
authors:
  - admin
tags:
  - Model Inversion
  - Machine Learning
  - MNIST
  - PyTorch
  - Trustworthy AI
categories:
  - Machine Learning Privacy
summary: "A hands-on reproduction of Fredrikson et al.'s model inversion attack using a linear softmax classifier trained on MNIST dataset."
image:
  caption: "Model inversion results reconstructed from a linear softmax classifier trained on MNIST."
  focal_point: "Center"
---

# Recreating a Model Inversion Attack on Handwritten Digits

*A hands-on reproduction of Fredrikson et al.'s model inversion attack (CCS 2015), using the real MNIST dataset, a linear softmax classifier, and a scratch implementation of Pytorch.*

---

## Why Model Inversion Matters in Trustworthy AI

During training, a model learns patterns from the data provided to it. In some cases, information about these patterns can be exploited by an attacker.
ML models do not simply learn a mapping from inputs to outputs.

This creates an important privacy question:

> **Can an attacker recover meaningful information about training data by querying a trained ML model?**

“Model inversion is one type of attack that investigates this privacy risk.”

Fredrikson, Jha, and Ristenpart introduced model inversion attacks that exploit confidence information returned by a trained classifier. Their work demonstrated that an attacker can optimize an input toward a target class and reconstruct a representative example of that class without directly accessing the original training data.

> **Instead of changing the model parameters, change the input.**

The attacker treats the input itself as the optimization variable and searches for an input that maximizes the model's confidence for a chosen target class.

---

# Background

Before discussing the attack, it is useful to look at the usual ML training process:

```text
Training data
     ->
ML model
     ->
Optimization
     ->
Learned model parameters
     ->
Predictions
```

During inversion, the model is already trained, so I keep its parameters fixed and optimize the input instead.

The attacker already has the trained model and instead searches for an input that produces a desired prediction:

```text
Frozen trained model
         ->
Choose target class
         ->
Initialize input
         ->
Calculate model confidence
         ->
Calculate gradient with respect to input
         ->
Update input pixels
         ->
Repeat
         ->
Reconstructed class prototype
```

The important part is that **the model weights are not being updated**.

Only the pixels of the synthetic input are optimized.

---

## Dataset

I used the standard **MNIST handwritten digit dataset**.

The dataset contains:

* **60,000 training images**
* **10,000 test images**
* **28 × 28 pixel resolution**
* **10 classes**
* Digits **0 through 9**
* **784 input values** per image

Each image is converted from a 28 × 28 matrix into a vector containing 784 pixel values.

Conceptually:

```text
28 × 28 image
       ->
Flatten
       ->
784-dimensional vector
```

---

# Model

The target classifier is deliberately simple.

It has one fully connected layer only:

```text
784 input pixels
        ->
Fully connected linear layer
nn.Linear(784, 10)
        ->
10 class logits
        ->
Softmax
        ->
Probability for each digit
```

There are no convolutional layers and no hidden layers.

Compared with a CNN, this model has a much weaker ability to capture spatial patterns in handwritten digits. I chose this architecture because it keeps the inversion process relatively simple to analyze.

---

# Training the Target Model

The model was trained for **5 epochs** using plain stochastic gradient descent.

The resulting test accuracy was:

**91.79%**

This is considerably lower than the approximately 98–99% accuracy commonly achieved by convolutional neural networks on MNIST.

The lower accuracy is expected for this architecture because the classifier operates directly on the flattened pixels rather than learning spatial features through convolution.

A linear classifier does not explicitly model local spatial relationships between neighboring pixels. It therefore has a much weaker representation of handwritten digit structure.

For this experiment, the more interesting question was whether this relatively simple classifier still retained enough information for inversion.

> **Does a weak linear classifier still contain enough information for model inversion?**

The experiment shows that it does.

---


## Attack Objective

The attack tries to maximize the probability of the target class `c`.

The way I implement it is by optimizing the negative log-probability along with an additional L2 penalty:

```text
loss =
    -log_softmax(model(x))[c]
    + l2 * ||x||²
```

The procedure can be described as follows:

```text
x initialization
     ->
Forward pass using fixed model
     ->
Target-class log probability calculation
     ->
Gradient computation with respect to x
     ->
x update
     ->
Pixel value clipping to the range [0, 1]
     ->
Repeat
```

The key point here is:

```text
model weights  -> FROZEN
input pixels   -> UPDATED
```

---

# Why Optimize the Input?

Normally, gradients are used to update model parameters.

For example:

```text
Training:

input  -> model  -> loss
          ↓
      update weights
```

Model inversion reverses this relationship:

```text
model weights  -> frozen

input  model -> target confidence
  ↑
update input pixels
```

The gradient tells the attacker how changing each pixel affects the model's confidence in the selected class.

After enough iterations, the input begins to resemble a prototype of the target class.

---

# Attack Configuration

For every digit from **0 to 9**, I performed:

* 2,000 optimization iterations
* Gradient-based optimization
* Pixel values constrained to `[0, 1]`
* Small L2 regularization
* Frozen classifier parameters
* Independent reconstruction for each target class

The attacker therefore generates ten reconstructed images:

```text
Target 0 -> reconstructed 0
Target 1 -> reconstructed 1
Target 2 -> reconstructed 2
...
Target 9 -> reconstructed 9
```

---

# Results

The following figure shows the reconstructed images generated entirely from the trained model.

The attacker does not use a real MNIST image as the reconstruction target.

![Model inversion results on real MNIST using a linear softmax classifier](mnist_inversion_results.png)

*Figure 1. Model inversion results for all ten MNIST classes. Each image is reconstructed by optimizing the input pixels against the frozen linear classifier.*

---

## What Do the Reconstructions Show?

The reconstruction quality varied across the ten target digits. Some outputs contained recognizable structures, while others remained relatively noisy after optimization.

Some classes show clear coarse structure.

For example:

### Digit 0

The reconstruction contains a visible ring-like structure.

This is consistent with the broad shape expected for a handwritten zero.

### Digit 1

The reconstruction contains a strong vertical component.

Again, this corresponds to the characteristic structure of the digit.

### Other digits

Several other reconstructions are considerably noisier.

They contain visible patterns associated with the target class, but they are harder to identify immediately as a specific handwritten digit.

This difference is important.

Model inversion does not necessarily produce a clean photograph-like reconstruction.

What it reveals is the type of information encoded by the model's learned decision boundary.

---




# The Role of Confidence Scores

The attack takes advantage of the fact that the classifier reveals real-valued confidence information.

Conceptually:

```text
Input image
     ->
Frozen classifier
     ->
Class probabilities
     ->
Target class confidence
     ->
Gradient signal
     ->
Modify input
     ->
Target confidence increases
```

This loop continues until the input image is strongly related to the target class.

It follows that a major security lesson is:

> **The output interface of a ML model can affect its privacy risk.**

The model itself may remain unchanged, while exposing more detailed prediction information can make certain attacks substantially easier.

---



# Comparison With a CNN

The model used here is deliberately weaker than the CNN-based implementation described in related practical demonstrations.

The difference can be summarized as:

| Property          | Linear Model              | CNN                         |
| ----------------- | ------------------------- | --------------------------- |
| Architecture      | Single linear layer       | Multiple nonlinear layers   |
| Spatial structure | Weak                      | Strong                      |
| MNIST accuracy    | 91.79% in this experiment | Typically much higher       |
| Inversion surface | Relatively simple         | More complex                |
| Reconstruction    | Noisier                   | Potentially more structured |
| Computation       | Cheap                     | More expensive              |

The purpose of this experiment is not to claim that the linear model produces a stronger attack.

Instead, it demonstrates that the fundamental inversion mechanism can still operate even with a very simple classifier.

---

# An Important Privacy Limitation

The MNIST test is great for illustrating the process, but it cannot be taken as a direct analogy to an identity recovery attack.

It all boils down to the nature of classes.

For MNIST:

```text
class 0 -> many different people writing 0
class 1 -> many different people writing 1
...
class 9 -> many different people writing 9
```

And so the target class would cover a huge number of examples.

But in the case of identity classification:

```text
class A -> person A
class B -> person B
class C -> person C
```


That's why the consequences for privacy of model inversion heavily depend on:

* Dataset 
* Definition of classes 
* Model architecture 
* Information provided through the API 
* Access to queries 
* Regularization & optimization strategy

---


# Conclusions

1. **Possibly successful the invert a linear classifier model.**
2. **The reconstructed image is a class prototype and not necessarily a specific training sample.**
3. **MNIST images have 784 dimensions allowing the optimizer plenty of freedom to come up with a noisy solution.**
4. **The 91.79% classification accuracy is not very high resulting in inferior reconstructions compared to high performing classifiers.**
5. **Linear models are easier to analyze since it is quite simple to calculate the gradients with respect to the input.**
6. **The confidence score is valuable data for an attacker.**
7. **Providing only hard labels, lowering confidence score precision, or using proper noise can limit the information gained by the attacker.**
8. **Privacy considerations become much more important when the individual identities are tied directly to model classes.**

---



# References

1. Fredrikson, M., Jha, S., & Ristenpart, T. (2015). *Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*. Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS).

2. Boenisch, F. (2020). *Attacks against Machine Learning Privacy (Part 1): Model Inversion Attacks with the IBM-ART Framework.*

3. Boenisch, F. (2021). *Attacks against Machine Learning Privacy (Part 2): Membership Inference Attacks with TensorFlow Privacy.*

---

