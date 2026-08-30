
Ritesh Mishra

    Home
    About
    Research
    Blog
    Experience
    CV

On this page

    Why Does Model Inversion Matter? 

    Data Set 

    Goal of the Attack 

    What do the Reconstructed Images Tell Us?
    Digit 0
    Digit 1
    Other digits 

    1. Input is high dimensional 
    2. The model is not perfectly accurate 
    3. No convolutions are used for inductive bias 

    1. Information is present in the model parameters 
    2. Inputs can be optimized using a static model 
    3. Confidence scores add to the attack potential 

    Output the class label only 

   Experiments Summary

Inversion of Model Using Model Inversion Attack on Handwritten Digits
Aug 22, 2026·
admin
· 13 min read
Inversion of Model Using Model Inversion Attack on Handwritten Digits

The recreation of a model inversion attack described by Fredrikson et al. on handwritten digits using the real MNIST dataset, a linear softmax classifier and a from scratch PyTorch implementation of the attack.
Why is Model Inversion Important?

Learning machine learning models does not consist of only learning a function which maps input to output.

Statistical properties about the training data get embedded into the parameters of the model through its training process. This statistical information can sometimes be used for malicious purposes.

And hence an important privacy question arises.

Can one extract information from training data through queries to a trained machine learning model?

Model inversion is a type of attack that falls under this category.

Model inversion attacks were pioneered by Fredrikson, Jha and Ristenpart where they exploit confidence information provided by a trained classifier.

The core concept, however, is unexpectedly straightforward:

    Instead of modifying the model parameters, modify the input.

Model remains static. Attacker considers the input as the optimization variable and finds an input value maximizing the confidence of the model for the selected target class.
Introduction

Regular machine learning training works approximately as follows:

Training data
      ↓
Machine learning model
      ↓
Optimization
      ↓
Model parameters
      ↓
Predictions

In case of model inversion, workflow is different.

The attacker possesses the trained model and finds an input resulting in a particular output value:

Trained model
         ↓
Select target class
         ↓
Input initialization
         ↓
Compute model confidence
         ↓
Compute gradient w.r.t. input
         ↓
Modify input pixels
         ↓
Repeat
         ↓
Class prototype reconstruction


The key point here is that there is no updating of the model weights.

It is only the pixels of the artificial input that are optimized.
What Was Modified from the Original Experiment

As compared to the original experiment by Fredrikson et al., which studied the possibility of getting sensitive information about people’s identities through model inversion, the present work created an experiment of a simpler design and structure.

It includes:

    Real MNIST handwritten digit dataset
    Linear softmax classifier
    PyTorch
    Gradient ascent attack designed from scratch
    Input images of 28 × 28 resolution in grayscale
    Target classes of 0-9 digits

Such approach facilitates the understanding of the process of the attack.
Dataset

For the experiment, the MNIST handwritten digit dataset was used.

The dataset includes:

    60,000 training images
    10,000 testing images
    Images of 28 × 28 pixel resolution
    10 classes
    Digits 0 to 9
    784 inputs per image

Each image is represented as a vector that consists of 784 pixels and is taken from a 28 × 28 matrix.
Conceptually:

28×28 image
     ↓
Flatten
     ↓
784-dimensional vector

The attack does not make use of individual MNIST images during the process of reconstruction.

Rather, it exploits the trained classifier.
Model

The model that we are using as our target is relatively simple.

It is composed of a single layer which is fully-connected:

784 pixels in the input
        ↓
Linear layer
nn.Linear(784, 10)
        ↓
10 classes
        ↓
Softmax
        ↓
Digit probabilities

There are no convolutional layers and no hidden layers either.

Consequently, the representation capacity of this model is quite limited compared to the average CNN model for MNIST classification tasks.

It is useful since there are no superfluous architectural elements complicating our understanding of the model inversion process.
Training the Target Model

The model was trained for 5 epochs using plain stochastic gradient descent.

Test accuracy of the trained model is:

91.79%

It is far worse than 98–99% typical for MNIST recognition by a convolutional neural network.

But that is exactly what is to be expected.

In fact, a linear classifier does not model spatial relationships between neighboring pixels.

Hence, its representation capacity of handwritten digits is poor.

Nevertheless, there is an important question to address here:

    Is there sufficient information stored in the trained linear classifier for successful model inversion?

The experiment shows that there is.
The Model Inversion Attack

After training, the model parameters remain unchanged.

For each target digit class c, I generate a synthetic image.

This image does not represent any actual handwritten digit.

Then the attacker attempts to optimize the image so that the trained classifier is more and more confident in the classification of the image into the chosen target class.


Optimization is done directly on the pixels.

Attack Goal

For a particular target class c, the attack will maximize the probability of the target class.

In my case, I optimize the negative log-probability with an additional L2 term:

loss =
    -log_softmax(model(x))[c]
    + l2 * ||x||²

This optimization process looks like this:

Start with x
    ↓
Forward pass through the frozen model
    ↓
Compute log probability of the target class
    ↓
Compute gradient w.r.t x
    ↓
Update x
    ↓
Clamp pixel values into [0, 1]
    ↓
Loop

Here is the critical point:

Model parameters → FROZEN
Input pixels    → UPDATED

Why do Optimization on the Input?

Normally, the gradients are computed in order to update the model parameters.

For instance:

Training:

input → model → loss
          ↓
      weight update

With model inversion, the process works the other way round:

model weights → fixed

input → model → target confidence
  ↑
pixel update

The gradient indicates how changing a pixel changes the model’s confidence in the chosen class.

In sufficient number of iterations, the input starts resembling the prototype of the target class.
Attack Setup

For each digit between 0 and 9, I carried out:

   2,000 optimization iterations
   Gradient-based optimization
   Pixel values limited to [0, 1]
   Small L2 regularization
   Fixed classifier parameters
   Separate image reconstruction for each class

So the attacker will have ten reconstructed images:


Target 0 → reconstructed 0
Target 1 → reconstructed 1
Target 2 → reconstructed 2
...
Target 9 → reconstructed 9

Results

The following figure presents the reconstructions obtained from the model training process.

The adversary does not take any real MNIST image as the reconstruction target.

Model Inversion on Real MNIST with Linear Softmax Classifier

Figure 1. Results of the model inversion attack on all MNIST classes. The images were reconstructed through the optimization of the pixels against the fixed linear classifier.
What Do The Reconstructions Reveal?

The reconstructions do not have the same level of recognition of each digit.

There are several classes that demonstrate obvious coarse structure.

For instance:
Digit 0

The reconstruction reveals a ring-like structure.

This corresponds to the expected shape of a handwritten zero digit.
Digit 1

The reconstruction includes an evident vertical component.

It matches the known shape of the digit.
Other digits

A number of other reconstructions are also quite noisy.

They have obvious patterns characteristic of the target class, yet they do not readily appear to be a certain handwritten digit.

And this matters.

The point of model inversion is not to obtain an ideal photo-like reconstruction of something.

What this experiment shows us is what kind of data the model uses in order to classify the input according to the learned decision boundary.
Class Prototype vs Training Sample

Probably the most important thing to keep in mind regarding this experiment is that the reconstructed image is not necessarily a certain training example from the MNIST database.

MNIST includes lots of samples of each class, and there are plenty of different hand-written examples of digit 7.

Thus, when an attacker poses the question:

"What does the model think a 7 should look like?"

the optimization process will not necessarily produce a single training example.

Instead, the optimization process is more likely to come up with the following:

a class prototype

and not a:

    training sample

This is especially true for discussions of privacy.
Why the Attack is Possible

The attack hinges on the information provided by the model.

If the API reveals:

Prediction: 7

The attacker gets limited information.

However, the API reveals:

0 → 0.001
1 → 0.002
2 → 0.003
...
7 → 0.91
...

In this case, the attacker gets a lot more information about the way the model reacts to perturbations to the input data.

These confidence values give a much better optimization signal.

This is why confidence information can be used to violate privacy.
The Part Played By Confidence Scores

The attack is made possible by the fact that the classifier reveals confidence information in real numbers.


Conceptually, the process is as follows:

Input image
      ↓
Frozen classifier
      ↓
Class probabilities
      ↓
Confidence for target class
      ↓
Optimization signal
      ↓
Perturb input
      ↓
Boost confidence for target class
The process is repeated until the generated input is highly associated with the target class.

Consequently, the following security takeaway is relevant:

   The output interface of a machine learning model may influence its privacy vulnerability.

While the model might be left unaltered, making it more informative about its predictions may significantly increase the feasibility of certain attacks.
Why the Linear Model Is Interesting

Upon initial consideration, the linear classifier would appear to be too primitive to conduct an interesting privacy experiment.

Nevertheless, this simplicity comes in handy.

The classifier takes on the following form:

z = Wx + b

where:

      x - the input image
      W - the learned weight matrix
      b - the bias
      z - class logits
Then, the distribution is derived using softmax:

p(y=c|x)=softmax(Wx+b)c

Since there is a linear model up until softmax, the connection between input and the logits is simple.

Therefore, the gradient w.r.t the input can be calculated without any issues.

There is not an entire deep network with multiple nonlinear layers to go through.
Why Are the Reconstructions Noisy?

One interesting finding from this experiment is that the reconstructed images turn out relatively noisy.

There are several possible reasons for that.
1. High-dimensional input

MNIST has:

28×28=784

dimensions in the input.

So, the optimizer is free to optimize 784 pixel dimensions independently of each other.

Without having an explicit image prior or some powerful spatial regularization, optimization will make use of weird configurations of pixels, which increase the target class score.
2. Low accuracy of the model

Classifier has:

91.79% accuracy on test set

It is much lower than what is typical for MNIST CNN.

Thus, the learned decision boundary reflects poorly the true distribution of handwritten digits.

And when the attack is being performed against that boundary, the prototype inherits that weakness.

  3. The lack of convolutional inductive bias

CNNs implicitly take advantage of local spatial relations.

The linear classifier doesn't.

For instance, a CNN may learn hierarchical representations like this:

edges
 ↓
strokes
 ↓
curves
 ↓
digit parts
 ↓
whole digit

Linear model learns no such hierarchy.

Hence, inversion results of the linear model may have less natural spatial structure.
Comparison with a CNN

The current model is intentionally much weaker compared to CNN-based implementation from related practical experiments.

The differences are:
Property 	Linear Model 	CNN
Architecture 	Single linear layer 	Many nonlinear layers
Spatial structure 	Less 	More
MNIST accuracy 	91.79% in this experiment 	Usually significantly higher
Inversion surface 	Simple 	Complex
Reconstruction 	Noisy 	May be more structured
Computational cost 	Lower 	Higher

The objective of this experiment was to prove that the fundamental inversion process works with even the simplest classifier.
An Important Privacy Restriction

This particular MNIST experiment serves just to demonstrate the inversion process, but it mustn't be taken for an identity recovery.
  
The explanation lies in the class composition.

For MNIST:

class 0 -> multiple people writing 0
class 1 -> multiple people writing 1
...
class 9 -> multiple people writing 9

Thus, the target class is composed of a large number of samples.

While for identity classification scenario:

class A -> person A
class B -> person B
class C -> person C

As a result, if each class corresponds to a single individual, the reconstruction of the class prototype will provide highly identifiable data.

That is why the privacy risks associated with model inversion largely depend on:

    Dataset
    Class definition
    Architecture of the model
    Information leaked through the API
    Query access of the attacker
    Regularization and optimization procedure

What This Experiment Actually Proves

The experiment shows three key concepts.
1. Model parameters include information

The classifier parameters contain information about the statistical structure learned from MNIST.
   
2. Optimization of inputs based on a frozen model

There is no requirement for the attacker to retrain the classifier.

Rather:

Frozen model
     ↓
Optimization of input
     ↓
Increase target confidence
     ↓
Recovering of class prototype

3. Confidence scores make attacks more effective

With more output interface, there is more information available to the attacker about how the model works.

This implies that the confidence scores should be taken into account in ML models that are privacy-sensitive.
Potential Countermeasures

There were several methods proposed by Fredrikson et al. to reduce the use of confidence information.

Potential countermeasures can be:
Only return the predicted class

Instead of returning the whole probability vector:

[0.01, 0.02, ..., 0.91, ...]

return only:

Prediction: 7

It eliminates most of the confidence information.
Rounding off the confidence scores

Instead of exposing a highly precise probability score:

0.91372841

one could use less precise:

0.9
          
 Lessening precision could lead to less informative optimization.
Add noise properly calibrated

Adding noise to the model also reduces the reliability of the signal available to the attacker.

Nevertheless, privacy protection measures need to be carefully crafted, as too much noise may also undermine the utility of the model.

Security Implications

The most important lesson from this example is not that it is possible to reconstruct digits from MNIST.

The general lesson from this exercise is:

    A trained machine learning model contains information about the training dataset; this information can be exploited based on the prediction interface of the model.

The latter is especially important when it comes to machine learning APIs.

Thus, a system designer needs to think not only of:

How accurate is my model?

but also:

What information does my model leak?

Reproducibility

This experiment was done in:

    Python
    PyTorch
    MNIST
    Gradient-based optimization
    nn.Linear(784, 10) classifier

The attack implementation was done without utilizing any third-party tool for model inversion.

Thus, this experiment is useful as a small educational example of the mechanism.
Key Takeaways          

  
 The main conclusions that we can draw from this experiment are the following:

    Inversion of the model is possible even for such simple linear classifier.
    The reconstruction is an example of class prototype rather than exact copy of an image from the training set.
    784-dimensional input allows the optimizer to have quite a bit of room to produce noisy examples.
    Relatively low classifier accuracy of 91.79% results in poorer reconstructions compared to models with high accuracy.
    Linear models are very easy to analyze since their gradient w.r.t. the input can be computed easily.
    Confidence scores provide valuable information to the adversary.
    Only returning hard labels, decreasing precision of confidence scores or adding noise can protect the model from this kind of attack.
    Privacy problems become much more important if individual identities correspond directly to model classes.

Final takeaway

This experiment opened up a new view of the process of forgetting of the data by a model for me.

Indeed, the model does not simply store data in some kind of database. What it does is using training data to learn some parameters of the decision function.

However, the parameters can still carry information about the underlying distribution of the training data.

In this particular experiment, the simple linear classifier trained on MNIST provided enough information to produce class-level reconstructions of input when the input was optimized against the fixed model.

The fact that the reconstructions are noisy and not perfect is precisely why this experiment is interesting.

What the attacker asks is not:

    "Give me the training image."

But rather:

    "What image would you be most confident that it belongs to class c?" 
  
  
  

That is the fundamental insight of privacy regarding model inversion.
References

    Fredrikson, M., Jha, S., & Ristenpart, T. (2015). Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS).

    Boenisch, F. (2020). Attacks against Machine Learning Privacy (Part 1): Model Inversion Attacks with the IBM-ART Framework.

    Boenisch, F. (2021). Attacks against Machine Learning Privacy (Part 2): Membership Inference Attacks with TensorFlow Privacy.

Experiment Summary

Dataset
MNIST
60,000 training samples
10,000 test samples
28 x 28 pixels

        ↓

Target Model
Linear Softmax Classifier
nn.Linear(784, 10)

        ↓

Training
5 epochs
Stochastic Gradient Descent
91.79% test accuracy

        ↓

Model Inversion
10 target classes
2,000 iterations per class
Pixels of input optimized
Model weights fixed

        ↓

Output
10 class prototypes reconstructed

        ↓

Observation
Coarse structures identifiable
noisy reconstructions

        ↓

Privacy Insight
Model parameters + confidence information
reveal properties of training distribution

Last updated on Aug 22, 2026
Model Inversion
Machine Learning
MNIST
PyTorch
Trustworthy AI
⚡️ Turn Jupyter Notebooks into Blog Posts Jul 15, 2024 →

© 2026 Ritesh Mishra. This work is licensed under CC BY NC ND 4.0

Made with Hugo Blox — Open Source. Build yours →
