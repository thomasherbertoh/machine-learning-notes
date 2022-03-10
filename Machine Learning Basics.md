---
date created: 2022-03-10 08:46
date updated: 2022-03-10 11:13
tags:
  - '#Tasks'
  - '#task'
  - '#Classification'
  - '#Regression'
  - '#Density-Estimation'
  - '#Clustering'
  - '#cluster'
  - '#Dimensionality-reduction'
  - '#Data'
  - '#underfitting'
  - '#overfitting'
  - '#Underfitting'
  - '#Overfitting'
---

# Machine Learning Basics

There are many problems one may come across when trying to determine whether or not a picture is of a bird...

## #Tasks

A #task represents the type of prediction being made to solve a problem on some data. We can identify a #task with the set of functions that can potentially solve it. In general, it consists of functions assigning each input $x \in X$ an output $y \in Y$. $f:X \rightarrow Y$ $F_{task} \subset Y^X$
The nature of $X$, $Y$, and $F_{task}$ depends on the type of #task .

### #Classification

Find a function $f \in Y^X$ assigning each input $x \in X$ a **discrete** label. $f(x) \in Y = \{c_1, ..., c_k\}$

![[classification-function-birds.png]]

### #Regression

Find a function $f(x) \in Y$ assigning each input a **continuous** label.

### #Density-Estimation

Find a probability distribution $f \in \Delta(X)$ that fits the data $x \in X$

### #Clustering

Find a function $f \in \mathbb{N}^x$ that assigns each input $x \in X$ a #cluster index $f(x) \in \mathbb{N}$. All points mapped to the same index form a #cluster .

![[clustering-example.png]]

### #Dimensionality-reduction

Find a function $f \in Y^X$ mapping each (high-dimensional) input $x \in X$to a lower dimensional "embedding" $f(x) \in Y$, where $dim(Y) \ll dim(X)$.

![[dimensionality-reduction-example.png]]

```ad-note
Dimensionality reduction may be used to inspect a classification algorithm.
```

## #Data

Data is information about the problem we're trying to solve in the form of a distribution $p_{data}$.
#Classification and #Regression : $p_{data} \in \Delta (X \times Y)$
#Density-Estimation , #Clustering , and #Dimensionality-reduction : $p_{data} \in \Delta(X)$
The data distribution $p_{data}$ is generally unknown, but we can take samples from it to create the training, validation, and testing sets.^[It's important that we use the same distribution to generate all of these sets, otherwise we risk testing the model on data it's seen a disproportionately small or large amount of compared to its importance in the real world.]

![[training-cycle.png]]

When using a _probabilistic model_ of learning, this probability distribution over example/label pairs is called the **data generating distribution**.

### Training set design

The failure of a machine learning algorithm is often caused by a bad selection of training samples. For example, we might introduce unwanted correlations from which the algorithm derives wrong conclusions.

```ad-example
If we're trying to make a model that distinguishes between pictures of dogs and pictures of cats, and all of the pictures of dogs are taken on sunny days while all the pictures of cats are taken on cloudy days then the algorithm is more likely to learn the difference between pictures taken on sunny days and pictures taken on cloudy days.
```

## Models and Hypothesis Space

A model is a program that can solve our problem, or the implementation of a function $f \in F_{task}$ that can be easily computed. A set of models forms a hypothesis space $H \subset F_{task}$. The learning algorithm seeks a solution within the hypothesis space.

A model may be represented by the expression $f_w(x) = \sum_{j=0}^{M}w_jx^j$ and the hypothesis space may then be represented by $H_M = \{f_w : w \in \mathbb{R}^M\}$, where $w_j$ is the $j$th parameter and $M$ is a fixed value in $\mathbb{N}$ representing the degree of the function.

### Polynomial Curve Fitting Objectives

#### The ideal target

We want to minimise a generalising error function $E(f;p_{data})$ which determines how well a solution $f \in F_{task}$ fits some given data and guides the selection of the best solution in $F_{task}$.
$f^{\star} \in arg\ min_{f \in F_{task}}\ E(f;p_{data}) \mid f \in F_{task}$

```ad-problem
The search space is too large and we don't know $p_{data}$^[the distribution of the data regarding the problem we're solving]
```

#### The feasible target

We need to restrict our focus to finding functions that can be implemented and evaluated easily. We define a model hypothesis space $H \subset F_{task}$ and seek a solution within that space. $f^{\star}_{H} \in arg\ min_{f \in H}\ E(f;p_{data})$

```ad-problem
This cannot be computed exactly, as $p_{data}$ is unknown
```

#### The actual target

We need to work on a data sample, i.e. a training set $D_n = \{z_1, ..., z_n\}$ where $z_i = (x_i, y_i) \in X \times Y$ and $z_i \sim p_{data}$.
$f^{\star}_H(D_n) \in arg\ min_{f \in H}\ E(f;D_n)$

```ad-note
We're minimising the error using the training data, so we're not necessarily also minimising the error for the overall data distribution^[i.e. the testing data].
```

![[polynomial-curve-fitting.png]]

### Error functions

Typically generalisation and training error functions can be written in terms of a pointwise loss $l(f; z)$ measuring the error incurred by $f$ on the training example $z$. $E(f;p_{data}) = \mathbb{E}_{z \sim p_{data}}[l(f;z)]$^[pointwise loss based on a training example similar to the original data distribution] $E(f;D_n)=\frac{1}{n} \sum_{i=1}^{n}l(f;z_i)$^[average pointwise loss using the training data]

```ad-example
Objective: $$f^{\star}_{H_M}(D_n) \in arg\ min_{f \in H_M}E(f;D_n)$$
equivalent to $f_{w^{\star}}$ where $$w^{\star} \in arg\ min_{w \in \mathbb{R}^M} \frac{1}{n} \sum^{n}_{i=1}[f_w(x_i) - y_i]^2$$ ^[requires solving a linear system of equations]
```

Learning algorithms will search for the optimal value, but may end up at a different result as they are susceptible to finding local minima. This can lead to #underfitting or #overfitting .

#### #Underfitting

Underfitting may occur when the model hasn't had enough experience with the dataset to properly minimise error and allow effective generalisation. In this case, error on the training data would be roughly the same as the error on any other relevant data, but this error value would render the model practically unusable in any real-world scenario.

#### #Overfitting

Overfitting occurs when the model adapts to the training data to the point where it is only good for that data, as if it had learnt it "off-by-heart". This would lead to extremely low error on the training data, but extremely high error on any other data due to a lack of generalisability.

```ad-example
In the below image underfitting is shown in the left-hand column, and overfitting is shown in the right-hand column. The middle column is a representation of what we may hope for in a near-perfect scenario.
![[underfitting-overfitting-illustration.png]]
```

#### Approximation Error

It is likely that some error will be induced by the choice of hypothesis space, as the random nature of its selection may exclude the absolute minimum of the function.

#### Irreducible Error

It's worth noting that it's impossible to reduce error to zero due to randomness or variability in the system we're trying to predict. These are unobservable deterministic factors, but they could be interpreted as inherent variability.

### Improving #Generalisation

- Avoid obtaining minimum training error
- Reduce model capacity
- Change the objective with a #regularisation term
- Inject noise into the learning algorithm
- Stop the learning algorithm before convergence

#### #Regularisation

```ad-definition
Modification of the training error function with a term $\Omega(f)$ that typically penalises complex solutions.
$$E_{reg}(f;D_n) = E(f;D_n) + \lambda_n\Omega(f)$$ ^[$lamba_n$ is a trade-off parameter]
```

```ad-example
We can regularise by penalising polynomials with large coefficients
$$E_{reg}(f_w;D_n) = \frac{1}{n}\sum_{i=1}^{n}[f_w(x_i) - y_i]^2 + \frac{\lambda}{n} ||w||^2$$
```