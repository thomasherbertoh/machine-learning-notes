---
date created: 2022-03-08 09:57
date updated: 2022-03-08 10:52
tags:
  - '#Supervised-learning'
  - '#Unsupervised-learning'
  - '#Reinforcement-learning'
  - '#Preprocessing'
  - '#Feature-extraction'
  - '#Normalisation'
  - '#Dimensionality-reduction'
  - '#Feature-selection'
  - '#Feature-projection'
  - '#Classification'
  - '#Regression'
  - '#Clustering'
  - '#Cross-validation'
  - '#Bootstrapping'
  - '#features'
---

# Introduction

## Contents of the course

- #Supervised-learning ^[[[Supervised Learning#Supervised Learning|Supervised Learning]]]
- #Unsupervised-learning ^[[[Unsupervised Learning#Unsupervised Learning|Unsupervised Learning]]]
- #Reinforcement-learning ^[[[Reinforcement Learning#Reinforcement Learning|Reinforcement Learning]]]

```ad-note
There are [[Optional Knowledge#Other Learning Techniques|other learning techniques]].
```

## Machine Learning

````ad-question
title: What is machine learning?
Machine learning is the study of computer algorithms that improve automatically through experience. It is seen as a part of artificial intelligence.

```ad-quote
Machine learning is the science of getting computers to act without being explicitly programmed.

<cite>A. Samuel</cite>
```

```ad-quote
Machine learning is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities to take actions.

<cite>Christopher M. Bishop</cite>
```

```ad-quote
The goal of machine learning is to develop methods that can automatically detect patterns in data, and then to use the uncovered patterns to predict future data or other outcomes of interest.

<cite>Kevin P. Murphy</cite>
```

```ad-quote
Machine learning is about predicting the future based on the past.

<cite>Hal Daume III</cite>
```

```ad-quote
A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.

<cite>T. Mitchell</cite>
```

```ad-tldr
title: TL;DR
Machine learning is the study of algorithms that
- improve their performance $P$
- at some task $T$
- with experience $E$
```
````

```ad-question
title: When is machine learning used?
Machine learning is used when:
- Human expertise does not exist ^[navigating on Mars]
- Humans can't explain their expertise ^[speech recognition]
- Models must be customised ^[personalised medicine]
- Models are based on huge amounts of data ^[genomics]

Machine learning isn't always useful ^[there is no learning in calculating a payroll]
```

A well-defined learning task is given by a triplet $<T, P, E>$.

```ad-example
collapse:
$T:$ Recognising handwritten words
$P:$ Percentage of words correctly classified
$E:$ Database of human-labelled images of handwritten words
```

```ad-example
collapse:
$T:$ Driving on four-lane highways using vision sensors
$P:$ Average distance travelled before a human-judged error
$E:$ A sequence of images and steering commands recorded while observing a human driver
```

```ad-example
collapse:
$T:$ Categorise email messages as spam or legitimate
$P:$ Percentage of email messages correctly classified
$E:$ Database of emails, some with human-given labels
```

## Artificial Intelligence

```ad-definition
Programs with the ability to learn and reason like humans
```

```ad-quote
Our ultimate objective is to make programs that learn from their experience as effectively as humans do.

<cite>John McCarthy</cite>
```

## Deep Learning

```ad-definition
Subset of machine learning in which artificial neural networks adapt and learn from vast amounts of data
```

```ad-quote
Deep learning allows computational models that are composed of multiple processing layers to learn representations of data with multiple levels of abstraction

<cite>Nature</cite> ^[www.nature.com/articles/nature14539]
```

Deep learning means using a neural network with several layers of nodes between input and output. The series of layers between input and output compute relevant features automatically in a series of stages, just as our brains seem to.

### Deep Learning Revolution - Why Now?

- Flood of available data
- Increased computational power
- Growing number of machine learning algorithms and theory developed by researchers
- Increased support from the industry

## The Learning Process

- Measuring devices
  - Sensors
  - Cameras
  - Databases
- #Preprocessing
  - Noise filtering
  - #Feature-extraction
  - #Normalisation
- #Dimensionality-reduction
  - #Feature-selection
  - #Feature-projection
- Model learning
  - #Classification
  - #Regression
  - #Clustering
  - Description
- Model testing loop
  - #Cross-validation
  - #Bootstrapping
- Analysis results

## Data, features, and models

Every item in the dataset is turned into a vector in some way. We then choose the #features of these items which are also represented with vectors and could be thought of as the questions we can ask about the different items in the dataset.

```ad-example
We could represent an apple with the features `[red, round, leaf, 85g]` and a banana with the features `[yellow, curved, no leaf, 120g]`. We would then choose which of these features should hold the most weight in our model.
```

## Training and test data, generalisation

```ad-idea
Training data and test set should belong to the same "data distribution"

More technically, using a probabilistic model of learning, there is some probability distribution over example/label pairs called the "data generating distribution". Both the training data *and* the test set are generated based on this distribution
```

```ad-example
If a large amount of our training set contains red apples, then a proportionally large amount of our test set^[depending on the size of the training and test sets] should contain red apples.
```
