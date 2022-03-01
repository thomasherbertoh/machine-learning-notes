# Introduction
## Contents
- [[Supervised Learning#Supervised Learning|Supervised Learning]]
- [[Unsupervised Learning#Unsupervised Learning|Unsupervised Learning]]
- [[Reinforcement Learning#Reinforcement Learning|Reinforcement Learning]]
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
```ad-abstract
title: Definition
Programs with the ability to learn and reason like humans
```
```ad-quote
Our ultimate objective is to make programs that learn from their experience as effectively as humans do.

<cite>John McCarthy</cite>
```

## Deep Learning
```ad-abstract
title: Definition
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