---
date created: 2022-03-08 10:01
tags:
  - '#Classification'
  - '#Regression'
  - '#Ranking'
date updated: 2022-03-22 13:46
---

# Supervised Learning

```ad-definition
Building a model using labelled examples that learns to predict new examples
```

## #Classification

- Finite set of labels
- GIven a training set $T = {(x_1, y_1), ..., (x_m, y_m)}$ learn a function $f$ to predict $y$ given $x$^[$y$ is "categorical" $\therefore d = 1$ where $d$ stands for "dimensionality" or "number of dimensions"]

### Applications

- Facial recognition
- Character recognition
- Spam detection
- Medial diagnoses ^[suggest possible illnesses given symptoms]
- Biometrics ^[recognition/authentication using physical and/or behavioural characteristics, such as the face, an iris, or a signature]

## #Regression

- Each label is a "real" value ^[represented with a number, quantitative]

### Applications

- Economics/finance ^[predict the value of a stock]
- Epidemiology
- Car/plane navigation
- Temporal trends ^[weather over time]

## #Ranking

- Each label is a ranking ^[could reference a preference or priority]

```ad-example
Given a query and a set of web pages, rank them according to relevance ^[traditional search engine]
```

```ad-example
Given a query image, find the most visually similar images in the database ^[image-based search engine]
```

### Applications

- User preference
- Image retrieval
- Search
- Re-ranking N-best output lists

## Nearest Neighbours

## Decision Trees and Random Forests

## Kernel Methods

## Deep Neural Networks

### Feedforward, convolutional, and recurrent networks
