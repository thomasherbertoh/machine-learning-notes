---
date created: 2022-04-18 11:44
date updated: 2022-04-18 16:09
tags:
  - '#decision-tree'
  - '#Decision-Trees'
  - '#Decision-trees'
---

# Decision Trees

A #decision-tree is a tree-structured prediction model composed of terminal^[leaf] nodes and non-terminal nodes. Non-terminal nodes have two or more children and implement a routing function, whereas leaf nodes have no children and implement a prediction function. There are no cycles, and all nodes have at most one parenct except the root node.

## How #Decision-Trees Work

A #decision-tree takes an input $x \in X$ and routes it through its nodes until it reaches a leaf node. In the leaf a prediction takes place.

![[decision-tree-basic-example.png]]

### Non-Terminal Nodes

Each non-terminal node $Node(\phi, t_L, t_R)$ holds a routing function $\phi \in \{L, R\}^X$, a left child $t_L$ and a right child $t_R$. When $x$ reaches a node it will go to its left or right child depending on the value of $\phi(x) \in \{L, R\}$^[assuming binary trees].

![[decision-tree-basic-routing-non-terminal-node.png]]

### Terminal/Leaf Nodes

Each leaf node $Leaf(h)$ holds a prediction function $h \in F_{task}$^[typically constant]. Depending on the task we want to solve it could be $h \in Y^X$^[classification/regression], $h \in \Delta(X)$^[density estimation], or something else. Once $x$ reaches a leaf node the final prediction is given by $h(x)$.

![[decision-tree-basic-routing-teriminal-node.png]]

### Inference

$f_t$ is the function returning the prediction for the input $x \in X$ according to the decision tree $t$. It is recursively defined as $f_t(x) = h(x)$ if $t = Leaf(h)$, $f_{t_{\phi(x)}}(x)$ if $t = Node(\phi, t_L, t_R

### Decision Boundaries

#Decision-trees divide the feature space into axis parallel (hyper-)rectangles, where each rectangular region is labelled with one label.

```ad-example
For a binary classification problem, we could have something like the following:
![[decision-tree-decision-boundary.png]]
```

### Learning Algorithm

Given a training set $D_n = \{z_1, ..., z_n\}$ find $f_{t^*}$ where $t^* \in argmin_{t \in T}E(f_t; D_n)$ and $T$ is a set of decision trees.

The optimisation problem is easy^[there are many solutions] as long as we don't impose constraints, otherwise it could be NP-hard. A solution can be found using a simple, greedy strategy.

From now on we'll assume $E(f_t; D) = \frac{1}{|D|}\sum_{z \in D}l(f;z)$

Training is performed in batch mode.

First, fix a set of leaf predictions $H_{leaf} \subset F_{task}$^[constant functions]. Then, fix a set of possible routing functions $\Phi \subset \{L, R\}^X$. Now, to "train" the model, we employ a recursive strategy that repetitively partitions the training set and decides whether to grow leaves or non-terminal nodes.

```ad-definition
title: Impurity of a set
An impure group is a group containing multiple different classes, for example:
![[decision-tree-impure-group.png]]
A group with minimum impurity is a group containing exactly one class, for example:
![[decision-tree-minimum-impurity-group.png]]
```

#### Growing a Leaf

Given $D = \{z_1, ..., z_m\}$ the training set reaching the current node:

- The optimal leaf predictor can be computed as $h_{D}^{*} \in argmin_{h \in H_{leaf}}E(h;D)$
- The optimal error value, also known as impurity measure, can be computed as $I(D) = E(h_{D}^{*}; D)$

If some criterion is met we grow a leaf $Leaf(h_{D}^{*})$

#### Growing a Node
If no stopping criterion is met we have to find an optimal split function, $\phi_{D}^{*} \in argmin_{\phi \in \Phi}I_{\phi}(D)$
The impurity $I_{\phi}(D)$ of a split function $\phi$ given a training set  $D$ is computed in terms of the impurity of the split data: $$I_{\phi}(D) = \sum_{d \in \{L, R\}}\frac{|D_{d}^{\phi}|}{|D|}I(D_{d}^{\phi})$$ where $D_{d}^{\phi} = \{(x, y) \in D; \phi(x) = d\}$

```ad-definition
title: Impurity of a split function
The impurity of a split function is the lowest training error that can be attained by a tree consisting of a root and two leaves.
```