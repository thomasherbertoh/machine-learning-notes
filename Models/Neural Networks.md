---
date created: 2022-04-21 10:46
date updated: 2022-04-22 15:13
tags:
  - '#neural-network'
  - '#neural-networks'
  - '#linear-models'
  - '#perceptron'
  - '#perceptrons'
  - '#feed-forward'
  - '#back-propagation'
  - '#Back-propagation'
  - '#cost-function'
  - '#RELU'
---

# Neural Networks

Neural networks are semi-supervised models capable of both classification and regression. A #neural-network is effectively a graph made up of "layers", each one of which contains many nodes/neurons. Each of these neurons is a [[Perceptron|perceptron]].

Before #neural-networks were invented, non-linear problems were unsolvable with simple, #linear-models . It seems counter-intuitive that the #neural-network should be our saviour in this scenario as they are made up of #perceptrons , a famously linear model, but the activation function of the #perceptron is precisely what allows us to introduce non-linearity.

## Feed-Forward Neural Networks

Information is propagated from the input node(s) to the output node(s) through a directed, acyclic graph^[DAG]. In each node the information is mutated by algebraic functions implemented using the connections, weights, and biases of the hidden and output layers.

```ad-note
The hidden layers compute intermediate representations, which are then developed upon by either further hidden layers or the output layer.
```

Here we can see a simple, single layer, #feed-forward #neural-network . The formula to calculate each of the different $z$ values is $z_j = \sum_i w_{i, j}^{(1)}x_i + w_{0, j}^{(1)}$ and the formula to calculate each of the various $\hat y$ output values is $\hat y_k = f(\sum_i w_{i, k}^{(2)}h(z_i) + w_{0, k}^{(2)})$
![[single-layer-neural-network.png]]

### Limitations

```ad-problem
```

Unfortunately, we can't learn a model using the techniques we've seen thus far. For the output layers we have direct supervision, that is the ground-truth labels, but for the hidden layers we cannot know the desired target and therefore the perceptrons can't be trained.

### Back-propagation

```ad-solution
```

The technique that comes in to save the day this time is #back-propagation . This technique consists of the following three steps:

- Forward propagation
  - Sum inputs
  - Produce activations
  - Feed-forward
- Error estimation
- #Back-propagation
  - Back-propagate the error signal and use it to update the weights

#### The Main Idea

Given training samples $T = \{(x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$ adjust all the weights of the network $\Theta$ such that a cost function is minimised: $min_{\Theta}\sum_i L(y_i, f(x_i; \Theta))$

- Choose a loss function
- Update the weights of each layer using [[Linear Models#Gradient-Descent|gradient descent]]
- Use #back-propagation of the error signal to compute the gradient efficiently

### Further Problems

```ad-problem
```

In this form, neural networks are unable to exploit the benefits that may come from having many layers as they risk not only overfitting but more importantly they suffer from the vanishing gradient problem. This is because in neural networks you have to multiply a lot of small numbers, and the more layers there are the more numbers there are. Multiplying so many small numbers causes them to get very small very quickly, and therefore the gradient gets shallow early on in the training, thus slowing it down considerably.

### Basic Elements

Our goal is to approximate some ideal function $f^*: X \rightarrow Y$. A #feed-forward network requires defining a parametric mapping $f(x_i; \Theta)$ and learning parameters to get a good approximation of $f^*$ given the available data samples.

The function $f$ that we're learning is actually a composition of multiple functions which can be described by a DAG, with $f^{(1)}$ being the first layer, $f^{(2)}$ being the second layer, and so on. The depth is the maximum $i$ in the function composition chain.

![[neural-network-function-composition.png]]

#### Training

In training, we want to optimise $\Theta$ to drive $f(x; \Theta)$ closer to $f^*(x)$. To do this, we obtain training data such that each expected output value is equal to $f^*(x)$ for various instances of $x$. Note that this only specifies what we want the output layers to output; we leave the output of the intermediate layers unspecified.

Designing and training a neural network isn't all that different from training any other machine learning model with gradient descent.

#### Modelling Choices

##### Cost Function

The #cost-function measures how well the #neural-network performs on training data. We can apply any of the loss functions seen in [[Linear Models#Loss-Functions]]. For classification it is common to convert outputs into probabilities^[generally $[0, 1]$], that is, the outputs become values explaining how likely the #neural-network thinks it is that the current example belongs to each class.

```ad-note
The choice of the loss used is related to the choice of the output unit.
```

###### Output Units - Linear

Given features $h$, a layer of linear output units gives $\hat y = W^Th + b$. Linear units do not saturate, which can cause some difficulty for gradient-based optimisation algorithms.

```ad-note
Naturally, if the gradient of the output of the model is close to 0 it could be problematic.
```

###### Output Units - Softmax

Sometimes we want to produce normalised probabilities in the output layer, but with linear units we produce unnormalised log probabilities. One solution to this problem is softmax: $S(l_i) = \frac{e^{l_i}}{\sum_k e^{l_k}}$ where the values of $l_i$ are the scores, or logits.

###### Hidden Units

In a hidden unit, we have to accept an input $x$, compute an affine transformation $z = W^Tx + b$, apply an elementwise non-linear function $h(z)$, and then finally obtain the output $h(z)$.

```ad-note
The design of hidden units is an active area of research.
```

###### Rectified Linear Units (RELUs)

![[neural-networks-RELUs.png]]
Here we can see a #RELU with equation $h = max(0, x)$.

In the case of a #RELU , the gradient is always either $0$ or $1$, making it similar to linear units and therefore easy to optimise as they give large and consistent gradients when active.

Unfortunately, they're not differentiable everywhere, although this isn't really a problem in practice as we can return one-sided derivatives at $z = 0$^[gradient-based optimisation is prone to numerical error anyway].
