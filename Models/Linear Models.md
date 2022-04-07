---
date created: 2022-03-24 08:40
date updated: 2022-04-07 15:22
tags:
  - '#online-learning'
  - '#Dimensionality'
  - '#linearly-separable'
  - '#linear-model'
  - '#Loss-functions'
  - '#Loss-Functions'
  - '#loss-function'
  - '#Gradient-Descent'
  - '#Regularisation'
  - '#regulariser'
  - '#P-Norm'
  - '#Minimisation'
  - '#Convexity'
  - '#P-norms'
---

# Linear Models

```ad-definition
A #linear-model is a model that assumes the data is #linearly-separable , i.e., the data can be separated either by a line ^[in 2D space] or by a hyperplane.
```

## Defining a line

Any pair of values $(w_1, w_2)$ defines a line through the origin.

```ad-example
$0 = w_1f_1 + w_2f_2$
If we choose $w_1 = 1$ and $w_2 = 2$, we get the points $(-2, 1), (-1, 0.5), (0, 0), (1, -0.5)$ etc lying on the line.
![[linear-models-defining-a-line.png]]
This line can also be viewed as the line perpendicular to the *weight vector* $w = (1, 2)$
![[linear-models-defining-a-line-weight-vector.png]]
```

We can also intuitively move a line off the origin by using $a = w_1f_1 + w_2f_2$ for some $a$

```ad-example
$a = -1$ and $w = (1, 2) \therefore -1 = 1f_1 + 2f_2$. We get the points $(-2, 0.5), (-1, 0), (0, -0.5), (1, -1)$ etc meaning the line now intersects the $f_2$ axis at $-1$:
![[linear-models-defining-a-line-off-the-origin.png]]
```

## Classifying with a line

Given a point, we can classify it using our weight vector.

```ad-example
For the blue point $(1, 1)$, $1*1 + 2*1 = 3$.
For the red point $(1, -1)$, $1*1 + 2*-1 = -1$.
We can see that the sign indicates which side of the line the point lies on.
![[linear-models-classifying-with-a-line.png]]
```

## #Dimensionality

A linear model in $n$-dimensional space^[$n$ features] is defined by $n+1$ weights, meaning that in two dimensions we have a line $(0=w_1f_1 + w_2f_2 + b)$^[where $b=-a$], in three dimensions we have a plane $(0=w_1f_1 + w_2f_2 + w_3f_3 + b)$, and in $n$ dimensions, we have a hyperplane $(0 = b + \sum_{i=1}^{n}{w_if_i})$

To classify data using this model we simply check the sign of the result.

## Learning

To learn a linear model, we use a technique called #online-learning . In #online-learning we only see one example at a time. In comparison with the batch approach, where we're given some training data $\{(x_i, y_i) : 1 \le  i \le n\}$, typically i.i.d^[independent and identically distributed], we receive the data points one-by-one. Specifically, the algorithm receives an unlabelled example $x_i$, predicts a classification of the example, and is then told the correct answer $y_i$ so it can update its model before receiving the next example. This approach is particularly beneficial when we have a data stream or a large-scale dataset, or when we're working on privacy-preserving applications.

```ad-example
Let's say we have this initial line and weight vector $w = (1, 0)$, meaning we have the equation $0 = 1f_1 + 0f_2$.
![[learning-linear-classifier-0.png]]
We now receive the point $(-1, 1)$, which we classify as being part of the negative class as $1 * -1 + 0 * 1 = -1$. Unfortunately, this point is meant to be part of the positive class, so we have to update the model. Examining the values of the point, we see that $f_1$ contributed in the wrong direction and that $f_2$ didn't contribute at all, when it could have. In this case we should decrease $w_1$ (we'll go from $1$ to $0$) and increase $w_2$ (we'll go from $0$ to $1$) so we now have $w = (0, 1)$.

The first point is now classified correctly, and we receive the next point $(1, -1)$. Because $0 * 1 + 1 * -1 = -1$ we classify it as part of the negative class, and we're told that this is correct. In this case we do not need to update the model.
![[learning-linear-classifier-1.png]]
```

## #Loss-Functions

When training a model we need to choose a criterion to minimise. In the case of the model $0 = b + \sum_{j=1}^{n}{w_jf_j}$ we could choose to find the values of $w$ and $b$ for which $\sum_{i=1}^{n}{1[y_i(w \cdot x_i + b) \le 0]}$ is minimal. That is, minimising the $0/1$ loss or number of mistakes. Unfortunately, this particular function is far from continuous and is definitely not differentiable, making it incredibly hard^[NP-Hard, in fact] to find the minimum value of it. #Loss-functions are usually "ranked" based on how they score the difference between the actual label $y$ and the predicted label $y^{'}$.

### Surrogate #Loss-Functions

In many cases, we would like to minimise the $0/1$ loss. A surrogate #loss-function is a loss function that provides an upper bound on the actual loss function. Generally we want convex surrogate loss functions as they are easier to minimise. Some possibilities are the following:

- $0/1$ loss:
  - $l(y, y^{'}) = 1[yy^{'} \le 0]$
- Hinge:
  - $l(y, y^{'}) = max(0, 1 - yy^{'})$
- Exponential:
  - $l(y, y^{'}) = exp(-yy^{'})$
- Squared loss:
  - $l(y, y^{'}) = (y-y^{'})^2$

## #Gradient-Descent

Using derivatives we can find the slope of the function in our current position. Using this we can then choose which direction to move in order to minimise the function. Mathematically, when starting from a position $w$, our formula to move will look something like this $w_j = w_j - \frac{d}{dw_j}loss(w)$^[we subtract because if the derivative is negative it means that to go downhill we should move right, therefore we should increase the input]
We can add a further parameter to this equation, $\eta$^[eta], which we use as the learning rate: $w_j = w_j - \eta\frac{d}{dw_j}loss(w)$
Through further complicated mathematical steps we find another possible formula we can use to search for the minimum: $w_j = w_j + \eta \sum_{i=1}^{n}y_ix_{ij}exp(-y_i(w \cdot x_i + b))$
For each example $x_i$ we can take $w_j = w_j + \eta y_ix_{ij}exp(-y_i(w \cdot x_i + b))$ and for simplicity's sake we'll say $c = \eta exp(-y_i(w \cdot x_i + b))$.

```ad-question
title: When is $c$ large/small?
Reminding ourselves that $\eta$ is the learning rate, $y_i$ is the label, and $w \cdot x_i + b$ is the prediction, we can see that if the label and the prediction have the same sign then the size of the step we'll take will grow as the prediction grows. On the other hand, if the signs differ, then the size of the step will grow with how different the values are.
```

### Gradient

The gradient is the vector of partial derivatives with respect to all the coordinates of the weights: $\Delta_ WL = [\frac{\delta L}{\delta w_1}\frac{\delta L}{\delta w_2}...\frac{\delta L}{\delta w_N}]$

```ad-note
Not all optimisation problems are convex; they often have local minimums.
```

```ad-problem
title: Saddle points
At a saddle point, some directions curve upwards and others curve downwards. Here the gradient is zero even though we aren't at a minimum, meaning we get stuck if we fall exactly on the saddle point. If we can move slightly to one side we can get unstuck. Saddle points are very common in high dimensions.
![[gradient-descent-saddle-point.png]]
```

```ad-warning
title: The learning rate $\eta$ is a very important hyper-parameter
Choosing an appropriate value for the learning rate is vital to getting getting good results. As evidenced in the picture below, setting it too low will guarantee finding the exact minimum but will also make the whole process take longer than it has to. Setting it too high may make it impossible to apply the algorithm properly in more extreme cases.
![[gradient-descent-learning-rate.png]]
```

## #Regularisation

A #regulariser is a criterion we can add to the loss function of a model to make sure we don't overfit. It's a bias on the model that forces the learning to prefer certain types of weights over others $argmin_{w,b} \sum_{i=1}^{n} loss(yy^{'}) + \lambda\ regulariser(w, b)$

We generally want to avoid having huge weights, as a relatively small change in a feature could have a drastic impact on the prediction.

### #P-Norm

$r(w, b) = \sqrt[p]{\sum{|w_j|^p}} = ||w||^p$
Smaller values of $p$^[$p<2$] will encourage sparser vectors, whereas larger values will discourage large weights more.
Sum of weights^[1-norm/L1]: $r(w, b) = \sum{|w_j|}$
Sum of squared weights^[2-norm/L2]: $r(w, b) = \sqrt{\sum{|w_j|^2}}$

In jargon, 1-norm is referred to as "lasso" and 2-norm is known as "ridge".

L1 and L2 regularised cost functions are relatively easy to graphically represent in two dimensions; in the below diagram the L1 and L2 "balls" are displayed on the origin of the $(w_1, w_2)$ plane.
![[regularisation-pnorm.png]]
The optimal solution is where the contour first intersects the norm ball. Note that sparsity occurs in the corners of these balls. Due to this, we can intuitively imply that L2 is a good regularisation of the problem as we are much less likely to encounter scarsity.

```ad-tldr
title: TL;DR
L1 will tend to generate a small number of features setting all other features to $0$, while L2 will select more features which will all be close to 0.In fact, in the above diagram we can see that $w_1$ is 0.
```

### #Minimisation

If we can ensure that the sum of the loss function and the regulariser is convex then we can still use gradient descent: $argmin_{w, b} \sum_{i=1}^{n}loss(yy^{'}) + \lambda\ regulariser(w)$

#### #Convexity

```ad-definition
The line segment between any two points on the function is *above* the function.
```

Mathematically, $f$ is convex if for all $x_1, x_2$, $f(tx_1 + (1 - t)x_2) \le tf(x_1) + (1 - t)f(x_2)\ \forall\ 0 < t < 1$ where $f(tx_1 + (1 - t)x_2)$ is the value of the function at some point between $x_1$ and $x_2$ and $tf(x_1) + (1 - t)f(x_2)$ is the value at some point on the line segment between $x_1$ and $x_2$.

```ad-note
If both the loss function and the regulariser are convex, then their sum will also be convex.
```

```ad-note
P-norms are convex for $p \ge 1$
```

### #Gradient-Descent

We have the following optimisation criterion $argmin_{w, b} \sum_{i=1}^{n}{exp(-y_i(w \cdot x_i + b)) + \frac{\lambda}{2}||w||^2}$ where the loss function ^[$exp(-y_i(w \cdot x_i + b))$] penalises examples where the prediction is different to the label and the regulariser^[$||w||^2$] penalises large weights.

### Regularisation with #P-norms

- L1: $w_j = w_j +  \eta(loss\_correction\ - \lambda\ sign(w_j))$
- L2: $w_j = w_j +  \eta(loss\_correction\ - \lambda w_j)$
- Lp: $w_j = w_j +  \eta(loss\_correction\ - \lambda cw_{j}^{p-1})$

### Summary

- L1 is popular because it tends to result in sparse solutions^[lots of zero-weights]. However, it is not differentiable, so it only works for gradient descent solvers.
- L2 is also popular because for some loss functions it can be solved in a "single" step^[no gradient descent required, though solvers are often iterative anyway]
- Lp is less popular since the weights don't tend to be shrunk sufficiently.
