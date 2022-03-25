---
date created: 2022-03-24 08:40
date updated: 2022-03-25 09:29
tags:
  - '#online-learning'
  - '#Dimensionality'
  - '#linearly-separable'
  - '#linear-model'
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
