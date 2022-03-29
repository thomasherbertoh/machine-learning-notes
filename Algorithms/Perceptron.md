---
date created: 2022-03-25 09:20
date updated: 2022-03-25 10:19
tags:
  - '#converge'
  - '#algorithm'
  - '#linear-model'
---

# Perceptron

Perceptron is a simple machine learning #algorithm based on a ![[Linear Models|#linear-model]]

## Algorithm

```python
repeat until convergence:
	for each training example (f_1, f_2, ..., f_n, label):  # label = -1/1
		check if correct based on the current model
		if not correct, update all the weights:
				for each w_i:
					w_i = w_i + f_i * label
				b = b + label
```

The check on correctness could be performed assigning our prediction to a variable $prediction\ =\ b + \sum_{i=1}^{n}{w_if_i}$ and then checking this variable against the `label`, as in

```python
if prediction != label:
```

```ad-question
title: Which line does the perceptron find?
It isn't guaranteed that the perceptron will find the optimal line to satisfy the learning task, only that it will find *some* line that separates the data.
```

```ad-note
If the data we're learning from isn't linearly separable the algorithm clearly won't #converge , in which case an approach could be to impose a limit on the number of iterations. If we encounter this problem, though, we should really be asking ourselves if using a linear model is a good idea.
```

```ad-note
The sampling order can have profound effects on the outcome as seen in the image below, so when we say `for each training example (f_1, f_2, ..., f_n, label):` what we should really do is `randomly sample one example (f_1, f_2, ..., f_n, label):`.
![[perceptron-sampling-order.png]]
```
