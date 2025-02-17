---
layout: post
title:  "From Micrograd to PyTorch (and building a neural network) IN PROGRESS"
date:   2025-02-10 21:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---
## Introduction

In this post we will compare the PyTorch and Micrograd APIs, build a small neural network and train it using a small dataset.

There will be many exercises within this post to lead you through building your own neural network using the Value class that we discussed in the previous post.

## A comparison to PyTorch
The main difference between Micrograd and PyTorch is that PyTorch is based around tensors rather than scalars. Tensors are multidimensional arrays of scalars. This is necessary as it improves the speed of passes through a neural network (we will go into the passes later in this post). This is through making better use of memory bandwidth and computing outputs of neurons in parallel.

## Building a neural net library in Micrograd
Exercises: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobypullan/tobypullan.github.io/blob/main/Micrograd_p3s1.ipynb)


Answers: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobypullan/tobypullan.github.io/blob/main/Micrograd_p3s1_ANS.ipynb)


In this exercise we will begin by creating a neuron, layer and multi-layer perceptron (a type of neural network).

## Creating a tiny dataset and writing the loss function
For our MLP, our tiny dataset is a 2D list of numbers, `xs`. Each input within the 2D list of number should produce the corresponding output from the targets list, `ys`

```python
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
```

When we input a list from `xs` into the MLP, we get a single value out. We want a single number to calculate the total performance of the MLP - this number is called the loss. In this case we will be using the mean square error loss:

```python
# ypred is the output the MLP produced for the input from xs
# ygt (y ground truth) is the actual value the MLP should have output
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

The loss is the sum of the squared differences of the output of the MLP and the actual value it should have output. We find the difference between the output and the "ground truth" value so that when the values are very similar, the loss is minimised, the goal is to minimise the loss. The difference is squared to remove any negative numbers and the squared differences are summed as we want to minimise the difference between all the outputs and actual values - not just one.

## Collecting all the parameters of the neural net

## Optimising the network manually

## Summary of what we learned and how to go towards modern neural nets

## How PyTorch implements tanh

## Conclusion
doing the same thing but in PyTorch: comparison
01:43:55 building out a neural net library (multi-layer perceptron) in micrograd
01:51:04 creating a tiny dataset, writing the loss function
01:57:56 collecting all of the parameters of the neural net
02:01:12 doing gradient descent optimization manually, training the network
02:14:03 summary of what we learned, how to go towards modern neural nets
02:16:46 walkthrough of the full code of micrograd on github
02:21:10 real stuff: diving into PyTorch, finding their backward pass for tanh
02:24:39 conclusion
02:25:20 outtakes :)
