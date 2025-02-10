---
layout: post
title:  "From Micrograd to PyTorch (and building a neural network) IN PROGRESS"
date:   2025-02-10 21:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---
## Introduction

In this post we will compare the PyTorch and Micrograd APIs, build a small neural network and train it using a small dataset.

## A comparison to PyTorch
The main difference between Micrograd and PyTorch is that PyTorch is based around tensors rather than scalars. Tensors are multidimensional arrays of scalars. This is necessary as it improves the speed of passes through a neural network (we will go into the passes later in this post). This is through making better use of memory bandwidth and computing outputs of neurons in parallel.

## Building a neural net library in Micrograd
Exercises: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobypullan/tobypullan.github.io/blob/main/Micrograd_p3s1.ipynb)
Answers: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tobypullan/tobypullan.github.io/blob/main/Micrograd_p3s1_ANS.ipynb)
We will begin by building a neuron class. We will use a similar syntax to the PyTorch API. The constructor takes a parameter 'nin' which stands for number of inputs. 

## Creating a tiny dataset and writing the loss function


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
