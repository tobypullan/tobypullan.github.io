---
layout: post
title:  "Demystifying Backpropagation - Not Complete"
date:   2024-08-21 14:47:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---

## What is micrograd?

Micrograd is a library that implements backpropagation, a key algorithm in training neural networks. 

## What is backpropagation?

Backpropagation is a method used to calculate gradients in neural networks. It works by:
1. Moving backwards through the network
2. Computing the gradient (rate of change) at each neuron
3. Relating these gradients to a loss function

## What are neurons and neural networks?

- A neuron is a basic unit in a neural network, inspired by biological neurons.
- It processes inputs by:
  1. Multiplying each input by a "weight"
  2. Summing these weighted inputs
  3. Adding a "bias" term
  4. Producing a single output value
- A neural network is composed of many interconnected neurons.

## What is a loss function?

A loss function:
- Outputs a single number
- Measures how well the neural network's predictions match the actual target values
- Allows us to calculate gradients for each variable in the network
- Helps us understand how changing any value in the network would affect the overall performance (loss)

By minimizing this loss function, we can train the neural network to make better predictions.

Link to [Google Colab notebook][Colab-link] for exercises.

[Colab-link]: https://colab.research.google.com/drive/19SoOLoxupP_PlFnobmR36mo6UOHnXfZl?usp=sharing
