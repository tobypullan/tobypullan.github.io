---
layout: post
title:  "Micrograd Exercises - Not Complete"
date:   2024-08-21 14:47:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---
## Why have I done this?
I really enjoyed watching Andrej Karpathy's [Neural Networks: Zero to Hero][zero2hero-series] video series. To gain a greater understanding from his videos, I decided to make a some exercises that correspond with the chapters within the video. This has helped me to understand his teaching in more depth, and I hope that it can help others to get a little bit more out of his (already brilliant) videos.

## How can I do these exercises?
By following this Google Collab link that will take you to the notebook with exercises for each chapter of the [building micrograd][mgrad-video] video.


## Breif overview of [micrograd][mgrad-gh]
### What does micrograd do
It implements backpropagation. Backpropagation is where you step backwards through a __neural network__ and find the gradient at each __neuron__ with respect to the __loss function__.

### What is neuron and a neural network?
In a neural network a neuron is an abstraction of a neuron in the human brain. It takes some inputs, multiplies each of those inputs by a "weight", adds a "bias" and outputs a single value. A neural network is made up of many of these neurons.

### What is a loss function?
The loss function outputs a single number that describes how close the output of the neural network is to the actual things that it should be outputing. You are able to differentiate each variable within this function, and all variables within the neural network, with respect to the loss. This allows you to find out how changing any of the values in the network would effect the loss.

[mgrad-video]: https://www.youtube.com/watch?v=VMj-3S1tku0
[mgrad-gh]: https://github.com/karpathy/micrograd
[zero2hero-series]: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ