---
layout: post
title:  "On Backpropagation (Applications to Neural Networks)"
date:   2024-11-21 15:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---

# Introduction
This post begins by providing a definition for the value class, along with an explanation of the different attributes and methods that it contains. I then provide explanations for the differential with respect to L for each variable in a larger expression which will help to build an intuitive understanding for backpropagation within an actual neural network.
# Value Class
<img src="/assets/images/valueClassDiagram.png" style="padding-right:10px"/> 
- Add, mul, repr
- children
- prev
- each operation adds to set of children (adds values that produced it)
- op, stores operation that produced value
- label, for displaying value objects on graph representation of expression
- self.grad initially set to 0 so weight doesn't change
- (later self.backward - a function that !!!!!!!!!!!!!!!!!!!!! )
# Graph viz for expressions
- trace builds set of all nodes and edges
- draw_dot displays these nodes and edges, adds nodes where ops are for clarity
# Backprop by hand through Andrej's Graph
e = a * b
d = e + c
L = d * f
<img src="/assets/images/expressionGraph.png" style="padding-right:10px"/> 
## dL/dL
- How much does L change if I change L by h?
- Changes by h
- dL/dL - The rate of change of L with respect to L is constant so = 1
- Example:
y=x^2
|---|---|
|x  |  y|
|  1|  1|
|  2|  4|
|  3|  9|
|  4| 16|
|  5| 25|
	- Plot x against x
<img src="/assets/images/xAgainstX.png" style="padding-right:10px"/> 
	- dx/dx is 1
## dL/dD
- Derivative of L with respect to D
- How does L change when bump d by h
- L = D * F so dL/dD = F
- dL/dF = D
- f.grad = d.val
- d.grad = f.val
## MOST IMPORTANT NODE TO UNDERSTAND (Using the chain rule)
- dL/dC = ?
- We know how L is sensitive to D
- C goes through D to L
- So if we know impact of C on D and D on L, we know how C relates to D
- D = C + E
- dD/dC = 1
- dL/dD = F.val
- By chain rule dL/dC = F.val
- Wikipedia has a nice intuitive explanation
- dL/dE = ?
- Also = F.val as affects D the same amount that C affects D (dD/dC=dD/dE) 
- + operation distributes the gradient to the previous two nodes
## Another application of the chain rule
- dL/dA = ?
- dL/dB
- Chain rule says = (dL / dE) * (local gradient)
- local gradient = dE/dA
- Imagine you're the * node
- Only know that you did A * B = E
- So can work out local gradients, and then multiply by how E affects L (dL/dE) to get the gradient with respect to L of nodes A and B
- dE/dA = B.val
- dE/dB = A.val
# What have we learnt from doing this?
Iterated through all the nodes and locally applied the chain rule. We know what the local derivatives at each operation are. Using the chain rule, we can find how each node affects the output and this is backpropagation in action.

# Optimisation
- Nudge value in direction of the gradient to get an increase in L
- Easy!

# Backpropagation through a neuron
<img src="/assets/images/neurons.png" style="padding-right:10px"/> 
https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc
- A neural network is made up of neurons
- Neurons (in a neural network) multiply their inputs by weights stored in the neurons
- They sum the products of the inputs and weights and add a bias to that sum
- This sum is then put through an activation function 
- Usually a squashing function like sigmoid or tanh
<img src="/assets/images/tanh.png" style="padding-right:10px"/> 
<img src="/assets/images/sigmoid.png" style="padding-right:10px"/> 
## We are using tanh
- A hyperbolic function - needs more than just + and *
- Need exponentiation and division
- Could break down the tanh function into exponentiations and divisions
- But don't need to do that - only need to get the output of the tanh and the local derivative so that chain rule can be applied
- Local derivative of tanh(x) = 1-tanh(x)^2
## Backpropagation through neuron
- Finding the derivative of the weights with respect to output
- In a neural net, finding derivative with respect to output of the whole network through the loss function
- O = tanh(n)
- dO/dN = 1-o.data^2
- Next is a plus node, a plus distributes the gradient - look back at previous explanation
- dO/x2w2 = dO/dx1w1 = 0.5 (see diagram)
- x2 * w2 = x2w2
- d(x2w2)/dx2 = w2 (local derivative)
- dO/dx2w2 = 0.5 (from before)
- so dO/dx2 = d(x2w2)/dx2 * dO/dx2w2 = 0.5 * w2
- Chain rule again, I find it helpful to think about the intuitive explanation Wikipedia provides
- dO/dw2 = 0 as initially x2 set to 0 so changing w2 doesn't have any effect on the output as it is being multiplied by 0 
- Repeat for x1 and w1
# Implementing the backward pass automatically!
## Add
- out.backward is the function that propagates the gradient
- need to put gradient into self.grad and other.grad - think about in graph, other and self point into add node
- need to find local derivative, so with respect to the output
- self.grad = out.grad
- other.grad = out.grad
## Mul
- let other = A
- let self = B
- let out = O
- O = A * B
- local gradient (for A): dO/dA = B
- so A.grad = B.val * out.grad
- Repeat for B
- (Using letters made it easier for me to think through it)
## Tanh
- We have out.grad, need to chain it into self.grad
- LOut = tanh(self.val)
- dLOut/dSelf = 1-tanh(self.val)^2 = local derivative
- so dO/dSelf = (1-tanh(self.val)^2) * out.grad
## Doing the pass
- Remember to initially set the output gradient to 1 (dO/dO = 1)
- Can call node.backward at each value to get the gradient with respect to output
- Never call .backward on a node, until all nodes after it have had .backward called on them
- This can be achieved using topological sort - all edges go in only one direction
- Put this sort with value so that .backward (without underscore) can be called on a node and all the nodes before it will have gradient propagated through
## A bug!
- What if b = a + a and call b.backward
- Get a gradient of 1 not 2
- This is because self.grad was set to 1 and then other.grad was set to 1 - other overrides self, they should accumulate not override
- We get the bug when using a variable more than once
## Breaking down tanh (a fun exercise!)
### Adding constant functionality
- Cannot add a constant to a value object at the moment
- Python trying to access constant.data but that does not exist (eg 1.data)
- Fix this by checking whether other is an instance of a value, and if it isn't wrap the constant into a value object
- This doesn't work when constant * value object
- Use rmul to switch order of operands (rmul is called if mul doesn't work)
### Exponentials
- output is math.exp(self.data)
- local derivative: e^x stays as e^x 
- self.grad = out.data * out.grad
### Division
- Division is a special case of multiplication with the divisor (in the multiplication) raised to the power of -1
- So really implementing a power function
- Local derivative: other * self.data ** other - 1
- self.grad = local derivative * out.grad
