---
layout: post
title:  "On Backpropagation (Applications to Neural Networks)"
date:   2024-11-21 15:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
usemathjax: true
---
<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
## Introduction
This post begins by providing a definition for the value class, along with an explanation of the different attributes and methods that it contains. I then provide explanations for the differential with respect to L for each variable in a larger expression which will help to build an intuitive understanding for backpropagation within an actual neural network. Finally, we will implement a backward method for the value class, that when called on a value within an expression, will calculate the gradient of each value before with respect to the value that it is called on.

## Value Class
<img src="/assets/images/valueClassDiagram.png" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>
The value class contains methods and attributes that allow you to create expressions. These expressions can be backpropagated through using the backward method. This allows the value of the object to be altered to be changed in a way that has a predictable effect on the overall output of the expression.

### \__init__
What is going on when the value class is initialised?
```python
def __init__(self, data, _children=(), _op='', label=''):
	self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
```
Clearly, self.data is set to data and self.label is set to label (both parameters passed in when the object is initialised). "label" is what is displayed for the value object when it is within the graphical representation of the expression. self.grad is initialised at zero as this means that changing the value of the object will have no effect on the output of the expression, which initially is the behaviour that we want. self._backward is defined as a function that by default returns None. We will revisit _backward and backward later in this article. self._prev contains the nodes that were used to generate the current node, and self._op stores the operation that was used on those previous nodes to generate the current node.

### \__add__ (and \__mul__ and \__repr__)
All of these methods are called magic methods. Python contains a lot of magic methods (including \__init__ which you saw above) but we will focus on only these three in this section. [This article][MagicMethodsArticle] by Rafe Kettler on magic methods explains them very well. I used it when writing this section of the article to improve my explanations.

When two value objects are summed together, Python needs to know how to handle this. This is done using the \__add__ magic method - it defines behaviour under the + operator.

```python
# add method within value class
def __add__(self, other):
		# checks if other is a constant or a value, if a constant that makes into a value object
        other = other if isinstance(other, Value) else Value(other)
		# the output of the "+": the data of each value object are summed together.
        out = Value(self.data + other.data, (self, other), '+')
		
		# ignore this for now: (used for calculating the gradient automatically during backpropagation)
		# -------------------------
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
		# -------------------------
        return out

# define two value objects
a = Value(2.0,label='a')
b = Value(3.0,label='b')

# writing this
c = a + b
# is equivalent to writing this
c = a.__add__(b)
```
When two value objects are multiplied a similar process happens compared to when two value objects are summed together. The only difference is now the output is the data of the objects multiplied together instead of added, and the backward function looks a bit different.

\__repr__ defines the behavior of the value class when it is output in the notebook. \__repr__ outputs the value of the object and the gradient with respect to the output as a string, rather than a memory location (which is much less helpful).
```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```
There are more magic methods in the value class but they follow the same principles that I have described above.

## Graph viz for expressions
- trace builds set of all nodes and edges
- draw_dot displays these nodes and edges, adds nodes where ops are for clarity

## Backprop by hand through Andrej's Graph
Karpathy uses a graphical visualisation of expressions (shown in the image below) to improve intuition around finding derivatives at each node with respect to the output. The expression used in his video is shown below, along with the graph that shows how nodes are combined with operations towards the output.
```python 
e = a * b
d = e + c
L = d * f
```
<img src="/assets/images/expressionGraph.png" style="padding-right:10px"/> 

### dL/dL
The first node to find the derivative of is L. We are trying to find how all the nodes effect L, so we are finding dL/dL. This is like saying how much does L change if I change L by h (think back to differentiation by first principles)? Funnily enough, L changes by h if you change L by h.

So, the rate of change of L with respect to L is constant, dL/dL = 1.

If you're not convinced, take an expression, say y=x^2. Now, plot x against x. You can see that it is a straight line with gradient 1, hence the derivative of x with respect to itself is 1. This graph is shown below:
<table>
<tr>
    <th>x</th>
    <th>y</th>
</tr>
<tr>
    <td>1</td>
    <td>1</td>
</tr>
<tr>
    <td>2</td>
    <td>4</td>
</tr>
<tr>
    <td>3</td>
    <td>9</td>
</tr>
<tr>
    <td>4</td>
    <td>16</td>
</tr>
<tr>
    <td>5</td>
    <td>25</td>
</tr>
</table>

<img src="/assets/images/xAgainstX.png" style="padding-right:10px"/> 

### dL/dD and dL/dF
Next we need to find the derivative of D with respect to L. This is like saying "How does L change when we bump D by h". Looking back at our expression, we can see that L = D * F so dL/dD is F. Similarly, dL/dF = D. So, F.grad = D.data and D.grad = F.data. Neat!

### MOST IMPORTANT NODE TO UNDERSTAND (Using the chain rule)
This is where things start to get a bit more interesting. We now want to find the derivative of C with respect to L: dL/dC. We know D = C + E and L = D * F so we know how D will change if we change C by a bit and we know how L will change if we change D. 
This is the intuition behind the chain rule. The Wikipedia article on the [chain rule][ChainRuleArticle] has a nice explanation on this: "If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 * 4 = 8 times as fast as the man."
So lets begin by working out the impact of C on D, which is another way of saying what is dD/dC. dD.dC = 1.
Next we need to work out the impact of D on L - dL/dD = F. Using the chain rule, we know that dL/dC = dD/dC * dL/dD = 1 * F = F
Similarly, dL/dE = dD/dE * dL/dD = 1 * F = dL/dD.
Looking back at the expression in graph form, we see that the two nodes combined under the + operation now have the same gradient:
<img src="/assets/images/expressionPlusDemo.png">


### Applying the chaian rule again
The next gradients we want to find is A and B, both with respect to L: dL/dA and dL/dB. The "local" expression is e = a * b. As we saw at the plus node, if we know how A affects E, and how E affects D etc, we can work out how A affects L. We have already worked out how E affects L so to find how a affects L, we need to find dE/dA and multiply that by dL/dE. We can call dE/dA the local gradient so to find the gradient at a node with respect to the output of the expression, we do the local gradient * the gradient of the output of the local expression with respect to the output. So dL/dA = dE/dA * dL/dE and dL/dB = dE/dB * dL/dE. dL/dA =  B * F, dL/dB = A * F.

## What have we learnt from doing this?
Iterated through all the nodes and locally applied the chain rule. We know what the local derivatives at each operation are. Using the chain rule, we can find how each node affects the output and this is backpropagation in action.

## Optimisation
Now that we have the gradient at each node, we know how changing the value of the node to be more positive or more negative will effect the overall output of the expression (if the gradient is positive, increasing the value will increase the output of the expression). The output of a neural network comes from a loss function. This function takes the output of the neural network and compares it with the actual output that it should have produced. The goal is to minimise the loss function to make the outputs of the network as close to the actual output it should have produced as possible. Using the gradients at each node, the "weights" in the neural network can be changed so that the loss function is minimised. This is how neural networks are optimised.

## Backpropagation through a neuron
<img src="/assets/images/neurons.png" style="padding-right:10px"/>
Image from ["towards data science neurons in perceptrons"][towardsDSNeurons]

A neural network is made up of neurons. A neuron is an abstraction for a function. The function takes in some inputs. Each input is multiplied by a different weight stored in the neuron. The outputs of these multiplications are then summed together with a bias. The output of this sum is then put through an activiation function like tanh or sigmoid which squashes its value between 1 and -1 (different activation functions can be used, but we will be using tanh). You can see this in the image above.

This is the tanh function. Notice how the higher the magnitude the input to the function is, the closer to one the output is. However, there are diminishing returns - closer to zero an increase in the input will increase the output more than if you are further from zero and increase the input.
<img src="/assets/images/tanh.png" style="padding-right:10px"/>
This is the sigmoid function. It is similar to tanh, except it is bounded between 0 and 1 rather than between -1 and 1.
<img src="/assets/images/sigmoid.png" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/> 

### Tanh
Tanh(x) = (e^2x - 1)/(e^2x + 1). Currently, the value object can only handle addition, multiplication and subtraction. If we were to break down the tanh function into individual operations and backpropagate through it that way, we would have to implement exponentiation and division methods within the class. To avoid doing this, we can use the math.tanh function built into Python. 
The local derivative of tanh(x) is 1-tanh(x)^2 so we can implement the backward function in the value class like this:
```python
def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
```

### Backpropagation through neuron

<img src="/assets/images/neuronExpressionGraph.png">

We now have enough knowledge to backpropagate through an actual neuron within a neural network. Look back at the image of the neuron to remind yourself of the expression that we will be going back through. Before we begin, it should be noted that we don't need to work out the gradients at the inputs (x1, x2, ...) as these are not parameters of the network. We can only change the weights to effect the output of the network so we need to find the derivative of the weights with respect to the output of the neuron.

When backpropagating, we start with the last part of the expression graph, and in the case of a neuron, that is the tanh: O = tanh(n) where O is the output and n is the input to the tanh from the previous part of the expression graph. dO/dN = 1 - O.data^2 

$$
\frac{\mathrm{d}o}{\mathrm{d}n} = 1 - O.data^2
$$

Next is the plus node which distributes the gradient of the result of the operation to the nodes previous to it, as we saw in our first expression example.

$$
\frac{\mathrm{d}O}{\mathrm{d}x2w2} = \frac{\mathrm{d}O}{\mathrm{d}x1w1} = 0.5
$$

Each of the inputs (in this case x1 and x2) are multiplied by weights (in this case w1 and w2) in the neuron. Using the chain rule, we can find the derivative of the weights with respect to the output:

$$
x2w2 = x2 * w2
\frac{\mathrm{d}x2w2}{\mathrm{d}x2} = w2 (local derivative)
\frac{\mathrm{d}O}{\mathrm{d}x2w2} = 0.5
\frac{\mathrm{d}O}{\mathrm{d}x2} = \frac{\mathrm{d}O}{\mathrm{d}x2w2} * \frac{\mathrm{d}x2w2}{\mathrm{d}x2} = 0.5 * w2
$$
And this is the derivative of w2 with respect to the output of the neuron! Repeat for the other weights and we now know how changing any of the weights will change the output of the neuron.

If this was challenging, I recommend going back through the first expression graph and recalculating the derivatives, and to look at the intuitive explanation on the Wikipedia page for the [chain rule][ChainRuleArticle]

## Implementing the backward pass automatically!
In this section of this post, we will be adding functions to the operation methods within the value class. Eventually, this will allow .backward to be called on any node in the expression and the derivative of all previous nodes with respect to that node will be calculated.

### Add
The add method takes two value objects - self and other. The backward method should set self.grad and other.grad. We know that the plus node distributes the gradient of the output of the node to its previous nodes, so:

$$
self.grad = out.grad
other.grad = out.grad
$$


### Mul
The backward function for mul is a slightly more involved than for the add method. We need to find the local gradient and multiply it by the gradient of the output:

$$
other = A
self = B
out = O
O = A * B
\frac{\mathrm{d}O}{\mathrm{d}A} = B (local gradient for A)
A.grad = B.val * out.grad
$$
(I found it easier to think about using A, B and O as variable names)


### Tanh
We have out.grad and we need to chain it into self.grad - this time we don't have an other.grad as tanh only takes one input.

$$
out = tanh(self.val)

\frac{\mathrm{d}out}{\mathrm{d}self} = 1 - tanh(self.val)^2 (local gradient)

self.grad = 1 - tanh(self.val)^2 * out.grad
$$

### Doing the pass
Now that we have implemented backward function for all the operation methods, we can implement a backward method for the value class. First we need to set self.grad to 1 as we know dx/dx = 1.

We can only call .backward on a node, until all the nodes after it have had .backward called on them. This means we need to sort the nodes in the expression graph to ensure that we don't call the backward function on a node who's output does not have a .grad.
This can be achieved using a [topological sort][topoSort]. This ensures that "for every directed edge (u,v) from vertex u to vertex v, u comes before v in the ordering" (if .backawrd is called on nodes in this order then there is not a case where .backward will be called on a node who's output has no .grad).

```python
# method of value class
def backward(self):   
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
```

### A bug!
There is currently a bug in our implementation. If we have the expression b = a + a, we get a gradient of 1 not 2. This is because self.grad is set to 1 and other.grad is set to 1 but other and self are the same object in this case, so other.grad overrides self.grad. They should accumulate not override. This bug appears when we use a variable more than once in an expression.
When we are backpropagating through the expression graph, we are actually doing partial derivatives *** CHECK ***
- What if b = a + a and call b.backward
- Get a gradient of 1 not 2
- This is because self.grad was set to 1 and then other.grad was set to 1 - other overrides self, they should accumulate not override
- We get the bug when using a variable more than once

## Breaking down tanh (a fun exercise!)

$$
\tanh(x) = \frac{e^2x - 1}{e^2x + 1}
$$

### Adding constant functionality
In order to implement tanh with its individual components, we need to apply operations with constants to value objects. At the moment we can only apply operations to two value objects. When a value object is added to a constant, Python is trying to access constant.data
but that does not exist. To fix this, we can check whether other isinstance of a value, and if it isn't, wrap the constant into a value object. This fix works, but only when the constant is applied to the value object, not when the value object is applied to the
constant. To fix this, we can define radd, rmul, etc methods within the value class. These methods are so that if constant.\__add__(valueobject) is called, the interpreter knows how to handle the reversed order compared to the \__add__ method.

### Exponentials
Next we need to add exponentials to our value class. We can use math.exp(self.data) for the output of the object. The local derivative stays as e^x as d(e^x)/dx = e^x so self.grad = out.data * out.grad.

### Division
Finally, division is just a special case of multiplication where the divisor (in the multiplication) has been raised to the power of -1. This means for our implementation of division, we will really just be implementing a power function. For the output we can use
self.data**other (not other.data as we are only handling to the power of a constant not a value object). The local derivative is other * self.data ** (other - 1) so self.grad = other * self.data ** (other - 1) * out.grad.

## Summary
In this blog post, we have gone through Micrograd's implementation of the value class, with a focus on how the backward method works. We have used our knowledge of the chain rule to backpropagate by hand first through an example expression and through a neuron that would be used as part of a wider neural network.


[MagicMethodsArticle]: https://rszalski.github.io/magicmethods/
[ChainRuleArticle]: https://en.wikipedia.org/wiki/Chain_rule
[towardsDSNeurons]: https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc
[topoSort]: https://en.wikipedia.org/wiki/Topological_sorting