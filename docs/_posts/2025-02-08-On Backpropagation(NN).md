---
layout: post
title:  "On Backpropagation (Applications to Neural Networks)"
date:   2025-02-08 21:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
usemathjax: true
---
<script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
## Introduction
This post begins by providing a definition for the `value` class, which stores the data and methods associated with each node of the network. I then provide explanations for finding the derivatives of the network output with respect to the nodes. Finally, we will use this knowledge to implement a `backward` method for the `value` class, to implement backpropagation.

## Value Class
<img src="/assets/images/valueClassDiagram.png" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>

The `value` class contains methods and attributes that allow you to create expressions. The `backward` method within the 
`value` class automatically backpropagates through expressions that are made using `value` objects. 
This allows the `data` attribute of the `value` object to be changed in a way that has a predictable effect on 
the overall output of the expression.

### \__init__
What is going on when the `value` class is initialised?
```python
def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
```
`self.data` is set to `data` and `self.label` is set to `label` (both parameters passed in when the object is initialised). `label` is what is displayed for the `value` object when it is within the graphical representation of the expression. `self.grad` is initialised to zero as this means that changing the `data` of the object will have no effect on the output of the expression, which initially is the behaviour that we want. `self._backward` is defined as a function that by default returns `None`. We will revisit `_backward` and `backward` later in this article. `self._prev` contains the references to `value` objects that were used to generate the current node, and `self._op` stores the operation that was used on those previous nodes to generate the current node.

### \__add__ (and \__mul__ and \__repr__)
All of these methods are called magic methods. Python contains a lot of magic methods (including `__init__` which we saw above) but we will focus on only these three in this section. [This article][MagicMethodsArticle] by Rafe Kettler on magic methods explains them very well. I used it when writing this section of the article to improve my explanations.

When two `value` objects are summed together, Python needs to know how to handle this. This is done using the `__add__` magic method - it defines behaviour under the `+` operator.

```python
# add method within value class
def __add__(self, other):
# checks if other is a constant or a value, 
# if a constant that makes into a value object
        other = other if isinstance(other, Value) else Value(other)
# the output of the "+": the data of each 
# value object are summed together.
        out = Value(self.data + other.data, (self, other), '+')
		
# ignore this for now: (used for calculating 
# the gradient automatically during backpropagation)
# ------------------------------
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
# ------------------------------
        return out

# define two value objects
a = Value(2.0,label='a')
b = Value(3.0,label='b')

# writing this
c = a + b
# is equivalent to writing this
c = a.__add__(b)
```
When two `value` objects are multiplied a similar process happens compared to when two `value` objects are summed together. The only difference is now the output is the data of the objects multiplied together instead of added, and the `_backward` function looks a bit different.

`__repr__` defines the behavior of the `value` class when it is output in the notebook. `__repr__` outputs the 
`data` and `grad` attributes of the `value` object as a string, rather than a memory location.
```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```
There are more magic methods in the `value` class but they follow the same principles that I have described above.

## The data and grad attributes
Before we go any further, there are two important attributes to that we need to understand - the `data` and `grad` attributes.
- `data`: The current numerical value of this node in the expression.
- `grad`: This is the derivative of the final output of the expression with respect to the current node.

## Backprop by hand through Andrej's Graph
Karpathy uses a graphical visualisation of expressions (shown in the image below) to improve intuition around finding derivatives at each node. The expression used in his video is shown below, along with the graph that shows how nodes are combined with operations. We will move through this expression from right to left filling in the `grad` attributes for each `value` object.

In all cases, we are finding the derivative of the output of the expression (L) with respect to the node that we are currently focusing on. This is so that we know how changing the `data` of the node will affect the output of the expression.
```python 
e = a * b
d = e + c
L = d * f
```
<img src="/assets/images/expressionGraph.png" style="padding-right:10px"/> 

### dL/dL
The first node that we will find the derivative of is L with respect to L (how much does L change if I change L by h - think back to differentiation by first principles). L changes by h if you change L by h.

So, the rate of change of L with respect to L is constant for any value of L, dL/dL = 1.

### dL/dd and dL/df
Next we need to find the derivative of L with respect to d. This is like saying "How does L change when we bump d by h". Looking back at our expression, we can see that L = d * f so dL/dd is f. Similarly, dL/df = d. So, `f.grad` = `d.data` and `d.grad` = `f.data`. Neat!

### THE MOST IMPORTANT NODE TO UNDERSTAND (Using the chain rule)
This is where things start to get a bit more interesting. We now want to find the derivative of L with respect to c: dL/dc. We know d = c + e and L = d * f so we know how d will change if we change c by a bit and we know how L will change if we change d. 
This is the intuition behind the chain rule. The Wikipedia article on the [chain rule][ChainRuleArticle] has a nice explanation on this: "If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 * 4 = 8 times as fast as the man."
So lets begin by working out the impact of c on d. dd/dc = 1.
Next we need to work out the impact of d on L: dL/dd = f. Using the chain rule, we know that dL/dc = dd/dc * dL/dd = 1 * f = f
Similarly, dL/de = dd/de * dL/dd = 1 * f = dL/dd.
Looking back at the expression in graph form, we see that the two nodes combined under the + operation now have the same `grad`:
<img src="/assets/images/expressionPlusDemo.png">


### Applying the chain rule again
The next derivates we want to find are L with respect to a and with respect to b: dL/da and dL/db. The "local" expression is e = a * b. As we saw at the plus node, if we know how a affects e, and how e affects d etc, we can work out how a affects L. We have already worked out how e affects L so to find how a affects L, we need to find de/da and multiply that by dL/de. We can call de/da the local gradient. To find the "global" gradient of the expression, we do (the local gradient) * (the gradient of the output of the local expression). So dL/da = de/da * dL/de and dL/db = de/db * dL/de. dL/da =  b * f, dL/db = a * f.


## Optimisation
Now that we have the gradient at each node, we know how changing the value of the node to be more positive or more negative will effect 
the overall output of the expression (if the gradient is positive, increasing the value will increase the output of the expression). 


When training neural networks, given an input we know what the output should be. When an input is put into the neural network (neural networks 
are big expressions), the output is compared to what the output should be given the input. This 
is quantified using a loss function. We will go into more detail about loss functions later. The goal is to minimise the value produced 
by the loss function which can be done by changing the nodes in the expression - specifically, the weights. We know how to change the 
weights as we know the gradients at each node. This is how we optimise neural networks.


## Backpropagation through a neuron
<img src="/assets/images/neurons.png" style="padding-right:10px"/>
Image from ["towards data science neurons in perceptrons"][towardsDSNeurons]

A neural network is made up of neurons. A neuron is an abstraction for a function. The function takes in some inputs. Each input is multiplied by a different weight stored in the neuron. The outputs of these multiplications are then summed together with a bias. The output of this sum is then put through an activiation function like tanh or sigmoid which squashes its value between 1 and -1 (different activation functions can be used, but we will be using tanh).

This is the tanh function. Notice how the higher the magnitude the input to the function is, the closer to one the output is. However, there are diminishing returns - closer to zero an increase in the input will increase the output more than if you are further from zero and increase the input.
<img src="/assets/images/tanh.png" style="padding-right:10px"/>
This is the sigmoid function. It is similar to tanh, except it is bounded between 0 and 1 rather than between -1 and 1.
<img src="/assets/images/sigmoid.png" style="padding-right:10px"/> 

### Tanh

tanh(x) = (e^2x - 1)/(e^2x + 1) 

Currently, the `value` object can only handle addition, multiplication and subtraction. If we were to break down the mathematical tanh function into individual operations and backpropagate through it that way, we would have to implement exponentiation and division methods within the class. To avoid doing this, we can use the `math.tanh` function built into Python. 
The local derivative of tanh(x) is 1-tanh(x)^2 so we can implement the `_backward` function in the value class like this:
```python
# t = tanh(self.data)
def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
```

### Backpropagation through neuron

<img src="/assets/images/neuronExpressionGraph.png">

We now have enough knowledge to backpropagate through an actual neuron within a neural network. Look back at the image of the neuron to remind yourself of the expression. Before we begin, it should be noted that we don't need to work out the gradients at the inputs (x1, x2, ...) as these are not parameters of the network. We can only change the weights to affect the output of the network so we need to find the derivative of the weights with respect to the output of the neuron.

When backpropagating, we start with the last part of the expression graph, and in the case of a neuron, that is the tanh: o = tanh(n) where o is the output and n is the input to the tanh from the previous part of the expression graph: do/dn = 1 - o^2

Next is the plus node which distributes the gradient of the result of the operation to the nodes previous to it, as we saw in our first expression example: do/d(x2w2) = do/d(x1w1) = 0.5

Each of the inputs (in this case x1 and x2) are multiplied by weights (in this case w1 and w2) in the neuron. Using the chain rule, we can find the derivative of the weights with respect to the output:

x2w2 = x2 * w2

d(x2w2)/d(w2) = x2 (local derivative)

do/d(x2w2) = 0.5

do/d(w2) = do/d(x2w2) * d(x2w2)/d(w2) = 0.5 * x2

This is the derivative of the output of the neuron with respect to w2! Repeat for the other weights and we now know how changing any of the weights will change the output of the neuron.

If this was challenging, I recommend going back through the first expression graph and recalculating the derivatives, and to look at the intuitive explanation on the Wikipedia page for the [chain rule][ChainRuleArticle]

## Implementing the backward pass automatically!
In this section of this post, we will be adding functions to the operation methods within the value class. Eventually, this will allow `.backward` to be called on any node in the expression and the derivative of all previous nodes with respect to that node will be calculated.

### \__Add__
The add method takes two value objects - `self` and `other`. The `_backward` method should set `self.grad` and `other.grad`. We know that the plus node distributes the gradient of the output of the node to its previous nodes, so:

```python
self.grad = out.grad
other.grad = out.grad
```

### \__Mul__
The `_backward` function for mul is a slightly more involved than for the add method. We need to find the local gradient and multiply it by the gradient of the output to get the gradient at the node with respect to the output of the expression:

```python
out = Value(other.data * self.data, (self,other))
# d(out)/d(other) = self.data
# d(out)/d(self) = other.data
self.grad = other.data * out.grad
other.grad = self.data * out.grad
```


### Tanh
We have `out.grad` and we need to chain it into `self.grad` - this time we don't have an `other.grad` as tanh only takes one input.

```python
out = Value(tanh(self.data), (self))
#d(out)/d(self) = 1 - tanh(self.data)^2
self.grad = 1 - tanh(self.data)^2 * out.grad
```


### Doing the pass
Now that we have implemented `_backward` function for all the operation methods, we can implement a `backward` method for the value 
class. `backward` is different to `_backward` as `backward` can be called on a value object whereas `_backward` is specific to different 
operations and is called by the `backward` method as part of the backward pass (to find the gradients of the nodes).


First we set `self.grad` to 1 as we know d(self.data)/d(self.data) = 1. We then move backwards through the expression, calling the 
`backward` method on each node. The order of this backward pass is generated by a [topological sort][topoSort]. This sort ensures that every node 
that has `backward` called on it, has an `out.grad`.


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
There is currently a bug in our implementation. If we have the expression `b = a + a`, we get a `b.grad` = 1 not 2. This is because 
`self.grad` is set to 1 and `other.grad` is set to 1 but other and self are the same object in this case, so `other.grad` overrides 
`self.grad`. They should accumulate not override, so in all our previous `_backward` method implementations, we should use `+=` not just 
`=`. 

```python
#example: __add__ _backward should be written like this:
self.grad += out.grad
other.grad += out.grad
```


## Breaking down tanh (a fun exercise!)

tanh(x) = (e^2x - 1)/(e^2x + 1)


### Adding constant functionality
In order to implement tanh with its individual components, we need to be able to apply operations between constants and `value` objects. At the moment we can only apply operations between two `value` objects. When a `value` object is added to a constant, let `c` = a constant, Python is trying to access `c.data`
but that does not exist. To fix this, we can check whether `other` `isinstance` of a `value`, and if it isn't, wrap the constant into a `value` object. This fix works, but only when the constant is applied to the `value` object, not when the `value` object is applied to `c`. To fix this, we can define `__radd__`, `__rmul__`, etc methods within the `value` class. These methods are so that if `c.__add__(valueobject)` is called, the interpreter knows how to handle the reversed order compared to the `__add__` method.

### Exponentials
Next we need to add exponentials to our `value` class. We can use `math.exp(self.data)` for the output of the object. The local derivative stays as e^x as d(e^x)/dx = e^x so
```python
self.grad = out.data * out.grad
```
### Division
Finally, division is just a special case of multiplication where the divisor (in the multiplication) has been raised to the power of -1. This means for our implementation of division, we will really just be implementing a power function. For the output we can use
`self.data**other` (not `other.data` as we are only handling to the power of a constant not a `value` object).
```python
self.grad = other * self.data**(other - 1) * out.grad
```

## Summary
In this blog post, we have gone through Micrograd's implementation of the value class, with a focus on how the `backward` method works. We have used our knowledge of the chain rule to backpropagate by hand first through an example expression and through a neuron that would be used as part of a wider neural network.


[MagicMethodsArticle]: https://rszalski.github.io/magicmethods/
[ChainRuleArticle]: https://en.wikipedia.org/wiki/Chain_rule
[towardsDSNeurons]: https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc
[topoSort]: https://en.wikipedia.org/wiki/Topological_sorting