---
layout: post
title:  "On Backpropagation (Applications to Neural Networks)"
date:   2024-11-21 15:41:26 +0000
categories: Andrej Karpathy | Neural Networks Zero To Hero
---
## TODO
- _backward explanation

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
https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc
A neural network is made up of neurons. A neuron is an abstraction for a function. The function takes in some inputs. Each input is multiplied by a different weight stored in the neuron. The outputs of these multiplications are then summed together with a bias. The output of this sum is then put through an activiation function like tanh or sigmoid which squashes its value between 1 and -1 (different activation functions can be used, but we will be using tanh). You can see this in the image above.

This is the tanh function. Notice how the higher the magnitude the input to the function is, the closer to one the output is. However, there are diminishing returns - closer to zero an increase in the input will increase the output more than if you are further from zero and increase the input.
<img src="/assets/images/tanh.png" style="padding-right:10px"/>
This is the sigmoid function. It is similar to tanh, except it is bounded between 0 and 1 rather than between -1 and 1.
<img src="/assets/images/sigmoid.png" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/> 

### Tanh
- A hyperbolic function - needs more than just + and *
- Need exponentiation and division
- Could break down the tanh function into exponentiations and divisions
- But don't need to do that - only need to get the output of the tanh and the local derivative so that chain rule can be applied
- Local derivative of tanh(x) = 1-tanh(x)^2

### Backpropagation through neuron
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

## Implementing the backward pass automatically!
### Add
- out.backward is the function that propagates the gradient
- need to put gradient into self.grad and other.grad - think about in graph, other and self point into add node
- need to find local derivative, so with respect to the output
- self.grad = out.grad
- other.grad = out.grad

### Mul
- let other = A
- let self = B
- let out = O
- O = A * B
- local gradient (for A): dO/dA = B
- so A.grad = B.val * out.grad
- Repeat for B
- (Using letters made it easier for me to think through it)

### Tanh
- We have out.grad, need to chain it into self.grad
- LOut = tanh(self.val)
- dLOut/dSelf = 1-tanh(self.val)^2 = local derivative
- so dO/dSelf = (1-tanh(self.val)^2) * out.grad

### Doing the pass
- Remember to initially set the output gradient to 1 (dO/dO = 1)
- Can call node.backward at each value to get the gradient with respect to output
- Never call .backward on a node, until all nodes after it have had .backward called on them
- This can be achieved using topological sort - all edges go in only one direction
- Put this sort with value so that .backward (without underscore) can be called on a node and all the nodes before it will have gradient propagated through

### A bug!
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


[MagicMethodsArticle]: https://rszalski.github.io/magicmethods/
[ChainRuleArticle]: https://en.wikipedia.org/wiki/Chain_rule