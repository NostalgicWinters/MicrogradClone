A clone of Micrograd built by me for learning about the internal working of neural nets while following Andrej Karpathy's educational content.

### Micrograd
A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API.

## Features
- Scalar-based automatic differentiation engine
- Dynamic computation graph construction
- Reverse-mode backpropagation
- PyTorch-like API
- Simple neural network modules (MLP, Neuron, Layer)
- Training with gradient descent
- A draw_dot() function which lets you visualize your neural net

## Motivation
This project was built to deeply understand reverse-mode automatic differentiation and the internal mechanics of neural networks by implementing them from scratch, following Andrej Karpathy’s educational content.

## Example Usage
```python
from micrograd import Value

x = Value(2.0)
y = Value(-3.0)
z = x * y + x**2
z.backward()

print(x.grad, y.grad)
```

## Acknowledgements
This project is inspired by Andrej Karpathy’s Micrograd and his educational content.
Original repository: https://github.com/karpathy/micrograd

