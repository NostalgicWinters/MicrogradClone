# MicrogradClone

A lightweight implementation of an **automatic differentiation (autograd) engine** and **neural network library** built entirely from scratch. This project is designed to help understand computational graphs, reverse-mode backpropagation, and the mathematical foundations of deep learning without relying on frameworks like PyTorch or TensorFlow.

---

## Overview

**MicrogradClone** is an educational deep learning project inspired by **Andrej Karpathy's Micrograd**. It recreates a minimal automatic differentiation engine capable of constructing computational graphs and computing gradients through reverse-mode automatic differentiation.

The primary objective of this project is to provide an intuitive understanding of how neural networks learn by implementing every core component from first principles.

---

## Features

* Automatic differentiation (Autograd) engine
* Dynamic computational graph construction
* Reverse-mode backpropagation
* Scalar-based neural network implementation
* Support for common activation functions
* Gradient computation using the chain rule
* Clean, minimal, and beginner-friendly codebase

---

## Core Concepts

This project demonstrates the implementation of:

* Neural Networks
* Computational Graphs
* Reverse-Mode Automatic Differentiation
* Backpropagation
* Gradient Descent
* Chain Rule of Calculus

---

## Project Structure

```text
MicrogradClone/
├── main.ipynb                 # Main implementation and experiments
├── .ipynb_checkpoints/
│   └── main-checkpoint.ipynb
└── README.md
```

---

## How It Works

1. Create scalar values that automatically track operations.
2. Build a computational graph as mathematical operations are performed.
3. Execute reverse-mode backpropagation to compute gradients.
4. Update parameters using gradient descent.
5. Repeat the process to train simple neural networks.

---

## Motivation

Modern deep learning frameworks abstract away most of the underlying mathematics. This project focuses on understanding **what happens under the hood** by implementing the core mechanics manually.

It is intended for students, beginners, and anyone interested in learning the fundamentals of deep learning from first principles.

---

## Inspiration

This project is heavily inspired by:

* Andrej Karpathy
* Micrograd by Andrej Karpathy

---

## Future Improvements

Planned features include:

* Tensor-based implementation
* GPU acceleration
* Additional activation functions
* Loss functions
* Optimizers (SGD, Momentum, Adam)
* Mini-batch training
* Dataset examples
* Model serialization
* Computational graph visualization
* Unit tests and documentation

---

## Learning Outcomes

By building this project, you will gain practical experience with:

* How automatic differentiation works
* Neural network internals
* Gradient computation
* Backpropagation algorithms
* Training simple neural networks from scratch

---

## License

This project is intended for educational purposes. Feel free to fork, modify, and experiment with the code.

---

## Acknowledgements

Special thanks to **Andrej Karpathy** for creating **Micrograd**, an excellent educational resource that inspired this implementation.
