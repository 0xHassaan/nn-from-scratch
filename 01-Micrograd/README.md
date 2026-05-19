# micrograd

A tiny autograd engine in pure Python, following Karpathy's Neural Network Series.

## What it does
Builds a computational graph of operations and runs 
backpropagation through it automatically.

## What I learned
- Backprop is just the chain rule applied recursively through a graph. 
  Each operation knows how to pass gradients backward to its inputs.

- Training a neural network is just one number (loss) going down. 
  Every weight nudges itself in opposite direction of the gradient to make that happen.


## Files
- `micrograd.py` — the engine