import math

class Value:
  def __init__(self, data, children=(), _op="", label=""):
    self.data = data
    self.grad = 0.0
    self._prev = set(children)
    self._backward = lambda: None
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __radd__(self, other):
    return self + other

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad

    out._backward = _backward

    return out

  
  def __rmul__(self, other):
    return self * other
  
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward

    return out

  def __neg__(self):
    return self * -1

  def __rsub__(self, other):
    return self + (-other)

  def __sub__(self, other):
    return self + (-other)

  def __truediv__(self, other):
    return self * (other**-1.0)

  def __pow__(self, other):
    assert isinstance(other, (int, float)),"Only supporting int/float"
    out = Value(self.data ** other, (self,), f"**{other}")

    def _backward():
      self.grad += (other*(self.data)**(other-1.0)) * out.grad

    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data))
    out = Value(t, (self, ), "tanh")

    def _backward():
      self.grad += (1.0 - t**2) * out.grad

    out._backward = _backward

    return out
  

  def backward(self):
    self.grad = 1.0
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    for node in reversed(topo):
      node._backward()






import random

class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1.0, 1.0)) for i in range(nin)]
    self.b = Value(random.uniform(-1.0, 1.0))

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]


class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for i in range(nout)]


  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    params = []
    for n in self.neurons:
      ps = n.parameters()
      params.extend(ps)

    return params


class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    params = []
    for l in self.layers:
      ps = l.parameters()
      params.extend(ps)

    return params
  




x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


for k in range(1000):
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0

  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad

  print(k, loss)
  