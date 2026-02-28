
import math

class Value:
    def __init__(self, data, children = (), _op = "", label = ""):
        self.data = data
        self.children = children
        self._op = _op
        self._prev = set(children)
        self._backward = lambda:None 
        self.grad = 0
        self.label = label

    def has_children(self):
        return True if len(self._prev) > 0 else False

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        # print(out.children) # Commented out to avoid clutter

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f"** {other}")
        def _backward():
            self.grad += other * (self.data ** (other- 1)) * out.grad
        out._backward = _backward
        return  out
    
    def __rpow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self ** other
    
    def __neg__(self):
        return self * -1
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) *out.grad
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        return other / self

    def chain_rule(self):
        stack = []
        visited = set()
        def topological_sort(root):
            if root in visited: return
            visited.add(root)
            for child in root._prev:
                topological_sort(child)
            stack.append(root)

        topological_sort(self)
        self.grad = 1
        good_order = stack[::-1]
        for node in good_order:
            node._backward()

    def __repr__(self):
        return f"Value {self.data}"

# Test Case
x = Value(5.0); x.label = "x"
y = Value(4.0) ; y.label = "y"
d = Value(10.0); d.label = "d"

z = x - y ; z.label = "z"
l = d / z ; l.label  = "l"

l.chain_rule()

print(f"l.data: {l.data}")
print(f"l.grad: {l.grad}")
print(f"d.grad: {d.grad}")
print(f"z.grad: {z.grad}")
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")
