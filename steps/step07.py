from typing import Optional
import numpy as np


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"`data` must be a numpy array, but got {data.__class__.__name__}."
            )
        self.data = data
        self.grad = None
        self.creator = None

    def __repr__(self):
        return f"Variable({self.data})"

    def set_creator(self, func: "Function"):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        f = self.creator
        if f is not None:
            x = f.x
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, x: Variable) -> Variable:
        _x = x.data
        _y = self.forward(_x)
        y = Variable(_y)
        y.set_creator(self)
        self.x = x
        self.y = y
        return y

    def forward(self, _x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, _x: np.ndarray) -> np.ndarray:
        return _x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.x.data
        return gy * 2 * x


class Exp(Function):
    def forward(self, _x: np.ndarray) -> np.ndarray:
        return np.exp(_x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.x.data
        return gy * np.exp(x)


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array((0.5,)))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator is C
    assert y.creator.x is b
    assert y.creator.x.creator is B
    assert y.creator.x.creator.x is a
    assert y.creator.x.creator.x.creator is A
    assert y.creator.x.creator.x.creator.x is x
    
    y.backward()

    print(x.grad)
