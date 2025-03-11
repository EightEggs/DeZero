import numpy as np


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"`data` must be a numpy array, but got {data.__class__.__name__}."
            )
        self.data = data
        self.grad = np.zeros_like(data)

    def __repr__(self):
        return f"Variable({self.data})"


class Function:
    def __call__(self, x: Variable) -> Variable:
        self.x = x
        _x = x.data
        _y = self.forward(_x)
        y = Variable(_y)
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
    x = Variable(np.array((0.5,)))
    f1 = Square()
    f2 = Exp()
    f3 = Square()
    y1 = f1(x)
    y2 = f2(y1)
    y3 = f3(y2)

    y3.grad = np.array((1.0,))
    y2.grad = f3.backward(y3.grad)
    y1.grad = f2.backward(y2.grad)
    x.grad = f1.backward(y1.grad)

    print(x.grad)
