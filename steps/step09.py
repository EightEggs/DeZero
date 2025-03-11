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
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            if f is None:
                continue
            x, y = f.x, f.y
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            if y.grad is None:
                y.grad = np.zeros_like(y.data)
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


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


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


if __name__ == "__main__":
    x = Variable(np.array((0.5,)))
    y = square(exp(square(x)))
    y.backward()

    print(x.grad)
