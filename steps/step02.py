import numpy as np
from step01 import Variable


class Function:
    def __call__(self, x: Variable) -> Variable:
        _x = x.data
        _y = self.forward(_x)
        y = Variable(_y)
        return y

    def forward(self, _x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, _x):
        return np.square(_x)


if __name__ == "__main__":
    x = Variable(np.array((10, 20)))
    f = Square()
    y = f(x)
    print(type(y))
    print(y)
