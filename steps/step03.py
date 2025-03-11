import numpy as np
from step02 import Function, Square
from step01 import Variable


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)


if __name__ == "__main__":
    x = Variable(np.array((1.0, 2.0, 3.0)))
    f1 = Square()
    f2 = Exp()
    y = f2(f1(x))
    print(y)