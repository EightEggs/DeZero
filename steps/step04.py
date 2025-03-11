from typing import Callable
import numpy as np
from step03 import *


def numerical_diff(f: Callable[[Variable], Variable], x: Variable, eps: float = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == "__main__":
    x = Variable(np.array((0.5,)))

    def f(x):
        f1 = Square()
        f2 = Exp()
        f3 = Square()
        return f3(f2(f1(x)))

    print(f"x: {x}")
    print(f"f(x): {f(x)}")
    print(f"f'(x): {numerical_diff(f, x)}")
