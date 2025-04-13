# Make Variable easier to use
from contextlib import contextmanager
import heapq
import re
import time
from tkinter import N
import weakref
import numpy as np


class Config:
    enable_backprop = True

    @staticmethod
    @contextmanager
    def no_backprop():
        Config.enable_backprop = False
        try:
            yield
        finally:
            Config.enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name: str = None):
        if not isinstance(name, str) and name is not None:
            raise TypeError("`name` must be a string or None.")
        if type(data) in (int, float, None, np.int32, np.int64, np.float32, np.float64):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Invalid type for data: {type(data).__name__}. ")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __repr__(self):
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Variable({p})"

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    def transpose(self, *axes):
        return Variable(self.data.transpose(*axes))

    def reshape(self, *shape):
        return Variable(self.data.reshape(*shape))

    def astype(self, dtype):
        return Variable(self.data.astype(dtype))

    def set_creator(self, func: "Function"):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if not Config.enable_backprop:
            raise RuntimeError("backprop is disabled by `Config.no_backprop()`.")

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                # 用负号使堆按generation从大到小排序
                heapq.heappush(funcs, (-f.generation, f))

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)[1]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

    def cleargrad(self):
        self.grad = None


def as_variable(x) -> Variable:
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.generation = 0

    def __lt__(self, other: "Function"):
        return self.generation < other.generation

    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        if Config.enable_backprop:
            self.generation = max(x.generation for x in inputs)
            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *_xs: np.ndarray):
        raise NotImplementedError()

    def backward(self, *gys: np.ndarray):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (gy, gy)


def add(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Add()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


def neg(x: Variable):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (gy, -gy)


def sub(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Sub()(x1, x0)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy * x1.data
        gx1 = gy * x0.data
        return gx0, gx1


def mul(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy / x1.data
        gx1 = -gy * x0.data / (x1.data**2)
        return gx0, gx1


def div(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Div()(x1, x0)


class Pow(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0**x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs
        gx0 = gy * x1.data * x0.data ** (x1.data - 1)
        gx1 = np.log(x0.data) * gy * x0.data**x1.data
        return gx0, gx1


def pow(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Pow()(x0, x1)


def rpow(x0: Variable, x1: Variable):
    x1 = as_variable(x1)
    return Pow()(x1, x0)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__neg__ = neg
Variable.__pow__ = pow
Variable.__rpow__ = rpow
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv


if __name__ == "__main__":
    a = Variable(np.array(2.0))
    print("a =", a)
    print("a + 3 =", a + 3)
    print("3 + a =", 3 + a)
    print("a * 3 =", a * 3)
    print("3 * a =", 3 * a)
    print("a - 3 =", a - 3)
    print("3 - a =", 3 - a)
    print("-a =", -a)
    print("a ** 3 =", a**3)
    print("3 ** a =", 3**a)
    print("a / 3 =", a / 3)
    print("3 / a =", 3 / a)
    print("a ** 0.5 =", a**0.5)
    y = a**3 + 2 * a - 4 / a + 3**a
    y.backward()
    print("y = a**3 + 2 * a - 4 / a + 3**a =", y)
    print("a.grad =", a.grad)

    def numerical_diff(f, x, eps=1e-6):
        y0 = f(x - eps)
        y1 = f(x + eps)
        return (y1 - y0) / (2 * eps)

    def f(x):
        return x**3 + 2 * x - 4 / x + 3**x

    print("numerical_diff(f, 2.0) =", numerical_diff(f, 2.0))
