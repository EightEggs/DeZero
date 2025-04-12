# 解决循环引用，提高代码效率
from contextlib import contextmanager
import heapq
import time
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
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"`data` must be a numpy array, but got {data.__class__.__name__}."
            )
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def __repr__(self):
        return f"Variable({self.data})"

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


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.generation = 0

    def __lt__(self, other: "Function"):
        return self.generation < other.generation

    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        if not all(isinstance(x, Variable) for x in inputs):
            raise TypeError("`inputs` must be a list of Variables.")
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
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x**2
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable):
    return Square()(x)


if __name__ == "__main__":
    start = time.time()
    for i in range(10000):
        x = Variable(np.random.randn(10000))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        x.cleargrad()
    end = time.time()
    print(f"Time: {end - start:.3f} seconds")

    with Config.no_backprop():
        x = Variable(np.random.randn(10000))
        a = square(x)
        y = add(square(a), square(a))
        print(y.data)
        print(x.grad)
