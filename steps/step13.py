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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        if not all(isinstance(x, Variable) for x in inputs):
            raise TypeError("`inputs` must be a list of Variables.")
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
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
    x0 = Variable(np.array((2,)))
    x1 = Variable(np.array((3,)))
    y = add(square(x0), x1)
    y.backward()
    print(x0.grad)
    print(x1.grad)
