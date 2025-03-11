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

        self.x = inputs
        self.y = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *_xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y


def add(x0: Variable, x1: Variable):
    return Add()(x0, x1)


if __name__ == "__main__":
    x0 = Variable(np.array((0.1,)))
    x1 = Variable(np.array((0.2,)))
    y = add(x0, x1)
    print(y)
