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
    def __call__(self, inputs: list[Variable]) -> list[Variable]:
        if not all(isinstance(x, Variable) for x in inputs):
            raise TypeError("`inputs` must be a list of Variables.")
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(y) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.x = inputs
        self.y = outputs
        return outputs

    def forward(self, _xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()

    def backward(self, gys: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs: list[np.ndarray]) -> tuple[np.ndarray]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)


if __name__ == "__main__":
    x0 = Variable(np.array((0.1,)))
    x1 = Variable(np.array((0.2,)))
    y = Add()([x0, x1])
    print(y)
