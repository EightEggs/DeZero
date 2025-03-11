import numpy as np


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"`data` must be a numpy array, but got {data.__class__.__name__}."
            )
        self.data = data
    
    def __repr__(self):
        return f"Variable({self.data})"


if __name__ == "__main__":
    x = Variable(np.array((1.0, 2.0, 3.0)))
    print(x)
