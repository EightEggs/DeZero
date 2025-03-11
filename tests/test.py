if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import unittest as ut
from steps.step09 import *


class TestClass(ut.TestCase):
    def test_forward(self):
        x = Variable(np.array((2.0,)))
        y = exp(square(x))
        self.assertTrue(y.data == np.exp(np.square(2.0)))

    def test_backward(self):
        x = Variable(np.array((2.0,)))
        y = exp(square(x))
        y.backward()
        self.assertTrue(x.grad == 4.0 * np.exp(4.0))

    def numerical_diff(self, f, x, eps=1e-4):
        x0 = Variable(x + eps)
        x1 = Variable(x - eps)
        y0 = f(x0)
        y1 = f(x1)
        return (y0.data - y1.data) / (2 * eps)

    def test_gradient(self):
        x = Variable(np.random.randn(1))
        y = exp(square(x))
        y.backward()
        num_grad = self.numerical_diff(lambda x: exp(square(x)), x.data)
        self.assertTrue(np.allclose(x.grad, num_grad))


if __name__ == "__main__":
    ut.main()
