# f(x) = x^2 + 5sin(x) -> f'(x) = 2x + 5cos(x)
# x(t + 1) = x(t) - n.f'(x(t)) = x(t) - n.(2x(t) + 5cos(x(t)))

import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 2 * x + 5 * np.cos(x)

def cost(x):
    return x**2 + 5 * np.sin(x) 

def myGD(eta, x0):
    x = [x0]
    while True:
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x

x1 = myGD(.1, -5)
x2 = myGD(.1, 5)

result = 'Solution x = %f, cost = %f, obtained after = %d iterations'

print(result %(x1[-1], cost(x1[-1]), len(x1)))
print(result %(x2[-1], cost(x2[-1]), len(x2)))