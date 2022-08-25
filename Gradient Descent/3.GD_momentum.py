# x(t + 1) = x(t) - v(t)
# v(t) phải vừa mang thông tin của độ dốc (đạo hàm), vừa mang thông tin của đà (v(t-1))
# v(t) = gamma.v(t - 1) + eta.f' (thường chọn gamma = 0.9)
# Example: f(x) = x^2 + 10.sin(x) -> f'(x) = 2x + 10.cos(x)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

def display_function(x = []):
    # f(x) = x^2 + 10.sin(x)
    x0 = np.linspace(-4, 6, 1000, endpoint = True)
    y0 = x0 ** 2 + 10 * np.sin(x0)

    y = [cost(a) for a in x]
    plt.plot(x, y, 'b.')
    plt.plot(x0, y0, 'y', linewidth = 2)
    plt.xlabel('Ox')
    plt.ylabel('Oy')
    plt.show()

def grad(x):
    return 2 * x + 10 * np.cos(x)

def cost(x):
    return x ** 2 + 10 * np.sin(x)

def gradient_descent(eta = .1):
    x = [5]
    
    while True:
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    
    return x

def GD_momentum(eta = .1, gamma = .9, max_count = 1e6):
    x = [5]
    v = np.zeros_like(x[-1])
    count = 0

    while count < max_count:
        count += 1
        v = gamma * v + eta * grad(x[-1])
        x_new = x[-1] - v
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    
    return x

x = gradient_descent(.1)
result = 'Solution x = %f, cost = %f, obtained after = %d iterations'
print(result %(x[-1], cost(x[-1]), len(x)))
display_function(x)

x = GD_momentum(.1, 0.9)
result = 'Solution x = %f, cost = %f, obtained after = %d iterations'
print(result %(x[-1], cost(x[-1]), len(x)))
display_function(x)