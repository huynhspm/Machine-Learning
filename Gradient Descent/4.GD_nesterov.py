# Ý tưởng cơ bản là dự đoán hướng đi trong tương lai, tức nhìn trước một bước! 
# Cụ thể, nếu sử dụng số hạng momentum - gamma.v(t - 1) để cập nhật thì ta có 
# thể xấp xỉ được vị trí tiếp theo của hòn bi là theta - gamma.v(t - 1) (coi v(t) = gamma.v(t - 1))
# chúng ta không đính kèm phần gradient ở đây vì sẽ sử dụng nó trong bước cuối cùng 
# Vậy, thay vì sử dụng gradient của điểm hiện tại, NAG đi trước một bước, sử dụng gradient của điểm tiếp theo

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

def display_result(x, y, w):
    visualize_data(x, y)
    visualize_model(w)
    plt.axis([0, 1, 0, 10])
    plt.xlabel('Ox')
    plt.ylabel('Oy')
    plt.show()

def visualize_data(x, y):
    plt.plot(x.T, y.T, 'b.')

def visualize_model(w):
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(0, 1, 2, endpoint = True)
    y0 = w_0 + w_1 * x0
    plt.plot(x0, y0, 'y', linewidth = 2)

def extended_data(x):
    N = x.shape[0]
    one = np.ones((N, 1))
    return np.concatenate((one, x), axis = 1)

def grad(w, X, y):
    N = X.shape[0]
    return 1/N * X.T.dot(X.dot(w) - y)

def cost(w, X, y):
    N = X.shape[0]
    return .5/N * np.linalg.norm(y - X.dot(w), 2) ** 2

def numerical_grad(w, X, y, eps = 1e-6):
    res = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        res[i] = (cost(w_p, X, y) - cost(w_n, X, y)) / (2 * eps)
    return res

def check_grad(w, X, y, eps = 1e-6):
    grad1 = grad(w, X, y)
    grad2 = numerical_grad(w, X, y)
    return np.linalg.norm(grad1 - grad2) < eps

def check_converged(this_w, last_w, eps = 1e-6):
    return np.linalg.norm(this_w - last_w) < eps

def GD_momentum(x, y, learning_rate = .5, gamma = .9, max_count = 1e6):
    X = extended_data(x)
    d = X.shape[1]
    w_init = np.random.randn(d, 1)
    w = [w_init]
    v = np.zeros_like(w_init)
    count = 0
    iter_check_converged = 5

    if check_grad(w[-1], X, y) == False:
        print('error in grad')
        return

    while count < max_count:
        count += 1
        v = gamma * v[-1] + learning_rate * grad(w[-1], X, y)
        w_new = w[-1] - v
        w.append(w_new)
        if count % iter_check_converged == 0 and check_converged(w[-1], w[-iter_check_converged]):
            return w
    return w

def GD_NAG(x, y, learning_rate = .5, gamma = .9, max_count = 1e6):
    X = extended_data(x)
    d = X.shape[1]
    w_init = np.random.randn(d, 1)
    w = [w_init]
    v = np.zeros_like(w_init)
    count = 0
    iter_check_converged = 5

    while count < max_count:
        count += 1
        approximate_w = w[-1] - gamma * v[-1]
        v = gamma * v[-1] + learning_rate * grad(approximate_w, X, y)
        w_new = w[-1] - v
        w.append(w_new)
        if count % iter_check_converged == 0 and check_converged(w[-1], w[-iter_check_converged]):
            return w
    
    return w

N = 1000
x = np.random.rand(N, 1)
y = 4 + 3 * x + .2 * np.random.randn(N, 1)

w = GD_momentum(x, y)
result = 'Momentum: Solution x = %s, obtained after = %d iterations'
print(result %(w[-1].T, len(w)))
display_result(x, y, w[-1])

w = GD_NAG(x, y)
result = 'NAG: Solution x = %s, obtained after = %d iterations'
print(result %(w[-1].T, len(w)))
display_result(x, y, w[-1])