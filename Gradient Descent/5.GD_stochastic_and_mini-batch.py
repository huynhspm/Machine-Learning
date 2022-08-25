import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

def display_result(w, x, y, error_list):
    visualize_data(x, y)
    visualize_model(w)
    plt.axis([0, 1, 0, 10])
    plt.xlabel('Ox')
    plt.ylabel('Oy')
    plt.show()

    visualize_error_list(error_list)

def visualize_data(x, y):
    plt.plot(x.T, y.T, 'b.')

def visualize_model(w):
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(0, 1, 2, endpoint = True)
    y0 = w_0 + w_1 * x0
    plt.plot(x0, y0, 'y', linewidth = 2)

def visualize_error_list(error_list):
    arr = [i for i in range(len(error_list))]
    arr = np.array([arr]).T
    plt.plot(arr, error_list, 'b')
    plt.xlabel('Ox')
    plt.ylabel('Oy')
    plt.show()

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

def GD_stochastic(x, y, learning_rate = .05, max_count = 1e4):
    X = extended_data(x)
    d = X.shape[1]
    w_init = np.random.randn(d, 1)
    w = [w_init]
    error_list = []
    count = 0
    iter_check_converged = 10

    while count < max_count:
        shuffle_id = np.random.permutation(N)
        for i in shuffle_id:
            count += 1
            xi = X[i, :].reshape(1, d)
            yi = y[i, :].reshape(1, 1)
            w_new = w[-1] - learning_rate * grad(w[-1], xi, yi) 
            w.append(w_new)
            error_list.append(cost(w[-1], xi, yi))
            if count % iter_check_converged == 0 and check_converged(w[-1], w[-iter_check_converged]):
                 return (w, error_list)
    return (w, error_list)

def create_mini_batch(X, y, batch_size):
    mini_batches = []
    N = X.shape[0]
    shuffle_id = np.random.permutation(N)
    n_mini_batches = (N + batch_size - 1) // batch_size

    for i in range(n_mini_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, N)
        true_id = shuffle_id[start : end]
        X_mini = X[true_id]
        y_mini = y[true_id]
        mini_batches.append((X_mini, y_mini))

    return mini_batches

def GD_mini_batch(x, y, learning_rate = .05, batch_size = 32, max_count = 1e4):
    X = extended_data(x)
    d = X.shape[1]
    w_init = np.random.randn(d, 1)
    error_list = []
    w = [w_init]
    count = 0
    iter_check_converged = 10

    if check_grad(w[-1], X, y) == False:
        print('error in grad')
        return

    while count < max_count:
        mini_batches = create_mini_batch(X, y, batch_size)
        for mini_batch in mini_batches:
            count += 1
            X_mini, y_mini = mini_batch
            w_new = w[-1] - learning_rate * grad( w[-1], X_mini, y_mini)
            w.append(w_new)
            error_list.append(cost(w[-1], X_mini, y_mini))
            if count % iter_check_converged == 0 and check_converged(w[-1], w[-iter_check_converged]):
                return (w, error_list)
    return (w, error_list)

N = 1000
x = np.random.rand(N, 1)
y = 3 * x + 4 + .2 * np.random.randn(N, 1)

result = 'Solution w = %s, obtained after = %d iterations'

(w, error_list) = GD_stochastic(x, y)
print(result %(w[-1].T, len(w)))
display_result(w[-1], x, y, error_list)

(w, error_list) = GD_mini_batch(x, y, batch_size=50)
print(result %(w[-1].T, len(w)))
display_result(w[-1], x, y, error_list)