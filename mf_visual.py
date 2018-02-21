import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, accuracy
from surprise import SVD

def matrix_factorization(data, m, n, k, lamb, eta, epochs, biased=False):
    f_U = np.ones(m)
    f_V = np.ones(n)
    # for i, j, _ in data:
    #     f_U[i] += 1
    #     f_V[j] += 1
    N = data.shape[0]

    U = np.random.rand(k, m) - 0.5
    V = np.random.rand(k, n) - 0.5
    mu, A, B = 0, np.zeros(m), np.zeros(n)
    if biased:
        mu = np.mean(data[:, 2])
        A = np.random.rand(m) - 0.5
        B = np.random.rand(n) - 0.5

    tol = 1e-4
    first_change = 0
    prev_err = get_err(data, U, V, A, B, mu, N, lamb)
    for epoch in range(epochs):
        idxs = np.random.permutation(N)
        for idx in idxs:
            i, j, y = data[idx]
            U[:, i] -= eta * (lamb / f_U[i] * U[:, i] - ((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])) * V[:, j])
            V[:, j] -= eta * (lamb / f_V[j] * V[:, j] - ((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])) * U[:, i])
            if biased:
                A[i] -= eta * (lamb / f_U[i] * A[i] - ((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])))
                B[j] -= eta * (lamb / f_V[j] * B[j] - ((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])))
        err = get_err(data, U, V, A, B, mu, N, lamb)
        print('EPOCH', epoch + 1, err)
        change = np.abs(err - prev_err)
        prev_err = err
        if epoch == 0:
            first_change = change
        if change / first_change < tol:
            break

    return U, V, A, B

def get_err(data, U, V, A, B, mu, N, lamb):
    loss_reg = lamb / 2 * sum([np.linalg.norm(x) ** 2 for x in [U, V, A, B]])
    loss_data = 1 / 2 * sum([((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])) ** 2 for i, j, y in data])
    return (loss_reg + loss_data) / N

def score(data, U, V, biased=False, A=None, B=None, mu=None):
    if not biased:
        A = np.zeros(U.shape[1])
        B = np.zeros(V.shape[1])
        mu = 0

    err = sum([((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])) ** 2 for i, j, y in data])
    return err / len(data)

def plot(lambs, errs, title):
    plt.figure()
    plt.semilogx(lambs, errs)
    plt.title(title)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Mean squared error (MSE)')
    plt.savefig('figures/' + title)

def projection(X):
    _, _, V = np.linalg.svd(X, full_matrices=False)
    return V[:, 0:2]

k = 20

Y_train = np.loadtxt('data/train.txt', '\t') - np.array([1, 1, 0])
Y_test = np.loadtxt('data/test.txt', '\t') - np.array([1, 1, 0])

mu = np.mean(Y_train[:, 2])
lambs = np.logspace(-4, -4, 1)
epochs = 100
errs_unbiased = np.zeros(len(lambs))
errs_biased = np.zeros(len(lambs))
for iteration, lamb in enumerate(lambs):
    U_ub, V_ub, _, _ = matrix_factorization(Y_train, 943, 1682, k, lamb, 0.03, epochs)
    U_b, V_b, A_b, B_b = matrix_factorization(Y_train, 943, 1682, k, lamb, 0.03, epochs, True)
    err_unbiased = score(Y_test, U_ub, V_ub)
    err_biased = score(Y_test, U_b, V_b, True, A_b, B_b, mu)
    errs_unbiased[iteration] = err_unbiased
    errs_biased[iteration] = err_biased
    print('Test error (biased):', err_biased)
    print('Test error (unbiased):', err_unbiased)

# plot(lambs, errs_unbiased, 'Unbiased')
# plot(lambs, errs_biased, 'Biased')

reader = Reader()
data_train = Dataset.load_from_file('data/train.txt', reader=reader).build_full_trainset()
data_test = Dataset.load_from_file('data/test.txt', reader=reader).build_full_trainset().build_testset()
model = SVD(n_factors=k)
model.fit(data_train)
accuracy.rmse(model.test(data_test))
V = model.qi.T

Vs = [V_ub, V_b, V]
titles = ['Unbiased Projection', 'Biased Projection', 'Surprise Projection']
for V, title in zip(Vs, titles):
    P = projection(V)
    projs = np.dot(P.T, V)
    plt.figure()
    plt.plot(projs[0], projs[1], '.')
    plt.title(title)
    plt.savefig('figures/' + title)