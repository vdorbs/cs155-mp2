import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns
import matplotlib.cm as cm
from surprise import Dataset, accuracy, SVD
from surprise.model_selection import train_test_split

from get_best_and_popular import get_best_and_popular
from basic_visual import get_ratings_by_id


def matrix_factorization(data, m, n, k, lamb, eta, epochs, biased=False):
    f_U = np.zeros(m)
    f_V = np.zeros(n)
    for i, j, _ in data:
        f_U[i] += 1
        f_V[j] += 1
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
        change = prev_err - err
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

    err = 0.5 * sum([((y - mu) - (np.dot(U[:, i], V[:, j]) + A[i] + B[j])) ** 2 for i, j, y in data])
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
epochs = 3
lamb = 1
U_ub, V_ub, _, _ = matrix_factorization(Y_train, 943, 1682, k, lamb, 0.03, epochs)
U_b, V_b, A_b, B_b = matrix_factorization(Y_train, 943, 1682, k, lamb, 0.03, epochs, True)
err_unbiased = score(Y_test, U_ub, V_ub)
err_biased = score(Y_test, U_b, V_b, True, A_b, B_b, mu)
print('Test error (biased):', err_biased)
print('Test error (unbiased):', err_unbiased)

data_surprise = Dataset.load_builtin('ml-100k')
data_train, data_test = train_test_split(data_surprise, test_size=0.1)
model = SVD(n_factors=k)
model.fit(data_train)
rmse = accuracy.rmse(model.test(data_test))
print('Test error (SVD):', rmse ** 2 / 2)
V = model.qi.T

best, most_popular = get_best_and_popular()
movie_selection = {
    'Best Movies': best,
    'Most Popular Movies': most_popular
}

def scatterplot(x, y, color, indices, title):
    indices = [i - 1 for i in indices]
    fig, ax = plt.subplots(figsize=(10, 10))
    scale_x = max(x[indices]) - min(x[indices])
    scale_y = max(y[indices]) - min(y[indices])
    scatter = ax.scatter(x[indices], y[indices], c=color, s=75)
    for i in indices:
        ax.annotate(movie_titles[i],
                    (x[i], y[i]),
                    xytext=(x[i] - scale_x/10, y[i] - scale_y/20))
    cb = fig.colorbar(scatter, ax=ax, orientation='horizontal')
    cb.set_label('Average rating')
    plt.title('{} — {}'.format(title, selection))
    plt.savefig('figures/{} {}'.format(title, selection))



ratings_by_id = get_ratings_by_id()
movie_data = np.loadtxt('data/movies.txt', dtype=str, delimiter='\t')
movie_titles = movie_data[:,1]
movie_genres = movie_data[:, 2:]
movies_with_genre = list()
number_of_movie_genres = len(movie_genres[0])

for i in range(number_of_movie_genres):
    movies_with_genre.append(list())
    for j, movie in enumerate(movie_genres):
        if movie[i] == '1':
            movies_with_genre[i].append(int(j))

movie_subject_names = ['Unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary',
'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
colormap = plt.cm.get_cmap('viridis')

Vs = [V_ub, V_b, V]
titles = ['Unbiased Projection', 'Biased Projection', 'Surprise Projection']
for V, title in zip(Vs, titles):
    P = projection(V)
    projs = np.dot(P.T, V)
    # Generate average points:
    subject_xs = []
    subject_ys = []

    colors = sns.hls_palette(len(movies_with_genre), l=.4, s=1.0)
    cpals = [sns.light_palette(c, as_cmap=True) for c in colors]
    print("I am TChala")
    ax = None
    for movies, cpal in zip(movies_with_genre, cpals):
        x_average = np.average(projs[0][movies])
        y_average = np.average(projs[1][movies])
        subject_xs.append(x_average)
        subject_ys.append(y_average)

        #ax = sns.kdeplot(projs[0][movies], projs[1][movies], n_levels=5, cmap=cpal, 
                         #cut=10, alpha=.5, shade=True, shade_lowest=False)
    ax = sns.regplot(subject_xs, subject_ys)
    for (m, x, y, c) in zip(movie_subject_names, subject_xs, subject_ys, colors):
        ax.text(x, y, m, horizontalalignment='left', size='medium', color=c, weight='semibold')
    plb.show()
    continue
    

    for selection, indices in movie_selection.items():
        color = [ratings_by_id[i] for i in indices]
        scatterplot(projs[0], projs[1], color, indices, title)
