"""Test code for knn + clustering"""
import itertools
import numpy as np
import numpy.random as npr
import numpy.linalg as nplg
import pylab as plt
from clustering import (wKMeans, wGMM)
from knn import KNN

def test_kmeans():
    km = wKMeans(2)
    dat = np.array([[1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [100, 100, 100, 100],
                    [101, 100, 101, 99]])
    priors = np.ones((4,)) / 4.
    means, dist_mat = km.run(dat, priors)
    print(means)
    print(dist_mat)
    means, _ = km.multi_run(dat, priors)
    print(means)

    # more kmeans testing
    v1 = npr.rand(10, 2)
    v2 = npr.rand(15, 2) + .9 * np.ones((15, 2))
    priors = np.ones((25,)) / 25.
    dat = np.vstack((v1, v2))
    means, _ = km.multi_run(dat, priors)
    print(means)
    print(km.assign_iter(means, dat))
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1])
    plt.scatter(means[:,0], means[:,1], c='r')
    plt.show()

    return dat, means

def test_gmm_probs():
    # forward probabilities of gmm
    G = wGMM(1)
    dim = 3
    covar = np.eye(dim)
    covar[1,1] = 2.
    prec, det = G._decompose_all_covars(covar[None])
    print('GMM probs')
    print(prec)
    print(det)
    dt = .2
    xi = np.arange(-10., 10., dt)
    gen = itertools.product(xi, xi, xi)
    x = np.array(list(gen))
    print(np.shape(x))    
    _, _, fprobs = G.probs(x[None], prec, det, np.array([1.]))
    print(np.amax(fprobs))
    print(np.sum(fprobs))
    # Should be approximately equal to 1
    print(np.sum(fprobs) * dt**dim)
    input('cont?')


def test_gmm():
    # NOTE: need scipy to run test
    # more specific testing:
    print('GMM testing')
    G = wGMM(2)

    # mmult ~ guarantees right order calc
    print('mmult test')
    G._mmult(npr.rand(20, 5),
             npr.rand(5, 8),
             npr.rand(20, 8))

    from scipy.stats import multivariate_normal
    for _ in range(3):
        mu = npr.rand(4)
        cov_raw = npr.rand(4, 4)
        cov = 5 * cov_raw @ cov_raw.T
        print(cov)
        print(nplg.eig(cov))
        dat = npr.rand(4)
        var = multivariate_normal(mean=mu, cov=cov)
        # compare against ours:
        precision, cov_det = G._decompose_covar(np.array(cov))
        print('precision check: should be identity')
        print(np.array(cov) @ precision)
        print('determinant check')
        print(nplg.det(np.array(cov)))
        print(cov_det)
        raw_diff = np.array(dat)[None,None] - np.array(mu)[None, None]
        pr = G.probs(raw_diff, np.array(precision)[None], np.array(cov_det)[None],
                     np.array([[1.]]))
        print('iter; probs')
        print(var.pdf(dat))
        print(pr)
        input('cont?')
    
    # fake data test
    # uniform weights
    v1 = npr.rand(10, 2)
    v2 = npr.rand(15, 2) + .9 * np.ones((15, 2))
    priors = np.ones((25,)) / 25.
    dat = np.vstack((v1, v2))
    means, covars, mix_coeffs, posts = G.run(dat, priors)
    print('means')
    print(means)
    print('covars')
    print(covars)
    print('mixing coeffs')
    print(mix_coeffs)
    print('posteriors')
    print(posts)
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1])
    plt.scatter(means[:,0], means[:,1])
    plt.show()

    # fake data test
    # > 4 clusters
    # > 2 clusters dominate weightes
    # upper-right:
    v1 = npr.rand(10, 2) + np.ones((10,2))
    # lower-right:
    v2 = npr.rand(20, 2)
    v2[:,0] = v2[:,0] - 1.
    v2[:,1] = v2[:,1] + 1.
    # lower-left:
    v3 = npr.rand(10, 2) - np.ones((10,2))
    # upper-leaf:
    v4 = npr.rand(20, 2)
    v4[:,0] = v4[:,0] + 1.
    v4[:,1] = v4[:,1] - 1.
    # priors
    wdiag = 1. * np.ones((10,))
    woff = 0.01 * np.ones((20,))
    priors = np.hstack((wdiag, woff, wdiag, woff))
    priors = priors / np.sum(priors)
    # dat construction
    dat = np.vstack((v1, v2, v3, v4))
    means, covars, mix_coeffs, posts = G.run(dat, priors)
    print('weight testing posts')
    print(posts)
    print('weight testing covars')
    print(covars)
    print('log likelihood: {0}'.format(G.log_like(means,
                                            covars,
                                            mix_coeffs,
                                            priors,
                                            dat)))
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1])
    plt.scatter(means[:,0], means[:,1])
    plt.show()

def test_knn():
    import numpy.random as npr
    # variances are in indep space:
    variances = np.array([[1., 1.], [0.1, 0.1]])

    v1i = npr.rand(10,2) + np.ones((10,2))
    v2i = npr.rand(20,2) - np.ones((20,2))
    full_indeps = np.vstack((v1i, v2i))

    knn = KNN(2, variances)

    print('weights')
    weights1 = knn._calc_weights(full_indeps[0], full_indeps, variances[0])
    weights2 = knn._calc_weights(full_indeps[0], full_indeps, variances[1])
    print(weights1)
    print(weights2)

    # window selection test:
    w1ins = np.array([6*i for i in range(10)])
    w2ins = np.array([200 + 6*i for i in range(20)])
    win_starts = np.hstack((w1ins,w2ins))
    # testing with complete non-overlap
    sel1 = knn._select_nonolap_windows(weights1, win_starts, 6)
    sel2 = knn._select_nonolap_windows(weights2, win_starts, 6)
    # testing with overlap
    sel2olap = knn._select_nonolap_windows(weights2, win_starts, 12)
    print('olap select')
    print(sel1)
    print(sel2)
    print(sel2olap)
    print('corresponding weights')
    print(weights1[sel1])
    print(weights2[sel2olap])

    # TODO: fit 1sample
    v1d = npr.rand(10,2) + 2*np.ones((10,2))
    v2d = npr.rand(20,2) - 2*np.ones((20,2))
    full_deps = np.vstack((v1d, v2d))
    # test with no overlap:
    means, covars, mixing_coeffs, _, _ = knn._fit_1sample(win_starts, 6, full_indeps[0],
                                                          full_indeps, full_deps, variances[0])
    print('fit1 no olap')
    print(means)
    print(mixing_coeffs)
    # test with overlap:
    means, covars, mixing_coeffs, _, _ = knn._fit_1sample(win_starts, 12, full_indeps[0],
                                                          full_indeps, full_deps, variances[0])
    print('fit2 olap')
    print(means)
    print(mixing_coeffs)


if __name__ == '__main__':
    import pylab as plt

    # probs testing:
    test_gmm_probs()

    # kmeans testing
    test_kmeans()

    # wGMM testing
    test_gmm()

    # knn testing
    test_knn()