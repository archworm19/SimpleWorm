"""K Nearest Neighbor Model

"""
import numpy as np
from clustering import wGMM

# TODO: model interface?
class KNN:

    def __init__(self,
                 num_means: int,
                 variances: np.ndarray,
                 num_iter = 5,
                 weight_thresh = 0.99):
        """KNN initiation

        Args:
            num_means (int): number of means to
                use in GMM
            variances (np.ndarray): Scale specific dimensions to
                increase or decrease
                their impact (increase --> stronger influence on selection)
                Expected shape: num_parameter types x flattened independent 
                variable dim
        """
        assert(weight_thresh > 0.0 and weight_thresh < 1.), "weight thresh should be between 0 and 1"
        self.variances = variances
        self.datastore = None  # TODO: delete?
        self.num_iter = num_iter
        self.weight_thresh = weight_thresh
        self.num_means = num_means
        self.gmm = wGMM(num_means)

    def _calc_weights(self,
                      indep_sample: np.ndarray,
                      full_indep_set: np.ndarray,
                      variance_i: np.ndarray):
        """Calculate the weights between 1 sample
        of indepenent dims and all of training independent samples.
        IMPORTANT NOTE: return payload is NOT normalized

        Args:
            indep_sample (np.ndarray): len N sample of independent dims
            full_indep_set (np.ndarray): full set of independent samples
                num_sample x N
            variance_i (np.ndarray): len N array representing single
                variance entry

        Returns:
            np.ndarray: len [num training samples] array of weights
                = exp(-1 * sum((indep_sample - train_sample)^2 / variance_i))
        """
        raw_diff = (full_indep_set - indep_sample[None])**2.
        quot = -1. * np.sum(raw_diff * (1. / variance_i[None]), axis=1)
        return np.exp(quot)
    
    def _select_nonolap_windows(self,
                                weights: np.ndarray,
                                win_starts: np.ndarray,
                                twindow_size: int):
        """Select non-overlapping windows according to
        weights (maximize) using Greedy strategy

        Arguments:
            weights (np.ndarray): len N array of weighted distances
            win_starts (np.ndarray): len N array of window starts
            twindow_size (int): size of each timewindow

        Returns:
            np.ndarray: indices (into wdists and win_starts)
                of greedily chosen samples
                in descending weight order
        """
        # initialize the colouring structure ~ use to guarantee no overlap
        minwin = int(np.amin(win_starts))
        maxwin = int(np.amax(win_starts))
        colours = np.zeros((maxwin + twindow_size - minwin))
        # iter thru wdists in reverse sorted order (large best)
        sinds = np.argsort(weights)[::-1]
        ret_inds = []
        for ind in sinds:
            # check if overlaps with an already-used window
            st = int(win_starts[ind] - minwin)
            ed = int(st + twindow_size)
            if np.sum(colours[st:ed] > 0.5):
                continue
            ret_inds.append(ind)
            colours[st:ed] = 1
        return np.array(ret_inds)
    
    def _fit_1sample(self,
                     win_starts: np.ndarray,
                     twindow_size: float,
                     indep_sample: np.ndarray,
                     full_indep: np.ndarray,
                     full_dep: np.ndarray,
                     variance_i: np.ndarray):
        """Fit 1 sample for 1 variance set

        Args:
            win_starts (np.ndarray): len num_sample array of window starts
            twindow_size (int): size of each timewindow
            indep_sample (np.ndarray): single sample
                of independent variables
                len N array
            full_indep (np.ndarray): set of all
                samples of independent variables
                num_sample x N
            full_dep (np.ndarray): set of all
                samples of dependent variables order
                matched to full_dep
                num_sample x M
            variances (np.ndarray): single variance
                set
                len N array

        TODO: return types

        """
        # weighted distance
        # --> len num_sample
        weights = self._calc_weights(indep_sample,
                                          full_indep,
                                          variance_i)

        # indices into weights
        select_inds = self._select_nonolap_windows(weights,
                                                   win_starts,
                                                   twindow_size)
        
        # safety check
        # should only use datapoints with high priors
        # low prior weights will practically make no difference
        # and potentially cause instability
        ord_weights = weights[select_inds]
        cu_weights = np.cumsum(ord_weights) / np.sum(ord_weights)
        # take either num_mean * 2 pts or where self.weight_thresh
        # is crossed; max of these 2
        bind = np.where(cu_weights >= self.weight_thresh)[0][0]
        wind = min(max(bind, int(2 * self.num_means)), len(ord_weights)-1)
        priors = ord_weights[:wind]
        priors = priors / np.sum(priors)

        # gmm run on dependent dims
        dat = full_dep[select_inds[:wind]]
        means, covars, mixing_coeffs, _ = self.gmm.run(dat, priors, self.num_iter)
        return means, covars, mixing_coeffs, priors, dat

    # TODO: docstrings
    # also: are these ok?
    # passing around data makes more efficient BUT harder to use

    def _calc_log_like_train(self, means, covars, mixing_coeffs, priors, dep_dat):
        # TRAIN ASSUMPTION: with top match (assumed to be self) left out
        loglike = self.gmm.log_like(means, covars, mixing_coeffs, priors[1:], dep_dat[1:])
        return means, covars, mixing_coeffs, loglike

    def _calc_log_like_test(self, means, covars, mixing_coeffs, priors, dep_dat):
        loglike = self.gmm.log_like(means, covars, mixing_coeffs, priors, dep_dat)
        return means, covars, mixing_coeffs, loglike

    def _train_epoch_1var(self, indep_dat, dep_dat, window_starts,
                            twindow_size, variance_i):
        # TODO: complete training epoch for 1 variance
        # TODO: averages log-like across dependent samples
        # ... NOTE: only need training data
        lls = []
        for i in range(len(indep_dat)):
            mu, sig, mixcoeff, priors, dat = self._fit_1sample(window_starts, twindow_size, indep_dat[i],
                                indep_dat, dep_dat, variance_i)
            tll = self._calc_log_like_train(mu, sig, mixcoeff, priors, dat)
            lls.append(tll)
        return np.mean(np.array(lls))

    # TODO: figure out interface ~ mark as completed???
    # TODO: get imports right
    def train_epoch(self, train_sampler):
        # TODO: goal: figure out correct variance
        # time window?
        twindow_size = train_sampler.get_twindow_size()
        # pull all samples
        num_samples = train_sampler.get_sample_size()
        dat_t, dat_id, window_starts = train_sampler.pull_samples(num_samples)
        # flatten the data
        train_indep_dat, train_dep_dat = train_sampler.flatten_samples(dat_t, dat_id)
        best_ind, best_ll = None, None
        for i, variance_i in enumerate(self.variances):
            ll = self._train_epoch_1var(train_indep_dat, train_dep_dat, window_starts,
                                        twindow_size, variance_i)
            if ll > best_ll:
                best_ind = i
                best_ll = ll
        # save as state
        self.train_indep_dat = train_indep_dat
        self.train_dep_dat = train_dep_dat
        self.train_variance = self.variances[best_ind]


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
    test_knn()
    

