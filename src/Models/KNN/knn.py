"""K Nearest Neighbor Model

"""
import Sampler
import numpy as np
import numpy.random as npr
from Models.KNN.clustering import wGMM

from Sampler.drawer import TDrawer
from Sampler.flattener import Flattener


# TODO: get this moved over to
# 1. Multiprocessing
# 2. New Sampling strat


class WindowSelectorSmall:
    # TODO: consider adding 'Large' version
    # operates on the drawer directly
    """Select Non-Overlapping windows in order
        of similarity to a target window
        
        Small? Assumes all of the data can fit comfortably in memory
            = takes in indep_dat as vstacked array"""
    def __init__(self):
        pass

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
                NOTE: assumes win_starts = cumulative across animals
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


"""
# TODO: where should this be?
self.drawer = drawer
self.flattener = flattener

# TODO: unpack all data = the SMALL assumption:
# ... umm: this should probably not be in window selector
# .. should be passed in!
t_dat, id_dat, t0s = self.drawer.draw_all_samples()
self.indep_dat, self.dep_dat = flattener.flatten_samples(np.vstack(t_dat), np.vstack(id_dat))
"""


# TODO: model interface?
class KNN:

    def __init__(self,
                 num_means: int,
                 variances: np.ndarray,
                 num_iter: int = 5,
                 weight_thresh: float = 0.99,
                 train_sample_perc: float = 1.):
        """KNN initiation

        Args:
            num_means (int): number of means to
                use in GMM
            variances (np.ndarray): Scale specific dimensions to
                increase or decrease
                their impact (increase --> stronger influence on selection)
                Expected shape: num_parameter types x flattened independent 
                variable dim
            train_sample_perc (float): train sample percent
                specifies percentage of training set to use
        """
        assert(weight_thresh > 0.0 and weight_thresh < 1.), "weight thresh should be between 0 and 1"
        self.variances = variances
        self.num_iter = num_iter
        self.train_sample_perc = train_sample_perc
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

        Returns:
            np.ndarray: means generated by GMM fit
                in dependent variable space
                num_mean x N
            np.ndarray: covariances generated by GMM
                num_mean x N x N
            np.ndarray: mixing coeffs ~ scales on
                gaussians in GMM
                len num_mean array
            np.ndarray: prior probabilities on data
                samples ~ generated by knn process
                <len num_sample array
                ... len matched to screened data
            np.ndarray: selected data used for GMM
                NOTE: data samples will depend on
                variances as low probability samples
                get screened ~ in dependent space
                <num_sample x N
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
        means, covars, mixing_coeffs, _post = self.gmm.run(dat, priors, self.num_iter)
        return means, covars, mixing_coeffs, priors, dat

    def _test_1_sample(self,
                     tr_win_starts: np.ndarray,
                     twindow_size: float,
                     test_indep_sample: np.ndarray,
                     test_dep_sample: np.ndarray,
                     tr_full_indep: np.ndarray,
                     tr_full_dep: np.ndarray,
                     variance_i: np.ndarray):
            # TODO: doc string
            # fit: compare test sample to all train possibilities
            means, covars, mixing_coeffs, priors, _dat = self._fit_1sample(tr_win_starts,
                                                                          twindow_size,
                                                                          test_indep_sample,
                                                                          tr_full_indep,
                                                                          tr_full_dep,
                                                                          variance_i)
            # test: test log-likelihood on GMMs + test_dep_sample
            return self._calc_log_like_test(means, covars, mixing_coeffs, priors, test_dep_sample)

    def _calc_log_like_train(self, means: np.ndarray, covars: np.ndarray, 
                             mixing_coeffs: np.ndarray, priors: np.ndarray,
                             dep_dat: np.ndarray):
        """Calculate training log-likelihood
        TRAIN ASSUMPTION: with top match (assumed to be self) left out
        Also: everything should be in the dependent variable space
        Args:
            means (np.ndarray): num_mean x N
            covars (np.ndarray): num_mean x N x N
            mixing_coeffs (np.ndarray): GMM mixing coefficients
                = scales for the difference gaussian in the mixture
                = len N array
            priors (np.ndarray): normalized samples weights
                = len num_sample array
            dep_dat (np.ndarray): dependent variable values
                for all samples
                = num_sample x N array

        Returns:
            float: log-likelihood with top match left out
                ... assumes to be trivial mapping cuz training
        """
        p2 = priors[1:]
        p2 = p2 / np.sum(p2)
        loglike = self.gmm.log_like(means, covars, mixing_coeffs, p2, dep_dat[1:])
        return loglike

    def _calc_log_like_test(self, means, covars, mixing_coeffs, priors, dep_dat):
        """Calculate test set log-likelihood
        Also: everything should be in the dependent variable space
        Args:
            means (np.ndarray): num_mean x N
            covars (np.ndarray): num_mean x N x N
            mixing_coeffs (np.ndarray): GMM mixing coefficients
                = scales for the difference gaussian in the mixture
                = len N array
            priors (np.ndarray): normalized samples weights
                = len num_sample array
            dep_dat (np.ndarray): dependent variable values
                for all samples
                = num_sample x N array

        Returns:
            float: log-likelihood
        """
        loglike = self.gmm.log_like(means, covars, mixing_coeffs, priors, dep_dat)
        return loglike

    def _train_epoch_1var(self, train_idxs: np.ndarray,
                          indep_dat: np.ndarray, dep_dat: np.ndarray,
                          window_starts: np.ndarray, twindow_size: int,
                          variance_i: np.ndarray):
        """Single traininv epoch for single KNN variance

        Args:
            train_idxs (np.ndarray): array of ints
                which indices to use for training
                Can search against all windows tho
            indep_dat (np.ndarray): all samples ~ independent variables
                num_sample x N
            dep_dat (np.ndarray): all samples ~ dependent variables
                num_sample x M
            window_starts (np.ndarray): window starts order matched
                to (in)dep data
                len num_sample array
            twindow_size (int): size of each time window
            variance_i (np.ndarray): single knn variance
                len N array

        Returns:
            float: log-likelihood averaged across al test windows
        """
        lls = []
        for i in train_idxs:
            mu, sig, mixcoeff, priors, dat = self._fit_1sample(window_starts, twindow_size, indep_dat[i],
                                indep_dat, dep_dat, variance_i)
            tll = self._calc_log_like_train(mu, sig, mixcoeff, priors, dat)
            lls.append(tll)
            print(tll)
        return np.mean(np.array(lls))

    def _select_all_samples(self, train_sampler: SamplerIface):
        """Pulls all samples

        Args:
            train_sampler (SamplerIface):

        Returns:
            np.ndarray: independent data
            np.ndarray: dependent data
            np.ndarray: window starts
        """ 
        # pull all samples
        num_samples = train_sampler.get_sample_size()
        train_sampler.epoch_reset()
        dat_t, dat_id, window_starts = train_sampler.pull_samples(num_samples)
        # flatten the data
        train_indep_dat, train_dep_dat = train_sampler.flatten_samples(dat_t, dat_id)
        return train_indep_dat, train_dep_dat, window_starts

    def train_epoch(self, train_sampler: SamplerIface, rand_seed: int = 42):
        """Train for 1 epoch
        Select variance that gives best training reconstruction
        Saves data internally

        Args:
            train_sampler (SamplerIface): a sampler that implements
                the sampler interface
        
        Returns:
            bool: completion bit
            List[float]: log-likelihood for each variance set
        """
        # goal: figure out correct variance
        # time window?
        twindow_size = train_sampler.get_twindow_size()
        # pull all samples:
        train_indep_dat, train_dep_dat, window_starts = self._select_all_samples(train_sampler)
        # train_idx samples ~ can still search against the rest of the training samples
        train_idx = np.arange(len(window_starts))
        rng_obj = npr.default_rng(rand_seed)
        rng_obj.shuffle(train_idx)
        train_idx = train_idx[:int(len(window_starts) * self.train_sample_perc)]

        best_ind, best_ll = None, None
        all_lls = []
        for i, variance_i in enumerate(self.variances):
            ll = self._train_epoch_1var(train_idx, train_indep_dat, train_dep_dat, window_starts,
                                        twindow_size, variance_i)
            if best_ll is None or ll > best_ll:
                best_ind = i
                best_ll = ll
            all_lls.append(ll)
        # save as state
        self.train_indep_dat = train_indep_dat
        self.train_dep_dat = train_dep_dat
        self.train_variance = self.variances[best_ind]
        return True, all_lls
    
    def test_loglike(self, 
                     train_sampler: SamplerIface,
                     test_sampler: SamplerIface,
                     best_variance_idx: int):

        twindow_size = test_sampler.get_twindow_size()
        # pull all train samples:
        train_indep_dat, train_dep_dat, window_starts = self._select_all_samples(train_sampler)

        # iter thru all test samples:
        test_sampler.epoch_reset()
        test_t_dat, test_id_dat, _ = test_sampler.pull_samples(1)
        lls = []
        while(len(test_t_dat) > 0):
            test_indep, test_dep = test_sampler.flatten_samples(test_t_dat, test_id_dat)
            ll = self._test_1_sample(window_starts, twindow_size, test_indep[0], test_dep[0],
                                     train_indep_dat, train_dep_dat, self.variances[best_variance_idx])
            lls.append(ll)
            test_t_dat, test_id_dat, _ = test_sampler.pull_samples(1)
            print(lls[-1])
