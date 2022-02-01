"""K Nearest Neighbor Model

"""
import numpy as np

# TODO: model interface?
class KNN:

    def __init__(self,
                 k: int,
                 variances: np.ndarray):
        """KNN initiation

        Args:
            k (int): number of neighbors to use
            variances (np.ndarray): Scale specific dimensions to
                increase or decrease
                their impact (increase --> stronger influence on selection)
                Expected shape: num_parameter types x flattened independent 
                variable dim
        """
        self.k = k
        self.variances = variances
        self.datastore = None


    # TODO: figure out interface
    # TODO: get imports right
    def train_epoch(self, sampler):
        # pull all samples
        num_samples = sampler.get_sample_size()
        dat_t, dat_id = sampler.pull_samples(num_samples)
        # save flattened indep and dep variables:
        self.train_indep_dat, self.train_dep_dat = sampler.flatten_samples(dat_t, dat_id)


    # TODO alg
    # > Matrix distance calc
    # > > N train samples x M test samples
    # > Greedy neighbor finding
    # > > Find closest entirely free window
    # > > Colour window as used
    # > > keep track of its weighted distance
    # TODO: probably more space efficient
    # to do one test sample at a time

    def _calc_weighted_dists(self,
                             indep_sample: np.ndarray,
                             variance_i: np.ndarray):
        """Calculate the weighted distances between 1 sample
        of indepenent dims and all of training independent samples.
        IMPORTANT NOTE: return payload is NOT normalized

        Args:
            indep_sample (np.ndarray): len N sample of independent dims
            variance_i (np.ndarray): len N array representing single
                variance entry

        Returns:
            np.ndarray: len [num training samples] array of weighted
                distance = exp(-1 * sum((indep_sample - train_sample)^2 / variance_i))
        """
        raw_diff = (self.train_indep_dat - indep_sample[None])**2.
        quot = -1. * np.sum(raw_diff * (1. / variance_i[None]), axis=1)
        return np.exp(quot)
    
    def _select_nonolap_windows(self,
                                wdists: np.ndarray,
                                win_starts: np.ndarray,
                                twindow_size: int):
        """Select non-overlapping windows according to
        weighted distance (wdist) using Greedy strategy

        Arguments:
            wdists (np.ndarray): len N array of weighted distances
            win_starts (np.ndarray): len N array of window starts
            twindow_size (int): size of each timewindow

        Returns:
            np.ndarray: indices (into wdists and win_starts)
                of greedily chosen samples
        """
        # initialize the colouring structure ~ use to guarantee no overlap
        minwin = int(np.amin(win_starts))
        maxwin = int(np.amax(win_starts))
        colours = np.zeros(minwin, maxwin + twindow_size)
        # iter thru wdists in sorted order (min dist best)
        sinds = np.argsort(wdists)
        ret_inds = []
        for ind in sinds:
            # check if overlaps with an already-used window
            st = win_starts[ind]
            ed = st + twindow_size
            if np.sum(colours[st:ed] > 0.5):
                continue
            ret_inds.append(ind)
            colours[st:ed] = 1
        return np.array(ret_inds)

    # TODO: this should perhaps be a different class
    # yeah: probs
    def wGMM(self, dep_dat: np.ndarray, priors: np.ndarray):
        """weighted Gaussian Mixture Model

        Args:
            dep_dat (np.ndarray): N x m array of dependent data
            priors (np.ndarray): len N array of priors or weighted
                for the corresponding dep_dat entries
        """
        # > initialize with Kmeans
        # > Repeated cycles (EM)
        # > > Prob that each sample belongs to given cluster
        # > > Reweight these probs using priors ~ can just mult (cuz will renorm after)
        # > > Normalize probs within cluster (Bayes law application --> P(clust | samples))
        # > > Calculate weighted means and covars for each cluster


    # TODO: training for 1 sample
    # > calc weighted dist
    # > select nonoverlapping windows
    # > GMM
    

