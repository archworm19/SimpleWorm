"""K Nearest Neighbor Model

    All data expected to be N x M x Twin shape
        N = number of samples
        M = other dim
        Twin = time

    Considerations:
    > Distance Scales
    > > Scale specific dimensions to increase or decrease
    > > their impact (increase --> stronger influence on selection)
    > Selection Dims
    > > For which distance calculate
    > Prediction Dims
    > Decoder
    > > 1. Mean
    > > 2. 

    Training --> Just Store all data
    Testing --> where KNN logic comes in
"""
import numpy as np

# TODO: model interface?
class KNN:

    def __init__(self,
                 k: int,
                 selection_dims: np.ndarray,
                 prediction_dims: np.ndarray,
                 variances: np.ndarray = 1.):
        """KNN initiation

        Args:
            k (int): number of neighbors to use
            selection_dims (np.ndarray): which dims of dataset
                will be used for selecting neighbors.
            prediction_dims (np.ndarray): which dims of dataset
                will be used for making predictions.
            variances (np.ndarray): Scale specific dimensions to
                increase or decrease
                their impact (increase --> stronger influence on selection)
        """
        self.k = k
        self.selection_dims = selection_dims
        self.prediction_dims = prediction_dims
        self.variances = variances
        self.datastore = None


    # TODO: figure out interface
    # TODO: get imports right
    def train_epoch(self, sampler):
        dshape = sampler.get_data_shape()
        self.datastore = sampler.pull_samples(dshape[0])


    
    def _find_neighbors(self, test_sample: np.ndarray):
        """Find neighbors for 1 sample
        
        Arguments:
            test_sample (np.ndarray): M x Twin sample
        """
        dsub = self.datastore[:, self.selection_dims]
        di = dsub - test_sample[None]
        ivar = 1. / self.variances[None]
        di_reduced = np.sum((di ** 2.) * ivar, axis=(1,2))
        inds = np.argsort(di_reduced)[:self.k]
        return inds


