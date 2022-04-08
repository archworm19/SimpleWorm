"""Experiments are defined by
    1. the underlying data
    2. the sampling factories
"""
from typing import List
import numpy as np
import numpy.random as npr
import Sampler.utils as utils
from Sampler.anml_factory import ANMLFactoryMultiSet
from Sampler.even_split_factory import SSFactoryEvenSplit

# utilities

def _get_max_dat_len(dat: List[np.ndarray]):
    """Helper function to find maximum data length
    across arrays
    Returns: int
    """
    ml = -1
    for d in dat:
        if len(d) > ml:
            ml = len(d)
    return ml


def _assert_dim_match(dat: List[np.ndarray]):
    """Assert that all dims match after dim0
    for arrays within dat List
    """
    if len(dat) > 1:
        v0 = np.array(np.shape(dat[0])[1:])
        for d in dat[1:]:
            vi = np.array(np.shape(d)[1:])
            assert(np.sum(v0 != vi) < 1), "dim match assertion failure"


def build_anml_factory_multiset(t_dat: List[List[np.ndarray]],
                                id_dat: List[List[np.ndarray]],
                                twindow_size: int,
                                dep_t_mask = np.ndarray,
                                dep_id_mask = np.ndarray,
                                indep_t_mask = np.ndarray,
                                indep_id_mask = np.ndarray,
                                rand_seed: int = 42):
    """multi set animal factory
    each list in t/id_dat is a different set
    each array in these lists = different animal

    break up train-test by animals
    > breaks up WITHIN each set
    Assumes: 2/3 train - test split (makes sense assuming 
        cross-validation in train)

    Args:
        t_dat (List[np.ndarray]): time series animal data
            each array element assumed to be
            from different animal
            list elements must match in all
            dims except 0th
            num_sample_i x N1 x N2 x ...
        id_dat (List[List[np.ndarray]]): identity data
            each array assumed to be from different animal
            each array assumed to have form:
            num_sample_i x M1
        twindow_size (int): time window size 
            used by sampler
        dep_t_mask (np.ndarray): boolean mask on timeseries
            data indicating dependent variables
            twindow_size x N1 x N2 x ...
        dep_id_mask (np.ndarray): boolean mask on identity data
            indicating dependent variables
            M1
        masks (np.ndarray): boolean arrays indicating
            which dimensions are independent / depdendent
            variables
        rand_seed (int, optional): Defaults to 42.
    """
    # ensure dims match across datasets
    flat_t, flat_id = [], []
    for i in range(len(t_dat)):
        assert(len(t_dat[i]) >= 3), "must have >=3 anmls in each set"
        for j in range(len(t_dat[i])):
            assert(len(t_dat[i][j]) == len(id_dat[i][j])), "time mismatch"
        flat_t.extend(t_dat[i])
        flat_id.extend(id_dat[i])
    _assert_dim_match(flat_t)
    _assert_dim_match(flat_id)

    # iter thru each set:
    gen = npr.default_rng(rand_seed)
    train_anml, test_anml = [], []
    for i in range(len(t_dat)):
        train_bools = utils.shuffle_sample(len(t_dat[i]), gen)
        train_anml.append(train_bools)
        test_anml.append(np.logical_not(train_bools))

    train_factory = ANMLFactoryMultiSet(t_dat,
                                id_dat,
                                train_anml,
                                twindow_size,
                                dep_t_mask,
                                dep_id_mask,
                                indep_t_mask,
                                indep_id_mask)
    test_factory = ANMLFactoryMultiSet(t_dat,
                               id_dat,
                               test_anml,
                               twindow_size,
                               dep_t_mask,
                               dep_id_mask,
                               indep_t_mask,
                               indep_id_mask,
                               1.)
    return train_factory, test_factory.generate_split()[0]


def build_small_factory(dat: List[np.ndarray],
                        twindow_size: int,
                        rand_seed: int = 42):
    """Build a Small Factory

    dat arrays must be T x ...
    
    Creates 2 identity variables
    > relative window start (divide by max data length)
    > array (animal) identity ~ which array (within List[np.ndarray])
    sample came from

    Size assumption: small ~ no need for memory mapping
    
    > Split into train (2/3) and test (1/3) sets
    > build a factory for train
    > build a sampler for test

    Arguments:
        dat (List[np.ndarray]): list of T x ... arrays
            assumed to be timeseries with time along axis 0
        twindow_size (int): sampling window size
            = Time window size models will act on
        rand_seed (int): rng seed for train vs. test splitting

    Returns:
        SSFactoryEvenSplit: sampler generator for Train sets
        SmallSampler: single sampler for whole of test set
        
    """
    _assert_dim_match(dat)

    # get max time data length for length normalization:
    max_dlen = _get_max_dat_len(dat)

    # seeding random number generator makes 
    # process deterministic
    rng_gen = npr.default_rng(rand_seed)

    train_bools, test_bools = [], []
    ident_dat = []
    # iter thru files ~ top-level
    for i in range(len(dat)):
        # pull rt0 here ~ random offsets
        rt0 = rng_gen.integers(twindow_size)
        boolz = np.ones((len(dat[i]))) < 0  # false by default
        for j in range(rt0, len(dat[i]), twindow_size):
            if rng_gen.random() < .667:
                boolz[j:j+twindow_size] = True
        train_bools.append(boolz)
        test_bools.append(np.logical_not(boolz))
        # identity data:
        wid = i * np.ones((len(boolz)))
        trang = (rt0 + np.arange(len(boolz))) / max_dlen
        ident_dat.append(np.vstack((wid, trang)).T)
    TrainFactory = SSFactoryEvenSplit(dat, ident_dat, train_bools, twindow_size, 0.5)
    TestFactory = SSFactoryEvenSplit(dat, ident_dat, test_bools, twindow_size, 1.0)
    return TrainFactory, TestFactory.generate_split(0)[0]
