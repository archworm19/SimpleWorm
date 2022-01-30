"""Experiments are defined by
    1. the underlying data
    2. the sampling factories
"""
from typing import List
import numpy as np
import numpy.random as npr
from anml_factory import ANMLFactory
from even_split_factory import SSFactoryEvenSplit

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


def build_anml_factory(dat: List[np.ndarray],
                       twindow_size: int,
                       dep_t_mask = np.ndarray,
                       dep_id_mask = np.ndarray,
                       indep_t_mask = np.ndarray,
                       indep_id_mask = np.ndarray,
                       rand_seed: int = 42):
    """animal factory
    break up train-test by animals
    Assumes: each array in dat is for different animal
    Assumes: 2/3 train - test split (makes sense assuming 
        cross-validation in train)

    Args:
        dat (List[np.ndarray]): animal data
        twindow_size (int): sampling twindow size 
            used by sampler
        masks (np.ndarray): boolean arrays indicating
            which dimensions are independent / depdendent
            variables
        rand_seed (int, optional): Defaults to 42.
    """
    # ensure dims match across datasets
    _assert_dim_match(dat)

    # get max data length for length normalization:
    max_dlen = _get_max_dat_len(dat)

    # shuffle-based sampling
    gen = npr.default_rng(rand_seed)
    inds = np.arange(len(dat))
    gen.shuffle(inds)
    numtr = int((2./3.) * len(inds))
    train_anmls = np.full((len(dat),), False)
    train_anmls[inds[:numtr]] = True

    # build identity data:
    ident_dat = []
    for i, d in enumerate(dat):
        ids = i * np.ones((len(d)))
        tz = np.arange(len(d)) / max_dlen
        ident_dat.append(np.vstack((ids, tz)).T)

    train_factory = ANMLFactory(dat,
                                ident_dat,
                                train_anmls,
                                twindow_size,
                                dep_t_mask,
                                dep_id_mask,
                                indep_t_mask,
                                indep_id_mask)
    test_factory = ANMLFactory(dat,
                                ident_dat,
                                np.logical_not(train_anmls),
                                twindow_size,
                                dep_t_mask,
                                dep_id_mask,
                                indep_t_mask,
                                indep_id_mask,
                                1.)
    return train_factory, test_factory.generate_split(0)[0]


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

    # get max data length for length normalization:
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


# TESTING

def _color_idents(dat_ids: List[np.ndarray],
                  t_ids: List[np.ndarray]):
    """Testing function to ensure no sampling
        overlap
    Assumes: t_id entries are normalized
    """
    # compress all entries to single dim:
    newts, colours = [], []
    for i in range(len(dat_ids)):
        newts.append(dat_ids[i] + t_ids[i])
        colours.append(i * np.ones((len(newts[-1]))))
    newts = np.hstack(newts)
    sinds = np.argsort(newts)
    newts = newts[sinds]
    colours = np.hstack(colours)[sinds]
    # collision check:
    assert(len(np.unique(newts)) == len(newts)), "collision failure"
    colour_ar = np.vstack((newts, colours)).T
    print(colour_ar)
    print(np.shape(colour_ar))


def test_even_split():
    # test samplers with fake data
    d1 = 1 * np.ones((300, 3))
    d2 = 2 * np.ones((400, 3))
    train_factory, test_sampler = build_small_factory([d1, d2], 6, 42)
    train_sampler, cross_sampler = train_factory.generate_split(1)
    idz, tz = [], []
    for v in [train_sampler, cross_sampler, test_sampler]:
        (dtseries, ident) = v.pull_samples(1000)
        print(ident)
        idz.append(ident[:, 0])
        tz.append(ident[:, 1])
        # TESTING: flattening and unflattening
        flatz = v.flatten_samples(dtseries, ident)
        udtseries, uident = v.unflatten_samples(flatz)
        assert(np.sum(dtseries != udtseries) < 1), "unflatten check1"
        assert(np.sum(ident != uident) < 1), "unflatten check1"
    _color_idents(idz, tz)


def test_anml():
    # test samplers with fake data
    d1 = 1 * np.ones((300, 3))
    d2 = 2 * np.ones((400, 3))
    d3 = 3 * np.ones((500, 3))
    # test mask ~ dims 0 and 2 --> indep; 1 --> dep
    dep_t_mask = np.array([False, True, False])
    dep_t_mask = np.tile(dep_t_mask[None], (6, 1))
    # neither id covariate is depenetent
    dep_id_mask = np.array([False, False])
    indep_t_mask = np.logical_not(dep_t_mask)
    indep_id_mask = np.logical_not(dep_id_mask)
    train_factory, test_sampler = build_anml_factory([d1, d2, d3], 6,
                                                      dep_t_mask,
                                                      dep_id_mask,
                                                      indep_t_mask,
                                                      indep_id_mask)
    train_sampler, cross_sampler = train_factory.generate_split(1)
    for v in [train_sampler, cross_sampler, test_sampler]:
        print(v)
        (dtseries, ident) = v.pull_samples(1000)
        print(dtseries[:3])
        print(ident[:3])
        indep_flat, dep_flat = v.flatten_samples(dtseries, ident)
        print(indep_flat[:3])
        input('cont?')


if __name__ == "__main__":
    #test_even_split()
    test_anml()
