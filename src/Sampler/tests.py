"""Test functions for experiments"""
from typing import List
import copy
import numpy as np
import numpy.random as npr
import pylab as plt
import Sampler.utils as utils
from experiments import (build_small_factory,
                         build_anml_factory,
                         build_anml_factory_multiset)


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
        print('sampler object')
        print(v)
        dtseries, ident, _ = v.pull_samples(1000)
        print('sampled time series')
        print(dtseries[:3])
        print('sampled identities')
        print(ident[:3])
        indep_flat, dep_flat = v.flatten_samples(dtseries, ident)
        print('independent flat samples')
        print(indep_flat[:3])
        print('dependent flat samples')
        print(dep_flat[:3])
        indep_t, indep_id = v.unflatten_samples(indep_flat, indep=True)
        dep_t, dep_id = v.unflatten_samples(indep_flat, indep=False)
        # inversion check:
        print('independent time series inverse')
        print(indep_t[:3])
        print('independent time series inverse')
        print(indep_id[:3])
        print('dependent time series inverse')
        print(dep_t[:3])
        print('dependent time series inverse')
        print(dep_id[:3])
        input('cont?')
    

def test_shuffle():
    gen = npr.default_rng(33)
    tr_anmls = utils.shuffle_sample(9, gen)
    print(tr_anmls)
    gen = npr.default_rng(66)
    tr_anmls = utils.shuffle_sample(9, gen)
    print(tr_anmls)
    gen = npr.default_rng(11)
    tr_anmls = utils.shuffle_sample(9, gen)
    print(tr_anmls)


def test_anml_split():
    gen = npr.default_rng(11)
    useable_dat = utils.shuffle_sample(9, gen)
    print(useable_dat)
    tr_inds, te_inds = utils.generate_anml_split(useable_dat, gen)
    print(tr_inds)
    print(te_inds)
    tr_inds, te_inds = utils.generate_anml_split(useable_dat, gen)
    print(tr_inds)
    print(te_inds)


def test_anml_set():
    # test samplers with fake data
    d1_1 = 1 * np.ones((300, 3))
    d1_2 = 2 * np.ones((400, 3))
    d1_3 = 3 * np.ones((500, 3))
    d2_3 = 1 * np.ones((300, 3))
    d2_2 = 2 * np.ones((400, 3))
    d2_1 = 3 * np.ones((500, 3))
    t_dat = [[d1_1, d1_2, d1_3], [d2_1, d2_2, d2_3]]
    t_indic = np.arange(500) / 500
    id_dat = []
    count = 0
    for i in range(len(t_dat)):
        ids_sub = []
        for j in range(len(t_dat[i])):
            dlen = np.shape(t_dat[i][j])[0]
            v1 = count * np.ones((dlen,1))
            v2 = copy.deepcopy(t_indic[:dlen])
            ids_sub.append(np.hstack((v1,v2[:,None])))
            count += 1
        id_dat.append(ids_sub)

    # test mask ~ dims 0 and 2 --> indep; 1 --> dep
    dep_t_mask = np.array([False, True, False])
    dep_t_mask = np.tile(dep_t_mask[None], (6, 1))
    # neither id covariate is depenetent
    dep_id_mask = np.array([False, False])
    indep_t_mask = np.logical_not(dep_t_mask)
    indep_id_mask = np.logical_not(dep_id_mask)
    train_factory, test_sampler = build_anml_factory_multiset(t_dat, id_dat, 6,
                                                              dep_t_mask, dep_id_mask, 
                                                              indep_t_mask, indep_id_mask)
    train_sampler, cross_sampler = train_factory.generate_split()
    plt.figure()
    for i, v in enumerate([train_sampler, cross_sampler, test_sampler]):
        print('sampler object')
        print(v)
        dtseries, ident, _ = v.pull_samples(100000)
        plt.scatter(ident[:,0]*500 + ident[:,1]*500, ident[:,0]*0 + i)
        print('sampled time series')
        print(dtseries[:3])
        print(np.unique(dtseries))
        print('sampled identities')
        print(ident[:3])
        print(np.unique(ident[:,0]))
        indep_flat, dep_flat = v.flatten_samples(dtseries, ident)
        print('independent flat samples')
        print(indep_flat[:3])
        print('dependent flat samples')
        print(dep_flat[:3])
        indep_t, indep_id = v.unflatten_samples(indep_flat, indep=True)
        dep_t, dep_id = v.unflatten_samples(indep_flat, indep=False)
        # inversion check:
        print('independent time series inverse')
        print(indep_t[:3])
        nn_inds = np.logical_not(np.isnan(indep_t))
        print(np.nansum(indep_t[nn_inds] != dtseries[nn_inds]))
        print('independent id inverse')
        print(indep_id[:3])
        nn_inds = np.logical_not(np.isnan(indep_id))
        print(np.nansum(indep_id[nn_inds] != ident[nn_inds]))
        print('dependent time series inverse')
        print(dep_t[:3])
        print('dependent id inverse')
        print(dep_id[:3])
        input('cont?')
    plt.show()


if __name__ == "__main__":
    #test_even_split()
    #test_anml()
    test_anml_set()
    #test_shuffle()
    #test_anml_split()
