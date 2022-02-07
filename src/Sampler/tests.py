"""Test functions for experiments"""
from typing import List
import numpy as np
from experiments import (build_small_factory, build_anml_factory)

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


if __name__ == "__main__":
    #test_even_split()
    test_anml()