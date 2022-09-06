"""Set sampling tests"""
import numpy as np
import numpy.random as npr
from Sampler.set_sampling import (DataGroup, _split_leaves, _split,
                                  gather_leaves, split_leaves)
from Sampler.source_sampling import DataSource


def test_split():
    rng = npr.default_rng(42)

    # split testing:
    T = 100
    v1, v2 = _split(rng, 0.5, [z for z in range(T)])
    assert(len(set(v1).intersection(set(v2))) == 0)
    assert(len(v1) == int(T / 2))


def _make_fake_tree():
    # fake source
    class FakeSource(DataSource):
        def __init__(self, idx):
            self.idx = idx
        def get_numpy_data(self):
            return {"A": np.arange(self.idx),
                    "B": np.arange(self.idx)}
    
    # fake tree:
    grp1 = DataGroup([FakeSource(10), FakeSource(20)], [])
    grp2 = DataGroup([FakeSource(15), FakeSource(25)], [])
    parent = DataGroup([], [grp1, grp2])
    return parent


def test_gather_leaves():
    parent = _make_fake_tree()
    lvs = gather_leaves(parent)
    assert(len(lvs) == 4)
    lv_dat_A = np.concatenate([lv.get_numpy_data()["A"] for lv in lvs], axis=0)
    assert(np.all(lv_dat_A == np.concatenate([np.arange(10),
                                              np.arange(20),
                                              np.arange(15),
                                              np.arange(25)])))


def test_split_leaves():
    rng = npr.default_rng(42)
    parent = _make_fake_tree()
    prime_root, comp_root = split_leaves(parent, rng, 0.5)
    lv_dat_A = np.concatenate([lv.get_numpy_data()["A"]
                               for lv in gather_leaves(prime_root)],
                              axis=0)
    assert(np.all(lv_dat_A == np.concatenate([np.arange(20),
                                              np.arange(25)])))
    lv_dat_A = np.concatenate([lv.get_numpy_data()["A"]
                               for lv in gather_leaves(comp_root)],
                              axis=0)
    assert(np.all(lv_dat_A == np.concatenate([np.arange(10),
                                              np.arange(15)])))


if __name__ == "__main__":
    test_split()
    test_gather_leaves()
    test_split_leaves()
    