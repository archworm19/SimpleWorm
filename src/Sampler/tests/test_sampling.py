"""Test Sampling
    > file_reps
    > set_sampling
    > drawer"""
import numpy as np
import numpy.random as npr
from Sampler import file_sets, file_reps_np, drawer
#from Sampler import set_sampling


def _make_fake_tree():
    # 2 top-level sets
    # each top-level set split into 2 subsets
    # each subset has 3 files
    # --> 4 subsets * 3 files each = 12 files
    # make the files
    t_ar = np.ones((5,5))
    id_ar = np.ones((5,2))
    t0_samples = np.arange(5)
    sub_sets = []
    for i in range(4):
        ss = file_sets.FileSet([], [])
        for j in range(3):
            # relies on idn for identification
            f = file_reps_np.save_file(i*3 + j, "TEMP", t_ar, id_ar, t0_samples)
            ss.files.append(f)
        sub_sets.append(ss)
    top_set0 = file_sets.FileSet(sub_sets[:2], [])
    top_set1 = file_sets.FileSet(sub_sets[2:], [])
    root = file_sets.FileSet([top_set0, top_set1], [])
    return root


def test_depth():
    root = _make_fake_tree()
    v = file_sets.get_depths(root)
    assert(len(v) == 4)
    for vi in v:
        assert(vi == 2)


def test_idx_mapping():
    root = _make_fake_tree()
    count = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                set_idx = [i,j]
                file_idx = k
                v = file_sets.map_idx_to_file(root, set_idx, file_idx)
                assert(v.idn == count)
                count += 1


def test_sample_exe():
    # fake tree --> 2 top-level > 2 each > 3 files
    root = _make_fake_tree()

    rng = npr.default_rng(42)

    # split on second level
    newset0 = file_sets.FileSet([], None)
    set_strat, _ = set_sampling.build_comp_sets(root, 1, 0.5, rng)
    file_strat, _ = set_sampling.build_comp_files(root, 1., rng)
    t0_strat = set_sampling.Allt0Strat()
    set_sampling.exe_plan([], newset0, root, set_strat, file_strat, t0_strat)
    filez = file_sets.get_files(newset0)
    exp_idns = {3, 4, 5, 6, 7, 8}
    f_idns = set([fi.idn for fi in filez])
    assert(exp_idns.issubset(f_idns) and exp_idns.issuperset(f_idns))
    
    # split on top level
    newset0 = file_sets.FileSet([], None)
    set_strat, _ = set_sampling.build_comp_sets(root, 0, 0.5, rng)
    file_strat, _ = set_sampling.build_comp_files(root, 1., rng)
    t0_strat = set_sampling.Allt0Strat()
    set_sampling.exe_plan([], newset0, root, set_strat, file_strat, t0_strat)
    filez = file_sets.get_files(newset0)
    exp_idns = {0, 1, 2, 3, 4, 5}
    f_idns = set([fi.idn for fi in filez])
    assert(exp_idns.issubset(f_idns) and exp_idns.issuperset(f_idns))


def _make_fake_files(t0_sub: bool = False):
    # make a single set with 3 animals
    tar0 = np.zeros((20, 8, 3))
    tar1 = np.ones((10, 8, 3))
    tar2 = np.ones((15, 8, 3)) * 2
    id0 = np.vstack((np.arange(20), np.zeros(20,))).T
    id1 = np.vstack((np.arange(10), np.ones((10,)))).T
    id2 = np.vstack((np.arange(15), 2*np.ones((15,)))).T
    if t0_sub:
        t00 = np.arange(3)
        t01 = np.arange(3)
        t02 = np.arange(3)
    else:
        t00 = np.arange(20)
        t01 = np.arange(10)
        t02 = np.arange(15)
    tz = [tar0, tar1, tar2]
    idz = [id0, id1, id2]
    t0z = [t00, t01, t02]
    filez = []
    file_root = 'TEMP'
    for i in range(len(tz)):
        filez.append(file_reps_np.save_file(i, file_root + str(i), 
                                         tz[i], idz[i], t0z[i]))
    # package into single set:
    return file_sets.FileSet([], filez)


def test_animal_sampler():
    root = _make_fake_files()
    rng = npr.default_rng(42)
    set1, set2 = set_sampling.get_anml_sample_switch(root, .667, 4, 2, 1, rng)
    f1 = file_sets.get_files(set1)
    f2 = file_sets.get_files(set2)
    target_t0s = [np.array([1, 2, 3, 9, 10, 11]),
                  np.array([1, 2, 3, 9]), 
                  np.array([5, 6, 7, 13, 14, 15])]
    for f, targ_f in zip(f1 + f2, target_t0s):
        _, _, t0s = f.open_file()
        assert(np.sum(t0s != targ_f) < 1)
    

def test_drawer():
    root = _make_fake_files()
    twindow_size = 3
    D = drawer.TDrawer(root, twindow_size)
    avail_samples = D.get_available_samples()
    # 45 timepoints - 6 illegals starts (3 twindow_size -1 * 3)
    assert(avail_samples == 39)
    # idxs for file 0:
    numA0 = 20+1-twindow_size 
    tsamp, idsamp = D.draw_samples(np.arange(numA0))
    assert(np.shape(tsamp) == (18, 3, 8, 3))
    assert(np.sum(tsamp != 0) < 1)
    assert(np.shape(idsamp) == (18, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 0)
    # idxs for file 1:
    numA1 = numA0 + (10+1-twindow_size)
    tsamp, idsamp = D.draw_samples(np.arange(numA0, numA1))
    assert(np.shape(tsamp) == (8, 3, 8, 3))
    assert(np.sum(tsamp != 1) < 1)
    assert(np.shape(idsamp) == (8, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 1)
    # idxs for file 2:
    numA2 = numA1 + (15+1-twindow_size)
    tsamp, idsamp = D.draw_samples(np.arange(numA1, numA2))
    assert(np.shape(tsamp) == (13, 3, 8, 3))
    assert(np.sum(tsamp != 2) < 1)
    assert(np.shape(idsamp) == (13, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 2)


def test_drawer_t0():
    # t0_sub specifies: use subset of available t0s
    root = _make_fake_files(t0_sub=True)
    twindow_size = 3
    D = drawer.TDrawer(root, twindow_size)
    avail_samples = D.get_available_samples()
    assert(avail_samples == 9)
    tsamp, idsamp = D.draw_samples(np.arange(9))
    exp_idsamp = np.array([[0., 0.],
                            [1., 0.],
                            [2., 0.],
                            [0., 1.],
                            [1., 1.],
                            [2., 1.],
                            [0., 2.],
                            [1., 2.],
                            [2., 2.]])
    assert(np.sum(exp_idsamp - idsamp) < .001)
    assert(np.shape(tsamp) == (9, 3, 8, 3))


def _get_file_ids(set_root: file_sets.FileSet):
    fz = file_sets.get_files(set_root)
    return [fi.idn for fi in fz]


def test_double_sample():
    # simulates real experiments
    # > divide into train/cv vs. test
    # > divides train/cv --> train vs. cv
    root = _make_fake_tree()
    rng = npr.default_rng(666)
    # take all t0s in first split
    set1, set2 = set_sampling.get_anml_sample_allt0(root, .667, rng)
    trcv_ids = _get_file_ids(set1)
    assert(trcv_ids == [0, 2, 5, 4, 6, 7, 11, 9])
    assert(_get_file_ids(set2) == [1, 3, 8, 10])
    # switching sample on set1 (train/cv)
    train_set, cv_set = set_sampling.get_anml_sample_switch(set1, .5,
                                                            4, 2, 0, rng)
    train_ids = _get_file_ids(train_set)
    assert(train_ids == [0, 4, 7, 11])
    assert(set(train_ids).issubset(set(trcv_ids)))
    cv_ids = _get_file_ids(cv_set)
    assert(cv_ids == [2, 5, 6, 9])
    assert(set(cv_ids).issubset(trcv_ids))
    assert(set(cv_ids).isdisjoint(train_ids))

    # TODO: what about t0s???
    # given root set is not a great test set for this


if __name__ == '__main__':
    test_depth()
    test_idx_mapping()
    """
    test_sample_exe()
    test_animal_sampler()
    test_drawer()
    test_drawer_t0()
    test_double_sample()
    """
