"""Test Sampling
    > file_reps
    > set_sampling
    > TODO: sampling"""
import numpy as np
import numpy.random as npr
from Sampler import set_sampling, file_reps, drawer


def _make_fake_tree():
    # 2 top-level sets
    # each top-level set split into 2 subsets
    # each subset has 3 files
    # --> 4 subsets * 3 files each = 12 files
    # make the files
    sub_sets = []
    for i in range(4):
        ss = file_reps.FileSet([], [])
        for j in range(3):
            # relies on idn for identification
            f = file_reps.SingleFile(i*3 + j, "tfile", "idfile",
                                     [5,5,5], [5,4], "float32")
            ss.files.append(f)
        sub_sets.append(ss)
    top_set0 = file_reps.FileSet(sub_sets[:2], None)
    top_set1 = file_reps.FileSet(sub_sets[2:], None)
    root = file_reps.FileSet([top_set0, top_set1], None)
    return root


def test_depth():
    root = _make_fake_tree()
    v = file_reps.get_depths(root)
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
                v = file_reps.map_idx_to_file(root, set_idx, file_idx)
                assert(v.idn == count)
                count += 1


def test_sample_helper():
    root = _make_fake_tree()
    # keep top level + split 2nd level:
    rng = npr.default_rng(42)
    newset0 = file_reps.FileSet([], None)
    set_sampling._sample_helper(newset0, rng, root, [1., 0.5, 1.])
    filez = file_reps.get_files(newset0)
    exp_idns = set([3,4,5,6,7,8])
    assert(len(filez) == 6)
    for fi in filez:
        assert(fi.idn in exp_idns)
    
    # split on top level
    # also an implicit test of deep copying
    newset0 = file_reps.FileSet([], None)
    set_sampling._sample_helper(newset0, rng, root, [0.5, 1., .66667])
    filez = file_reps.get_files(newset0)
    exp_idns = set([8,6,9,11])
    assert(len(filez) == 4)
    for fi in filez:
        assert(fi.idn in exp_idns)


def test_anml_sampler():
    root = _make_fake_tree()
    rng = npr.default_rng(42)
    new_root = set_sampling.get_anml_sample(root, .66667, rng)
    filez = file_reps.get_files(new_root)
    exp_idns = set([6, 8, 10, 11, 0, 1, 5, 3])
    for fi in filez:
        assert(fi.idn in exp_idns)


def _make_fake_files():
    # make a single set with 3 animals
    tar0 = np.zeros((5, 8, 3))
    tar1 = np.ones((10, 8, 3))
    tar2 = np.ones((15, 8, 3)) * 2
    id0 = np.vstack((np.arange(5), np.zeros(5,))).T
    id1 = np.vstack((np.arange(10), np.ones((10,)))).T
    id2 = np.vstack((np.arange(15), 2*np.ones((15,)))).T
    tz = [tar0, tar1, tar2]
    idz = [id0, id1, id2]
    filez = []
    file_root = 'TEMP'
    for i in range(len(tz)):
        filez.append(file_reps.save_file(i, file_root + str(i), 
                                         tz[i], idz[i]))
    # package into single set:
    return file_reps.FileSet([], filez)


def test_drawer():
    root = _make_fake_files()
    twindow_size = 3
    D = drawer.TDrawer(root, twindow_size)
    avail_samples = D.get_available_samples()
    assert(avail_samples == 24)
    # idxs for file 0:
    tsamp, idsamp = D.draw_samples([0, 1, 2])
    assert(np.shape(tsamp) == (3, 3, 8, 3))
    assert(np.sum(tsamp != 0) < 1)
    assert(np.shape(idsamp) == (3, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 0)
    # idxs for file 1:
    tsamp, idsamp = D.draw_samples(np.arange(3,3+8))
    assert(np.shape(tsamp) == (8, 3, 8, 3))
    assert(np.sum(tsamp != 1) < 1)
    assert(np.shape(idsamp) == (8, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 1)
    # idxs for file 2:
    tsamp, idsamp = D.draw_samples(np.arange(11, 11+13))
    assert(np.shape(tsamp) == (13, 3, 8, 3))
    assert(np.sum(tsamp != 2) < 1)
    assert(np.shape(idsamp) == (13, 2))
    assert(idsamp[0,0] == 0)
    assert(idsamp[0,1] == 2)


if __name__ == '__main__':
    test_depth()
    test_idx_mapping()
    test_sample_helper()
    test_anml_sampler()
    test_drawer()
