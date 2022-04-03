"""Test Sampling
    > file_reps
    > set_sampling
    > TODO: sampling"""
import copy
import numpy as np
import numpy.random as npr
from Sampler import set_sampling, file_reps, drawer


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
        ss = file_reps.FileSet([], [])
        for j in range(3):
            # relies on idn for identification
            f = file_reps.save_file(i*3 + j, "TEMP", t_ar, id_ar, t0_samples)
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


# just return the file
class IDFP(set_sampling.FilePlan):
    def __init__(self):
        pass
    def sample_file(self, target_file: file_reps.SingleFile):
        return copy.deepcopy(target_file)


def test_sample_exe():
    # fake tree --> 2 top-level > 2 each > 3 files
    root = _make_fake_tree()

    # keep top level + split 2nd level:
    pl_root = set_sampling.Plan(0, [], {})
    for i in range(2):
        vi = set_sampling.Plan(i, [], {})
        pl_root.sub_plan.append(vi)
        for j in [0]:
            vj = set_sampling.Plan(j, [], {})
            vi.sub_plan.append(vj)
            for k in range(3):
                vj.sub_files[k] = IDFP()

    newset0 = file_reps.FileSet([], None)
    set_sampling.exe_plan(newset0, root, pl_root)
    filez = file_reps.get_files(newset0)
    exp_idns = set([0,1,2,6,7,8])
    assert(len(filez) == 6)
    for fi in filez:
        assert(fi.idn in exp_idns)
    
    # split on top level
    # also an implicit test of deep copying
    pl_root = set_sampling.Plan(0, [], {})
    for i in [1]:
        vi = set_sampling.Plan(i, [], {})
        pl_root.sub_plan.append(vi)
        for j in [0, 1]:
            vj = set_sampling.Plan(j, [], {})
            vi.sub_plan.append(vj)
            for k in range(3):
                vj.sub_files[k] = IDFP()

    newset0 = file_reps.FileSet([], None)
    set_sampling.exe_plan(newset0, root, pl_root)
    filez = file_reps.get_files(newset0)
    exp_idns = set([6,7,8,9,10,11])
    assert(len(filez) == 6)
    for fi in filez:
        assert(fi.idn in exp_idns)


def test_anml_sampler():
    root = _make_fake_tree()
    rng = npr.default_rng(42)
    new_root = set_sampling.get_anml_sample(root, .66667, 1., rng)
    filez = file_reps.get_files(new_root)
    exp_idns = set([6, 8, 9, 10, 5, 4, 1, 2])
    for fi in filez:
        assert(fi.idn in exp_idns)


def _make_fake_files(t0_sub: bool = False):
    # make a single set with 3 animals
    tar0 = np.zeros((5, 8, 3))
    tar1 = np.ones((10, 8, 3))
    tar2 = np.ones((15, 8, 3)) * 2
    id0 = np.vstack((np.arange(5), np.zeros(5,))).T
    id1 = np.vstack((np.arange(10), np.ones((10,)))).T
    id2 = np.vstack((np.arange(15), 2*np.ones((15,)))).T
    if t0_sub:
        t00 = np.arange(3)
        t01 = np.arange(3)
        t02 = np.arange(3)
    else:
        t00 = np.arange(5)
        t01 = np.arange(10)
        t02 = np.arange(15)
    tz = [tar0, tar1, tar2]
    idz = [id0, id1, id2]
    t0z = [t00, t01, t02]
    filez = []
    file_root = 'TEMP'
    for i in range(len(tz)):
        filez.append(file_reps.save_file(i, file_root + str(i), 
                                         tz[i], idz[i], t0z[i]))
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


def test_drawer_t0():
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


if __name__ == '__main__':
    test_depth()
    test_idx_mapping()
    test_sample_exe()
    #test_anml_sampler()
    #test_drawer()
    #test_drawer_t0()
