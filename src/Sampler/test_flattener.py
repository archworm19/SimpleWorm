"""Test Flattener"""
import numpy as np
from Sampler.flattener import Flattener

def _get_masks_dat():
    # dims:
    T = 3
    d1 = 2
    d2 = 3
    # data ~ 3 stacks of T
    # id data = 3*T x 2
    # time on 0; 66 on 1
    dat_id = np.vstack((np.arange(3*T), 66*np.ones((3*T,)))).T
    # t data = 3*T x d1 x d2
    # dat_t[0] = 0; dat_t[1] = 1; etc.
    dat_t = np.zeros((3*T, d1, d2))
    for z in range(3*T):
        dat_t[z] = z
    
    # masks
    # dep id mask: 66
    dep_id_mask = np.array([False, True])
    # indep id mask: t
    indep_id_mask = np.array([True, False])
    # dep t mask: even timez
    dep_t_mask = np.mod(dat_t[:T], 2) == 0
    # indep_t_mask: odd timez
    indep_t_mask = np.mod(dat_t[:T], 2) == 1

    return [dat_id, dat_t], [dep_id_mask, indep_id_mask, dep_t_mask, indep_t_mask], T


def _draw_samples(T, dat_t, dat_id):
    # mimics what Drawer will do
    # = for testing purposes; drawer is more extensive
    t_samp, id_samp = [], []
    for i in range(0, len(dat_t)-T+1):
        t_samp.append(dat_t[i:i+T])
        id_samp.append(dat_id[i])
    return np.array(t_samp), np.array(id_samp)


def test_flatten():
    dat, masks, T = _get_masks_dat()
    [dat_id, dat_t] = dat
    [dep_id_mask, indep_id_mask, dep_t_mask, indep_t_mask] = masks
    F = Flattener(dep_t_mask, dep_id_mask, indep_t_mask, indep_id_mask)
    # get the samples:
    t_samp, id_samp = _draw_samples(T, dat_t, dat_id)
    # flatten all dat
    f_indep, f_dep = F.flatten_samples(t_samp, id_samp)
    exp_f_indep = np.array([[1., 1., 1., 1., 1., 1., 0.],
                            [2., 2., 2., 2., 2., 2., 1.],
                            [3., 3., 3., 3., 3., 3., 2.],
                            [4., 4., 4., 4., 4., 4., 3.],
                            [5., 5., 5., 5., 5., 5., 4.],
                            [6., 6., 6., 6., 6., 6., 5.],
                            [7., 7., 7., 7., 7., 7., 6.]])
    exp_f_dep = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  2.,  2.,  2.,  2.,  2.,  2., 66.],
                            [ 1.,  1.,  1.,  1.,  1.,  1.,  3.,  3.,  3.,  3.,  3.,  3., 66.],
                            [ 2., 2.,  2.,  2.,  2.,  2.,  4.,  4.,  4.,  4.,  4.,  4., 66.],
                            [ 3.,  3.,  3.,  3.,  3.,  3.,  5.,  5.,  5.,  5.,  5.,  5., 66.],
                            [ 4.,  4.,  4.,  4.,  4.,  4.,  6.,  6.,  6.,  6.,  6.,  6., 66.],
                            [ 5.,  5.,  5.,  5.,  5.,  5.,  7.,  7.,  7.,  7.,  7.,  7., 66.],
                            [ 6.,  6.,  6.,  6.,  6.,  6.,  8.,  8.,  8.,  8.,  8.,  8., 66.]])
    assert(np.sum(exp_f_indep != f_indep) < 1)
    assert(np.sum(exp_f_dep != f_dep) < 1)


def _reshape_mask(mask):
    mask2 = np.ones(np.shape(mask)) * np.nan
    mask2[mask] = 1.
    return mask2[None]


def test_unflatten():
    dat, masks, T = _get_masks_dat()
    [dat_id, dat_t] = dat
    [dep_id_mask, indep_id_mask, dep_t_mask, indep_t_mask] = masks
    F = Flattener(dep_t_mask, dep_id_mask, indep_t_mask, indep_id_mask)
    # get the samples:
    t_samp, id_samp = _draw_samples(T, dat_t, dat_id)
    # flatten all dat
    f_indep, f_dep = F.flatten_samples(t_samp, id_samp)
    # unflatten indep:
    uf_t_indep, uf_id_indep = F.unflatten_samples(f_indep, indep=True)
    # unflatten dep:
    uf_t_dep, uf_id_dep = F.unflatten_samples(f_dep, indep=False)

    # compare
    all_uf = [uf_t_indep, uf_id_indep, uf_t_dep, uf_id_dep]
    all_mask = [F.indep_t_mask, F.indep_id_mask, F.dep_t_mask, F.dep_id_mask]
    all_samp = [t_samp, id_samp, t_samp, id_samp]
    for i in range(len(all_uf)):
        mask2 = _reshape_mask(all_mask[i])
        re_create = mask2 * all_samp[i]

        # compare nans:
        assert(np.sum(np.isnan(re_create) != np.isnan(all_uf[i])) < 1)

        # compare nonnans:
        nn_loc = np.logical_not(np.isnan(re_create))
        assert(np.sum(re_create[nn_loc] != all_uf[i][nn_loc]) < 1)


if __name__ == "__main__":
    test_flatten()
    test_unflatten()
