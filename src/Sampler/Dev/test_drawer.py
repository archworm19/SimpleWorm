
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
