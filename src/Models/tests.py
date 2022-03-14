"""Testing Model integration with samplers"""
import numpy as np
import numpy.random as npr
import copy
from Sampler.experiments import build_anml_factory_multiset
from Models.KNN import knn

def knn_testing():
    # NOTE: stability requirement = num_samples >> number of dependent dims

    # need at least 3 animals for anml builder
    a1 = np.array([[1., 2., 3.]])
    a1 = np.tile(a1, (500, 1))
    dat = [a1 + npr.rand(500,3), a1+npr.rand(500,3),
                a1 + npr.rand(500,3)]
    
    # identity dat:
    iraw = np.reshape(np.arange(500), (-1, 1)) / 500.
    idat = [iraw, iraw, iraw]

    # independent vars = first 2 dims:
    twindow_size = 5
    indep_t_mask = np.tile(np.array([[True,True,False]]),
                         (twindow_size,1))
    indep_id_mask = np.array([True])
    dep_t_mask = np.logical_not(indep_t_mask)
    dep_id_mask = np.array([False])

    train_factory, test_sampler = build_anml_factory_multiset([dat], [idat], twindow_size, dep_t_mask, dep_id_mask,
                                                        indep_t_mask, indep_id_mask)
    
    # get a sampler:
    tr_sampler, cv_sampler = train_factory.generate_split()

    # build a KNN:
    # variances in independent space
    vars_t = np.ones((1,twindow_size,3))
    vars_id = np.ones((1,1))
    indep_vars, _ = tr_sampler.flatten_samples(vars_t, vars_id)
    print(indep_vars)
    k = knn.KNN(2, indep_vars)
    fbit, lls = k.train_epoch(tr_sampler)
    print(lls)

def knn_testing_tseries():
    # strong time effect
    # indep dims are random
    # dep dims are a func of time

    # need at least 3 animals for anml builder
    tz = np.arange(500) / 500.
    a1 = np.hstack((np.ones((500,2)), tz[:,None]))
    dat = [a1 + npr.rand(500,3), a1+npr.rand(500,3),
                a1 + npr.rand(500,3)]

    # identity dat:
    iraw = np.reshape(np.arange(500), (-1, 1)) / 500.
    idat = [iraw, iraw, iraw]

    # independent vars = first 2 dims:
    twindow_size = 5
    indep_t_mask = np.tile(np.array([[True,True,False]]),
                         (twindow_size,1))
    indep_id_mask = np.array([True])  # uses both id vars
    dep_t_mask = np.logical_not(indep_t_mask)
    dep_id_mask = np.array([False])

    train_factory, test_sampler = build_anml_factory_multiset([dat], [idat], twindow_size, dep_t_mask, dep_id_mask,
                                                        indep_t_mask, indep_id_mask)
    
    # get a sampler:
    tr_sampler, cv_sampler = train_factory.generate_split()

    # build a KNN:
    # variances in independent space
    # first: weight towards t series; then weight towards ids
    vars_t = np.ones((1,twindow_size,3))
    vars_id = np.ones((1,1))*1.  # ll = -1.745
    vars_id = np.ones((1,1))*.001  # ll = 0.798
    indep_vars, _ = tr_sampler.flatten_samples(vars_t, vars_id)
    print(indep_vars)
    k = knn.KNN(1, indep_vars)
    fbit, lls = k.train_epoch(tr_sampler)
    print(lls)


    
if __name__ == '__main__':
    knn_testing()

    knn_testing_tseries()
