"""test data sourcing ops"""
import numpy as np
from Sampler.data_source import DataSourceAlternating


def test_alternating_src():
    A = np.array([[1, 1] for _ in range(10)])
    B = np.array([[2, 2] for _ in range(10)])
    DS = DataSourceAlternating({"0": np.vstack([A, B, A, B, A]),
                                "1": np.vstack([B, A, B, A, B])},
                                10, True)
    dtrain = DS.get_numpy_data(["0", "1"], True)
    dtest = DS.get_numpy_data(["0", "1"], False)
    # train
    assert(np.all(dtrain["0"] == np.vstack([A, A, A])))
    assert(np.all(dtrain["1"] == np.vstack([B, B, B])))
    # test:
    assert(np.all(dtest["0"] == np.vstack([B, B])))
    assert(np.all(dtest["1"] == np.vstack([A, A])))

    # try again with opposite chunking strat
    DS = DataSourceAlternating({"0": np.vstack([A, B, A, B, A]),
                                "1": np.vstack([B, A, B, A, B])},
                                10, False)
    dtrain = DS.get_numpy_data(["0", "1"], True)
    dtest = DS.get_numpy_data(["0", "1"], False)
    # train
    assert(np.all(dtrain["0"] == np.vstack([B, B])))
    assert(np.all(dtrain["1"] == np.vstack([A, A])))
    # test:
    assert(np.all(dtest["0"] == np.vstack([A, A, A])))
    assert(np.all(dtest["1"] == np.vstack([B, B, B])))


if __name__ == "__main__":
    test_alternating_src()
