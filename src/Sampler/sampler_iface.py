"""
    Sampling interface
"""
import abc

class Sampler(abc.ABC):

    def set_new_epoch(self):
        pass

    def pull_train_samples(self, num_samples: int):
        pass

    def pull_test_samples(self, num_samples: int):
        pass

