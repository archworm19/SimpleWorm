"""
    Sampling interface
"""
import abc

class Sampler(abc.ABC):

    def pull_samples(self, num_samples: int):
        pass

    def epoch_reset(self):
        pass

    def shuffle(self, rng_seed: int = 42):
        pass


