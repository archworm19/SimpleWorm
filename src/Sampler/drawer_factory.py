""" 
    Factory for creating Drawer objects

    High-level purpose?
    > Draw training/cross-valid and test sets
    > > Drawer will draw training vs. cv bootstraps
"""
import numpy.random as npr
from Sampler import drawer, set_sampling, file_reps


class AnmlDrawerFactory:

    def __init__(self,
                 root_set: file_reps.FileSet,
                 anml_sample_prob: float,
                 t0_sample_prob: float,
                 rng: npr.Generator):
        """Initialize the factory
        The point of the factory, overall, is to draw
        different splits on the root set
        NOTE: Anml factory (specifically) ensures splitting
        at the animal level ~ single animal won't be a member of
        each produced set

        Args:
            root_set (file_reps.FileSet): the root FileSet of the overall root set
            anml_sample_prob (float): animal sampling probability
            t0_sample_prob (float): t0 (within animal) sampling probability
            rng (npr.Generator):
        """
        self.root_set = root_set
        self.anml_sample_prob = anml_sample_prob
        self.t0_sample_prob = t0_sample_prob
        self.rng = rng

    def new_drawer(self):
        # TODO: should return drawers for both sets!

        # TODO: we need the complement set as well
        # TODO: get_anml_sample should produce both!!
        new_set = set_sampling.get_anml_sample(self.root_set,
                                               self.anml_sample_prob,
                                               self.t0_sample_prob,
                                               self.rng)
        