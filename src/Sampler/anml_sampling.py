"""  
    Sampling strategies

    Operate on file representations (file_reps.py)

    TODO: rename file == general enough to handle more than just animal sampling

"""
from typing import List
import numpy.random as npr
from Sampler import file_reps


def _sample_inds(rng: npr.Generator,
                 num_elems: int,
                 sample_prob: int):
    # Responsibility: random sampling via shuffling
    inds = [i for i in range(num_elems)]
    rng.shuffle(inds)
    N = int(sample_prob * num_elems)
    return inds[:N]


def _sample_helper(path: List[int],
                   rng: npr.Generator,
                   cur_set: file_reps.FileSet,
                   sample_probs: List[float]):
    # TODO/NOTE: path = selected sets to this point
    # basecase: reached file level
    if len(sample_probs) == 1:
        assert(cur_set.files is not None), "sampler : file rep mismatch"
        num_files = len(cur_set.files)
        sel_files = _sample_inds(rng, num_files, sample_probs[0])
        full_paths = [path + [sf] for sf in sel_files]
        return full_paths
    else:
        num_subsets = len(cur_set.sub_sets)
        sel_subs = _sample_inds(rng, num_subsets, sample_probs[0])
        up_paths = [path + [ss] for ss in sel_subs]
        ret_paths = []
        for upp in up_paths:
            child = cur_set.get_subset(upp)
            v = _sample_helper(path + [upp], rng, child, sample_probs[1:])
            ret_paths.extend(v)
        return ret_paths
        

# TODO: STRAT
# > All strats can use _sample_helper
# > Different strats will wrap this guy (anml_sampler)
# > TODO: how to do a guide? (for set > bootstrap)
# > > _sample_helper should operate on an existing path
# > > Add in additional function for creating the mapping!!


def get_sample(root_set: file_reps.FileSet,
               sample_probs: List[float]):
    """Sample 
    TODO: finish doc string
    TODO: explain each elem of sample_probs / relationship to root_set struct

    Args:
        root_set (file_reps.FileSet): _description_
        sample_probs (List[float]): probability for each
            level of the hierarchy
            Ex: [1., 0.5] means sample every top set
                --> sample half the animals within each set
    """
    pass
    