"""  
    Sampling strategies

    Operate on file representations (file_reps.py)

    Design:
    > Everything operates on the tree
    > > FileSet tree
    > Sampling returns a subset of the tree
    > Rely on file names for record

"""
from typing import List
import numpy.random as npr
from Sampler import file_reps


def _sample_inds(rng: npr.Generator,
                 num_elems: int,
                 sample_prob: float):
    # Responsibility: random sampling via shuffling
    inds = [i for i in range(num_elems)]
    rng.shuffle(inds)
    N = int(sample_prob * num_elems)
    return inds[:N]


def _sample_helper(new_set: file_reps.FileSet,
                   rng: npr.Generator,
                   cur_set: file_reps.FileSet,
                   sample_probs: List[float]):
    """Make deepcopy of subset of current set
    new_set is the deep copy we're building up"""
    # basecase: reached file level --> add selected files to new_tree
    if len(sample_probs) == 1:
        assert(cur_set.files is not None), "sampler : file rep mismatch"
        num_files = len(cur_set.files)
        sfinds = _sample_inds(rng, num_files, sample_probs[0])
        sel_files = [file_reps.clone_single_file(cur_set.files[sfi])
                      for sfi in sfinds]
        new_set.files = sel_files
        return None
    else:
        num_subsets = len(cur_set.sub_sets)
        sel_subs = _sample_inds(rng, num_subsets, sample_probs[0])
        for ss in sel_subs:
            new_new_set = file_reps.FileSet([], None)
            new_set.sub_sets.append(new_new_set)
            child = cur_set.sub_sets[ss]
            _sample_helper(new_new_set, rng, child, sample_probs[1:])


def get_anml_sample(root_set: file_reps.FileSet,
                    anml_sample_prob: float,
                    rng: npr.Generator):
    """Anml sampling strategy
    > Assumes static hierarchy
    > Keeps all sets ~ samples animals within sets

    Args:
        root_set (file_reps.FileSet): _description_
        anml_sample_prob (float): sample probability
            for a given animal within each set
    
    Returns:
        file_reps.FileSet: root of the deepcopy of the subset
            of input set
    """
    depths = file_reps.get_depths(root_set)
    for de in depths[1:]:
        assert(de == depths[0]), "all depths must be the same"
    sample_probs = [1. for _ in range(de)] + [anml_sample_prob]
    new_set = file_reps.FileSet([], None)
    _sample_helper(new_set, rng, root_set, sample_probs)
    return new_set
