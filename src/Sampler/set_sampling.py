"""  
    Sampling strategies

    Operate on file representations (file_reps.py)

    Design:
    > Everything operates on the tree
    > > FileSet tree
    > Sampling returns a subset of the tree
    > Rely on file names for record

"""
import abc
from typing import List, Dict
import numpy.random as npr
from Sampler import file_reps


def _sample_inds(rng: npr.Generator,
                 num_elems: int,
                 sample_prob: float):
    # Responsibility: random sampling via shuffling
    # returns primary and complement
    inds = [i for i in range(num_elems)]
    rng.shuffle(inds)
    N = int(sample_prob * num_elems)
    return inds[:N], inds[N:]


# TODO: best way to do complements?
# Idea 1
# > methods
# > > sample files
# > > > get complements
# > > sample within files
# Idea 2
# > Methods
# > > plan
# > > > generate a plan that can be modified or executed
# > > > ... have a number of helper methods for plan generation
# > > execute
# > > > what we have right now

class FilePlan(abc.ABC):

    def sample_file(self, target_file: file_reps.SingleFile):
        # Returns deep copy of SingleFile
        pass


class Plan:
    # pseudo-dataclass
    def __init__(self,
                 set_idx: int,
                 sub_plan: List,
                 sub_files: Dict[int, FilePlan]):
        self.set_idx = set_idx
        self.sub_plan = sub_plan
        self.sub_files = sub_files


def _exe_plan(new_set: file_reps.FileSet,
              cur_set: file_reps.FileSet,
              cur_plan: Plan):
    """Execute current level of the plan

    Args:
        new_set (file_reps.FileSet): new parent set
        cur_set (file_reps.FileSet): current parent set
        cur_plan (Plan): current plan
    """
    if len(cur_plan.sub_plan) == 0:
        assert(cur_set.files is not None), "plan: file rep mismatch"
        sel_files = []
        for ind in cur_plan.sub_files:
            fplan = cur_plan.sub_files[ind]
            sel_files.append(fplan.sample_file(cur_set.files[ind]))
        new_set.files = sel_files
    else:
        # NOTE: ss = child Plan
        for ss in cur_plan.sub_plan:
            set_idx = ss.set_idx
            new_sub_set = file_reps.FileSet([], None)
            new_set.sub_sets.append(new_sub_set)
            _exe_plan(new_sub_set, cur_set.sub_sets[set_idx], ss)


class DefaultFilePlan(FilePlan):
    # default: randomly sample a percentage of the t0s

    def __init__(self, sample_prob: float,
                 rng: npr.Generator):
        """sample_prob (float): sampling probability"""
        self.sample_prob = sample_prob
        self.rng = rng
    
    def sample_file(self, target_file: file_reps.SingleFile):
        return file_reps.sample_file_subset(target_file, self.sample_prob, self.rng)


def _default_plan_creation(parent_set: file_reps.FileSet,
                           parent_plan1: Plan, parent_plan2: Plan,
                           sample_prob: List[float],
                           file_sample_prob: float, rng: npr.Generator):
    if len(sample_prob) == 1:  # assumed to be for selecting fiels
        assert(len(parent_set.files) > 0), "no files found"
        # select files:
        sel_inds, comp_inds = _sample_inds(rng, len(parent_set.files), sample_prob[0])
        for si in sel_inds:
            parent_plan1.sub_files[si] = DefaultFilePlan(file_sample_prob, rng)
        for ci in comp_inds:
            parent_plan2.sub_files[ci] = DefaultFilePlan(file_sample_prob, rng)
    else:  # not at file level yet
        # don't split sets
        for i, subset in enumerate(parent_set.sub_sets):
            v1 = Plan(i, [], {})
            v2 = Plan(i, [], {})
            parent_plan1.sub_plan.append(v1)
            parent_plan2.sub_plan.append(v2)
            _default_plan_creation(subset, v1, v2, sample_prob[1:], file_sample_prob, rng)


# TODO: default plan creation
# = split parent set into 2 complementary sets + retain 
def default_plan_creation(set_root: file_reps.FileSet,
                          sample_probs: List[float], 
                          file_sample_prob: float,
                          rng: npr.Generator):
    p1_root = Plan(0, [], {})
    p2_root = Plan(0, [], {})
    _default_plan_creation(set_root, p1_root, p2_root, sample_probs, file_sample_prob, rng)
    return p1_root, p2_root


def get_anml_sample(root_set: file_reps.FileSet,
                    anml_sample_prob: float,
                    t0_sample_prob: float,
                    rng: npr.Generator):
    """Anml sampling strategy
    > Assumes static hierarchy
    > Keeps all sets ~ samples animals within sets

    Args:
        root_set (file_reps.FileSet): _description_
        anml_sample_prob (float): sample probability
            for a given animal within each set
        t0_sample_prob (float): selection probability
            for each t0 within each file
    
    Returns:
        file_reps.FileSet: root of the deepcopy of the first 
            subset of input set
        file_reps.FileSet: root of the deepcopy of the complement
            subset
    """
    depths = file_reps.get_depths(root_set)
    for de in depths[1:]:
        assert(de == depths[0]), "all depths must be the same"
    sample_probs = [1. for _ in range(de)] + [anml_sample_prob]
    # get plans:
    p1, p2 = default_plan_creation(root_set, sample_probs, t0_sample_prob, rng)
    # execute:
    new_set = file_reps.FileSet([], None)
    sample_set1 = _exe_plan(new_set, root_set, p1)
    sample_set2 = _exe_plan(new_set, root_set, p2)
    return sample_set1, sample_set2
