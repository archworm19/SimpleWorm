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


def exe_plan(new_set: file_reps.FileSet,
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
            target_file = cur_set.files[ind]
            sel_files.append(fplan.sample_file(target_file))
        new_set.files = sel_files
    else:
        # NOTE: ss = child Plan
        for ss in cur_plan.sub_plan:
            set_idx = ss.set_idx
            new_sub_set = file_reps.FileSet([], None)
            new_set.sub_sets.append(new_sub_set)
            exe_plan(new_sub_set, cur_set.sub_sets[set_idx], ss)


# NOTE: stuff below here is likely to be specific, have limited generality


class DefaultFilePlan(FilePlan):
    # default: randomly sample a percentage of the t0s

    def __init__(self, sample_prob: float,
                 rng: npr.Generator):
        """sample_prob (float): sampling probability"""
        self.sample_prob = sample_prob
        self.rng = rng
    
    def sample_file(self, target_file: file_reps.SingleFile):
        return file_reps.sample_file_subset(target_file, self.sample_prob, self.rng)


# TODO: level sampler
# ... should be generally useful for plan creation


def _split_files(parent_set: file_reps.FileSet,
                 parent_plan1: Plan,
                 parent_plan2: Plan,
                 file_sample_prob: float,
                 t0_sample_prob: float,
                 rng: npr.Generator):
    # split files into primary and complementary files
    # --> save for the 2 plans
    # ASSUMES: at bottom level --> don't have to worry about further calls
    # select files:
    sel_inds, comp_inds = _sample_inds(rng, len(parent_set.files), file_sample_prob)
    for si in sel_inds:
        parent_plan1.sub_files[si] = DefaultFilePlan(t0_sample_prob, rng)
    for ci in comp_inds:
        parent_plan2.sub_files[ci] = DefaultFilePlan(t0_sample_prob, rng)


def _split_sets(parent_set: file_reps.FileSet,
                parent_plan1: Plan,
                parent_plan2: Plan,
                set_sample_prob: float,
                rng: npr.Generator):
    # split a set into 2 complementary sets
    # Returns: indices; inserted the new plans into parent plans
    sel_inds, comp_inds = _sample_inds(rng, len(parent_set.sub_sets), set_sample_prob)
    for si in sel_inds:
        parent_plan1.sub_plan.append(Plan(si, [], {}))
    for ci in comp_inds:
        parent_plan2.sub_plan.append(Plan(ci, [], {}))
    return sel_inds, comp_inds


def _plancopy(parent_set: file_reps.FileSet,
              parent_plan: Plan,
              t0_sample_prob: float,
              rng: npr.Generator):
    # Takes over after splitting --> select everybody

    # basecase: no more subsets --> must be files
    if len(parent_set.sub_sets) == 0:
        for i in range(len(parent_set.files)):
            parent_plan.sub_files[i] = DefaultFilePlan(t0_sample_prob, rng)
    else:
        for i, ss in enumerate(parent_set.sub_sets):
            new_plan = Plan(i, [], {})
            parent_plan.sub_plan.append(new_plan)
            _plancopy(ss, new_plan, t0_sample_prob, rng)


def _level_sample_planner(current_level: int,
                          parent_set: file_reps.FileSet,
                          parent_plan1: Plan,
                          parent_plan2: Plan,
                          split_level: int,
                          level_prob: float,
                          t0_sample_prob: float,
                          rng: npr.Generator):
    # split_level = which level to split at
    # level_prob = split probability for plan1
    # --> (1 - level_prob) for plan2

    # basecase: are we at splitting level:
    if current_level == split_level:
        if len(parent_set.sub_sets) == 0:
            _split_files(parent_set, parent_plan1, parent_plan2, level_prob, t0_sample_prob, rng)
        else:
            sel_inds, comp_inds = _split_sets(parent_set, parent_plan1, parent_plan2, level_prob, rng)
            indz = [sel_inds, comp_inds]
            pplanz = [parent_plan1, parent_plan2]
            for i in range(2):
                for j, child in enumerate(pplanz[i].sub_plan):
                    sij = indz[i][j]
                    _plancopy(parent_set.sub_sets[sij], child, t0_sample_prob, rng)
    
    # descend next level
    else:
        # ensure we're not at base
        assert(len(parent_set.sub_sets) > 0), "illegal level specified"
        for i, child in enumerate(parent_set.sub_sets):
            p1 = Plan(i, [], {})
            p2 = Plan(i, [], {})
            parent_plan1.sub_plan.append(p1)
            parent_plan2.sub_plan.append(p2)
            _level_sample_planner(current_level + 1, child,
                                  p1, p2, split_level,
                                  level_prob, t0_sample_prob, rng)


def level_sample_planner(set_root: file_reps.FileSet,
                         split_level: int,
                         split_prob: float,
                         t0_sample_prob: float,
                         rng: npr.Generator):
    """level sample planner
    Split root set into 2 complementary sets
    > Splits at level specified by split_level
    > Resulting sets will be identical before split

    Args:
        set_root (file_reps.FileSet): root set
        split_level (int): level of hierarchy at which
            we want to split
        split_prob (float): probability of given subset/subfile
            being allocated to primary set during split
        t0_sample_prob (fl0at): probability of timepoint
            being assigned to a set for selected files
        rng (npr.Generator):

    Returns:
        file_reps.FileSet: set
        file_reps.FileSet: complementary set
    """
    p1_root = Plan(0, [], {})
    p2_root = Plan(0, [], {})
    _level_sample_planner(0, set_root, p1_root, p2_root, split_level, split_prob,
                          t0_sample_prob, rng)
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
    sample_set1 = exe_plan(new_set, root_set, p1)
    sample_set2 = exe_plan(new_set, root_set, p2)
    return sample_set1, sample_set2
