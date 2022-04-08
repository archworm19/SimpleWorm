"""  
    Sampling strategies

    Operate on file representations (file_reps.py)

    Design:
    > Everything operates on the tree
    > > FileSet tree
    > Sampling returns a subset of the tree
    > Rely on file names for record

    Plan vs. Execution Design:
    > Execution: takes in a sampling plan --> applies it to base FileSet
    > > Static across strategies
    > Plan: tree structure much like FileSet
    > > Alter the plan to customize sampling strategy

"""
import abc
from typing import List, Dict
import numpy as np
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

class AllSampleFilePlan(FilePlan):
    # just take all available t0s

    def __init__(self):
        pass

    def sample_file(self, target_file: file_reps.SingleFile):
        return file_reps.clone_single_file(target_file)

class RandFilePlan(FilePlan):
    # randomly sample a percentage of the t0s

    def __init__(self, sample_prob: float,
                 rng: npr.Generator):
        """sample_prob (float): sampling probability"""
        self.sample_prob = sample_prob
        self.rng = rng
    
    def sample_file(self, target_file: file_reps.SingleFile):
        return file_reps.sample_file_subset(target_file, self.sample_prob, self.rng)


class SwitchingFilePlan(FilePlan):
    # twindow: analysis timewindow
    # blocksize: number of adjacent windows that will be grouped together
    # start_hi: whether 0th block is selected or not
    # Logic?
    # > break trial into blocks
    # > blocks alternate between On/Off
    
    def __init__(self, twindow_size: int, block_size: int,
                 start_hi: bool,
                 offset: int):
        self.twindow_size = twindow_size
        self.block_size = block_size
        self.start_hi = start_hi
        self.offset = offset
    
    def _get_t0s(self, target_file: file_reps.SingleFile):
        tz, _, init_t0s = file_reps.open_file(target_file)
        return len(tz), init_t0s
    
    def sample_file(self, target_file: file_reps.SingleFile):
        L, init_t0s = self._get_t0s(target_file)
        markz = np.zeros((L,))
        hbit = 1 * self.start_hi
        for i in range(self.offset, L, self.block_size):
            markz[i:i+1+self.block_size-self.twindow_size] = hbit
            hbit = 1 - hbit
        # filter with init_t0s:
        fmask = np.zeros((L,))
        fmask[init_t0s] = 1
        markz = markz * fmask
        sel_t0s = np.where(markz)[0]
        return file_reps.sample_file(target_file, sel_t0s)


def _split_files(parent_set: file_reps.FileSet,
                 parent_plan1: Plan,
                 parent_plan2: Plan,
                 file_sample_prob: float,
                 file_plan1: FilePlan,
                 file_plan2: FilePlan,
                 rng: npr.Generator):
    # split files into primary and complementary files
    # --> save for the 2 plans
    # ASSUMES: at bottom level --> don't have to worry about further calls
    # select files:
    sel_inds, comp_inds = _sample_inds(rng, len(parent_set.files), file_sample_prob)
    for si in sel_inds:
        parent_plan1.sub_files[si] = file_plan1
    for ci in comp_inds:
        parent_plan2.sub_files[ci] = file_plan2


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
              file_plan: FilePlan,
              rng: npr.Generator):
    # Takes over after splitting --> select everybody

    # basecase: no more subsets --> must be files
    if len(parent_set.sub_sets) == 0:
        for i in range(len(parent_set.files)):
            parent_plan.sub_files[i] = file_plan
    else:
        for i, ss in enumerate(parent_set.sub_sets):
            new_plan = Plan(i, [], {})
            parent_plan.sub_plan.append(new_plan)
            _plancopy(ss, new_plan, file_plan, rng)


def _level_sample_planner(current_level: int,
                          parent_set: file_reps.FileSet,
                          parent_plan1: Plan,
                          parent_plan2: Plan,
                          split_level: int,
                          level_prob: float,
                          file_plan1: FilePlan,
                          file_plan2: FilePlan,
                          rng: npr.Generator):
    # split_level = which level to split at
    # level_prob = split probability for plan1
    # --> (1 - level_prob) for plan2

    # basecase: are we at splitting level:
    if current_level == split_level:
        if len(parent_set.sub_sets) == 0:
            _split_files(parent_set, parent_plan1, parent_plan2, level_prob,
                         file_plan1, file_plan2, rng)
        else:
            sel_inds, comp_inds = _split_sets(parent_set, parent_plan1, parent_plan2, level_prob, rng)
            indz = [sel_inds, comp_inds]
            pplanz = [parent_plan1, parent_plan2]
            fplanz = [file_plan1, file_plan2]
            for i in range(2):
                for j, child in enumerate(pplanz[i].sub_plan):
                    sij = indz[i][j]
                    _plancopy(parent_set.sub_sets[sij], child, fplanz[i], rng)
    
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
                                  level_prob,
                                  file_plan1, file_plan2, rng)


def level_sample_planner(set_root: file_reps.FileSet,
                         split_level: int,
                         split_prob: float,
                         file_plan1: FilePlan,
                         file_plan2: FilePlan,
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
        file_plan[1,2]: file plans for the primary and complemntary
            sets
        rng (npr.Generator):

    Returns:
        file_reps.FileSet: set
        file_reps.FileSet: complementary set
    """
    p1_root = Plan(0, [], {})
    p2_root = Plan(0, [], {})
    _level_sample_planner(0, set_root, p1_root, p2_root, split_level, split_prob,
                          file_plan1, file_plan2, rng)
    return p1_root, p2_root


def _get_anml_sample(root_set: file_reps.FileSet,
                     file_plan1: FilePlan,
                     file_plan2: FilePlan,
                     anml_sample_prob: float,
                     rng: npr.Generator):
    depths = file_reps.get_depths(root_set)
    for de in depths[1:]:
        assert(de == depths[0]), "all depths must be the same"

    pla1, pla2 = level_sample_planner(root_set, depths[0],
                                      anml_sample_prob,
                                      file_plan1, file_plan2,
                                      rng)
    new_set1 = file_reps.FileSet([], None)
    new_set2 = file_reps.FileSet([], None)
    exe_plan(new_set1, root_set, pla1)
    exe_plan(new_set2, root_set, pla2)
    return new_set1, new_set2


def get_anml_sample_switch(root_set: file_reps.FileSet,
                           anml_sample_prob: float,
                           block_size: int,
                           twindow_size: int,
                           offset: int,
                           rng: npr.Generator):
    """Anml sampling strategy + with switching file strategy
    > Assumes static hierarchy
    > Keeps all sets ~ samples animals within sets

    Args:
        root_set (file_reps.FileSet): _description_
        anml_sample_prob (float): sample probability
            for a given animal within each set
        block_size (int): number of adjacent timewindows
            that will fall in the same set
        twindow_size (int): analysis timewindow size
        offset (int): where to start t0 sampling
    
    Returns:
        file_reps.FileSet: root of the deepcopy of the first 
            subset of input set
        file_reps.FileSet: root of the deepcopy of the complement
            subset
    """
    file_plan1 = SwitchingFilePlan(twindow_size, block_size, True, offset)
    file_plan2 = SwitchingFilePlan(twindow_size, block_size, False, offset)
    return _get_anml_sample(root_set, file_plan1, file_plan2,
                            anml_sample_prob, rng)


def get_anml_sample_allt0(root_set: file_reps.FileSet,
                          anml_sample_prob: float,
                          rng: npr.Generator):
    file_plan = AllSampleFilePlan()
    return _get_anml_sample(root_set, file_plan, file_plan,
                            anml_sample_prob, rng)
