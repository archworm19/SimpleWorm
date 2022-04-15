"""  
    Sampling strategies

    Operate on file representations (file_reps.py)

    Design:
    > Everything operates on the tree
    > > FileSet tree
    > Sampling returns a subset of the tree
    > Rely on file names for record

    Plan Design:
    > Execution: take in
    > > Set Sampler Strategy
    > > File Sampler Strategy
    > > t0 Sampler Strategy
    > Execution passes in: 1. path List[int], 2. Current set/file when applicable

"""
import abc
from typing import List, Dict
import numpy as np
import numpy.random as npr
from Sampler import file_reps


# Execution


class SetSamplerStrat(abc.ABC):
    # Which subsets to sample
    def sample(self, set_path: List[int],
               current_set: file_reps.FileSet):
        # Returns: List[int] = indices of children to use
        pass


class FileSamplerStrat(abc.ABC):
    # which subfiles to sample
    def sample(self, set_path: List[int],
               current_set: file_reps.FileSet):
        # Returns: List[int] = indices of children to use
        pass


class t0SamplerStrat(abc.ABC):
    # select t0s from available
    def sample(self, set_path: List[int],
               file_idx: int,
               target_file: file_reps.SingleFile):
        # returns: deepcopy of target file
        pass


def exe_plan(set_path: List[int],
             new_set: file_reps.FileSet,
             cur_set: file_reps.FileSet,
             set_strat: SetSamplerStrat,
             file_strat: FileSamplerStrat,
             t0_strat: t0SamplerStrat):
    """Execute a sampling plan
    > allows internal files; files don't have to be leaves

    Args:
        set_path (List[int]): sub-sets that have been visited
            length of list = how deep we are in set hierarchy
        new_set (file_reps.FileSet): parent file_reps.FileSet;
            we will insert selected children
        cur_set (file_reps.FileSet): parent file_reps.Fileset;
            from which we are copying subsets
        set_strat (SetSamplerStrat): set sampling strategy
        file_strat (FileSamplerStrat): file sampling strategy
        t0_strat (t0SamplerStrat): t0 sampling strategy
            = available twindows within file
    """
    # check if there are files at given level:
    if len(cur_set.files) > 0:
        # get available files:
        select_file_inds = file_strat.sample(set_path, cur_set)
        for si in select_file_inds:
            cpy_file = t0_strat.sample(set_path, si, cur_set.files[si])
            new_set.files.append(cpy_file)
    
    # check if there are subsets:
    if len(cur_set.sub_sets) > 0:
        # get available subsets:
        select_subset_inds = set_strat.sample(set_path, cur_set)
        for si in select_subset_inds:
            new_sub = file_reps.FileSet([], [])
            new_set.sub_sets.append(new_sub)
            exe_plan(set_path + [si], new_sub, cur_set.sub_sets[si],
                     set_strat, file_strat, t0_strat)


# Useful strategies


def _sample_inds(rng: npr.Generator,
                 num_elems: int,
                 sample_prob: float):
    # Responsibility: random sampling via shuffling
    # returns primary and complement
    inds = [i for i in range(num_elems)]
    rng.shuffle(inds)
    N = int(sample_prob * num_elems)
    return inds[:N], inds[N:]


# Set strategies


class LvlSplitter(SetSamplerStrat):
    # > random upon init ~ calling sample always returns the same result
    # > if at spec level --> return subset of inds; else --> return all inds
    def __init__(self,
                 split_level: int,
                 sample_map: Dict[str, List[int]]):
        self.split_level = split_level
        self.sample_map = sample_map
    
    def sample(self, set_path: List[int], current_set: file_reps.FileSet):
        if len(set_path) == self.split_level:
            return self.sample_map[str(set_path)]
        return [z for z in range(len(current_set.sub_sets))]


class LeafFileSplitter(FileSamplerStrat):
    # only select files at leaves; select no internal files 
    def __init__(self, sample_map: Dict[str, List[int]]):
        self.sample_map = sample_map
    
    def sample(self, set_path: List[int], current_set: file_reps.FileSet):
        return self.sample_map[str(set_path)]


# t0 strats


class Allt0Strat(t0SamplerStrat):
    def __init__(self):
        pass
    def sample(self, set_path: List[int], file_idx: int, target_file: file_reps.SingleFile):
        return file_reps.clone_single_file(target_file)


class Switchingt0Strat(t0SamplerStrat):
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
    
    def sample(self, set_path: List[int], file_idx: int, target_file: file_reps.SingleFile):    
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


# Useful Strat Builders


def build_comp_sets(root_set: file_reps.FileSet,
                    split_level: int,
                    split_prob: float,
                    rng: npr.Generator):
    """Build complementary Subset sampling strategies

    Args:
        root_set (file_reps.FileSet): root set
        split_level (int): level in hierarchy at which to split
        split_prob (float): probability of subset being assign to
            primary set
        rng (npr.Generator): 

    Returns:
        LvlSplitter: strat 1
        LvlSplitter: strat 2
    """
    # get all subsets at specified level
    # --> List[List[int]], List[int]
    paths, num_sub = file_reps.get_num_sets_lvl(root_set, split_level)
    # split into 2 complementary sets:
    sample_map, comp_map = {}, {}
    # allocate within each group:
    for pathi, num_subi in zip(paths, num_sub):
        sinds, cinds = _sample_inds(rng, num_subi, split_prob)
        sample_map[str(pathi)] = sinds
        comp_map[str(pathi)] = cinds
    return LvlSplitter(split_level, sample_map), LvlSplitter(split_level, comp_map)


def build_comp_files(root_set: file_reps.FileSet,
                     split_prob: float,
                     rng: npr.Generator):
    """Build complementary file sampling strategies
    > LeafSplitter = will ignore internal files

    Args:
        root_set (file_reps.FileSet): root set
        split_prob (float): probability of subset being assign to
            primary set
        rng (npr.Generator): 

    Returns:
        LeafFileSplitter: strat 1
        LeafFileSplitter: strat 2
    """
    # --> List[List[int]], List[List[Files]]
    paths, filez = file_reps.get_files_struct(root_set)
    # split into 2 complementary sets:
    sample_map, comp_map = {}, {}
    # allocate within each group:
    for pathi, filezi in zip(paths, filez):
        sinds, cinds = _sample_inds(rng, len(filezi), split_prob)
        sample_map[str(pathi)] = sinds
        comp_map[str(pathi)] = cinds
    return LeafFileSplitter(sample_map), LeafFileSplitter(comp_map)


# Animal Sampling
# > sample at file level
# > sample within subsets


def _get_anml_sample(root_set: file_reps.FileSet,
                     t0_strat1: t0SamplerStrat,
                     t0_strat2: t0SamplerStrat,
                     anml_sample_prob: float,
                     rng: npr.Generator):
    depths = file_reps.get_depths(root_set)
    for de in depths[1:]:
        assert(de == depths[0]), "all depths must be the same"

    # None --> never split; select all
    set_strat = LvlSplitter(None, {})
    file_strat1, file_strat2 = build_comp_files(root_set, anml_sample_prob, rng)

    # primary set:
    new_set1 = file_reps.FileSet([], [])
    exe_plan([], new_set1, root_set, set_strat, file_strat1, t0_strat1)

    # complementary set:
    new_set2 = file_reps.FileSet([], [])
    exe_plan([], new_set2, root_set, set_strat, file_strat2, t0_strat2)
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
    t0_strat1 = Switchingt0Strat(twindow_size, block_size, True, offset)
    t0_strat2 = Switchingt0Strat(twindow_size, block_size, False, offset)
    return _get_anml_sample(root_set, t0_strat1, t0_strat2, anml_sample_prob, rng)


def get_anml_sample_allt0(root_set: file_reps.FileSet,
                          anml_sample_prob: float,
                          rng: npr.Generator):
    # get animal sample + take all t0s from selected files
    t0_strat = Allt0Strat()
    return _get_anml_sample(root_set, t0_strat, t0_strat,
                            anml_sample_prob, rng)


def sample_avail_files(root_set: file_reps.FileSet,
                       target_cell: int):
    """Sample available files
    'Available files' = files that have no nans
        for the target cell
    ASSUMES: time series shape for files = T x N
        where N = number of cells

    Args:
        root_set (file_reps.FileSet):
        target_cell (int): target cell that
            must be defined for all timepoints
    
    Returns:
        file_reps.FileSet: root of deepcopy of
            subset
    """
    # this class is internal as it's not generally useful
    class CFileSampler(FileSamplerStrat):
        # only select files at leaves; select no internal files 
        def __init__(self, tcell):
            self.target_cell = tcell
        
        # Returns: List[int] = indices of children to use
        def sample(self, set_path: List[int], current_set: file_reps.FileSet):
            inds = []
            for i, fi in enumerate(current_set.files):
                tdat, _, _ = file_reps.open_file(fi)
                vnan = np.sum(np.isnan(tdat[:, self.target_cell]))
                if vnan < 1:
                    inds.append(i)
            return inds
    
    # strats:
    # None --> never split; select all
    set_strat = LvlSplitter(None, {})
    file_strat = CFileSampler(target_cell)
    t0_strat = Allt0Strat()

    new_set = file_reps.FileSet([], [])
    exe_plan([], new_set, root_set,
             set_strat, file_strat, t0_strat)
    return new_set
