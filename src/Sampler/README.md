Sampler Design

Models Need:
> Drawer
> > Specifies available data
> > Handles pulling of data
> Flattener
> > Un/Flattens data


Drawer Factory:
> Initialized with 
> > file_reps.FileSet = base set
> > random seed -> rng
> Yields
> > Drawer


Experiments:
> Map raw data to Drawer Factory