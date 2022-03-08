"""T Variance Analysis

    Stimulus memorization tradeoff:
    > 2 window sizes
    > > T_disc
    > > > Discretization window sizes
    > > T_awin
    > > > Analysis window size
    > > Consider all analysis windows within each
    > > Discretization window
    > > Combine across animals
    > Q? for fixed T_awin, at what T_disc does variance drop off?
    > == this is the variance where stim memorization is possible
"""

if __name__ == "__main__":
    T_awin = 24
    T_discs = [10, 20, 50, 100, 200]
    