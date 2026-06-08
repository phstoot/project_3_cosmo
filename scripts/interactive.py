import numpy as np
import matplotlib.pyplot as plt
from cosmosim.simulation import Universe

# analyse profiling results
import pstats

stats = pstats.Stats("results/profile.prof")
stats.sort_stats("cumulative").print_stats(30)



if __name__ == '__main__':
    test = Universe()
    test.run(steps=900)
    test.plot()