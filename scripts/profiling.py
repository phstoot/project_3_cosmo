import cProfile
from cosmosim.simulation import Universe

# profiler = cProfile.Profile()
# profiler.enable()

# test = Universe()
# test.run(steps=100)
# test.potential_to_acceleration() 

# profiler.disable()
# profiler.dump_stats("../results/profile.prof")



if __name__ == '__main__':
    test = Universe()
    test.run(steps=900)
