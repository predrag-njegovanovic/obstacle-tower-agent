import os
import multiprocessing
import definitions

from parallel_env_wrapper import ParallelEnvironment

NUM_THREADS = multiprocessing.cpu_count()

if __name__ == "__main__":
    obs_env_path = os.path.join(definitions.OBSTACLE_TOWER_DIR, 'obstacletower')
    env = ParallelEnvironment(obs_env_path, 2)
    env.start_parallel_execution()
    states = env.reset()
    print(states)
    # for i in range(10):
    #     samples = wrapper.sample()
    #     results = wrapper.step(samples)
    #     print(results)
