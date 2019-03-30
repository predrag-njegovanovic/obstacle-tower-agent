import os
import multiprocessing
import definitions

from env_wrapper import ParallelWrapper

NUM_THREADS = multiprocessing.cpu_count()

if __name__ == "__main__":
    obs_env_path = os.path.join(definitions.OBSTACLE_TOWER_DIR, 'obstacletower')
    wrapper = ParallelWrapper(obs_env_path)
    parallel_env = wrapper.wrapped_environment(1)
