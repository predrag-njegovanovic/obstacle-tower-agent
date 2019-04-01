import os
import multiprocessing
import definitions

from gym import spaces
from obstacle_tower_env import ObstacleTowerEnv
from parallel_env_wrapper import ParallelEnvironmentWrapper

NUM_THREADS = multiprocessing.cpu_count()

if __name__ == "__main__":
    obs_env_path = os.path.join(definitions.OBSTACLE_TOWER_DIR, 'obstacletower')
    env = ObstacleTowerEnv(obs_env_path, retro=False, realtime_mode=False)

    state, reward, done, info = env.step(env.action_space.sample())
    image, keys, time = state
    print(image)
    # wrapper = ParallelEnvironmentWrapper(obs_env_path, NUM_THREADS)
    # wrapper.start_parallel_execution()
    # for i in range(10):
    #     samples = wrapper.sample()
    #     results = wrapper.step(samples)
    #     print(results)
