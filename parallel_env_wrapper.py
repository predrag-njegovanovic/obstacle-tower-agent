import cv2
import torch

from multiprocessing import Process, Pipe
from obstacle_tower_env import ObstacleTowerEnv


def start_environment(connection, worker_id, env_path, retro, realtime_mode):
    obstacle_tower = ObstacleTowerEnv(env_path,
                                      worker_id=worker_id,
                                      retro=retro,
                                      timeout_wait=90,
                                      realtime_mode=False)
    obstacle_tower.reset()
    obstacle_tower.seed(worker_id)
    while True:
        command, action = connection.recv()
        if command == 'sample':
            connection.send(obstacle_tower.action_space.sample())
        if command == 'step':
            observation, reward, done, info = obstacle_tower.step(action)
            state, keys, time = observation

            if done:
                state = obstacle_tower.reset()
                state, keys, time = observation

            connection.send((prepare_state(state), keys, time, reward, info, done))
        elif command == 'reset':
            state, keys, time = obstacle_tower.reset()
            connection.send((prepare_state(state), keys, time))
        elif command == 'close':
            connection.close()


def prepare_state(state):
    """
    Convert array to pytorch.Tensor and reshape it as (C, H, W)
    """
    frame = cv2.resize(state, (84, 84, 3))
    height, width, channels = frame.shape
    state_tensor = torch.Tensor(frame).view(channels, height, width)
    return state_tensor


class ParallelEnvironmentWrapper:
    def __init__(self, env_path, num_of_processes, retro=False, realtime_mode=False):
        self.parent_connections, self.child_connections = zip(*[Pipe() for _
                                                                in range(num_of_processes)])
        self.env_path = env_path
        self.retro = retro
        self.realtime_mode = realtime_mode
        self.processes = None

    def start_parallel_execution(self):
        self.processes = [Process(target=start_environment,
                                  args=(child, worker_id, self.env_path,
                                        self.retro, self.realtime_mode),
                                  daemon=True)
                          for worker_id, child in enumerate(self.child_connections)]

        for process in self.processes:
            process.start()

    def sample(self):
        [parent.send(('sample', None)) for parent in self.parent_connections]
        samples = [parent.recv() for parent in self.parent_connections]
        return samples

    def step(self, actions):
        for action, parent in zip(actions, self.parent_connections):
            parent.send(('step', action))

        # [(state, key, time, reward, info, done)...]
        results = [parent.recv() for parent in self.parent_connections]
        return results

    def reset(self):
        [parent.send(('reset', None)) for parent in self.parent_connections]

        states = [parent.recv() for parent in self.parent_connections]
        return states

    def close(self):
        [parent.send(('close', None)) for parent in self.parent_connections]
        [parent.close() for parent in self.parent_connections]
