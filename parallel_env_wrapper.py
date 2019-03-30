from multiprocessing import Process, Pipe
from obstacle_tower_env import ObstacleTowerEnv


def start_environment(connection, env_path, retro, realtime_mode):
    obstacle_tower = ObstacleTowerEnv(env_path, retro=retro, realtime_mode=realtime_mode)

    while True:
        command, data = connection.recv()
        if command == 'step':
            state, reward, done, info = obstacle_tower.step(data)
            if done:
                state = obstacle_tower.reset()
            connection.send((state, reward, info))
        elif command == 'reset':
            state = obstacle_tower.reset()
            connection.send(state)
        elif command == 'close':
            connection.close()


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
                                  args=(child, self.env_path,
                                        self.retro, self.realtime_mode),
                                  daemon=True) for child in self.child_connection]

        for process in self.processes:
            process.start()

    def step(self, actions):
        for action, parent in zip(actions, self.parent_connections):
            parent.send(('step', action))

        results = [parent.recv() for parent in self.parent_connections]
        return results

    def reset(self):
        [parent.send(('reset', None)) for parent in self.parent_connections]

        states = [parent.recv() for parent in self.parent_connections]
        return states

    def close(self):
        [parent.send(('close', None)) for parent in self.parent_connections]
        [parent.close() for parent in self.parent_connections]
