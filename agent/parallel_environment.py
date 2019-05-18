import torch

from multiprocessing import Process, Pipe
from obstacle_tower_env import ObstacleTowerEnv

from agent.utils import prepare_state


def start_environment(connection, worker_id, env_path, retro, realtime_mode):
    obstacle_tower = ObstacleTowerEnv(
        env_path, worker_id=worker_id, retro=retro, timeout_wait=90, realtime_mode=False
    )
    obstacle_tower.reset()
    while True:
        command, action = connection.recv()
        if command == "sample":
            connection.send(obstacle_tower.action_space.sample())
        if command == "step":
            cumulative_reward = 0

            # frame skipping
            for i in range(6):
                observation, reward, done, info = obstacle_tower.step(action)
                state, keys, time = observation

                if reward > 0:
                    reward = 1

                cumulative_reward += reward

                if done:
                    break

            connection.send(
                (prepare_state(state).tolist(), keys, time, cumulative_reward, done)
            )
        elif command == "reset":
            state, keys, time = obstacle_tower.reset()
            connection.send((prepare_state(state).tolist(), keys, time))
        elif command == "close":
            connection.close()


class ParallelEnvironment:
    def __init__(self, env_path, num_of_processes, retro=False, realtime_mode=False):
        self.parent_connections, self.child_connections = zip(
            *[Pipe() for _ in range(num_of_processes)]
        )
        self.env_path = env_path
        self.retro = retro
        self.realtime_mode = realtime_mode
        self.processes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def start_parallel_execution(self):
        self.processes = [
            Process(
                target=start_environment,
                args=(child, worker_id, self.env_path, self.retro, self.realtime_mode),
                daemon=True,
            )
            for worker_id, child in enumerate(self.child_connections)
        ]

        for process in self.processes:
            process.start()

    def sample(self):
        [parent.send(("sample", None)) for parent in self.parent_connections]
        samples = [parent.recv() for parent in self.parent_connections]
        return samples

    def step(self, actions):
        for action, parent in zip(actions, self.parent_connections):
            parent.send(("step", action))

        # zip(*[(state, key, time, reward, done)...])
        state, key, time, reward, done = zip(
            *[parent.recv() for parent in self.parent_connections]
        )

        state_tensor = torch.Tensor(state).to(self.device)
        key_tensor = torch.Tensor(key)
        time_tensor = torch.Tensor(time)
        reward_tensor = torch.Tensor(reward)
        done_tensor = torch.Tensor(done)

        return state_tensor, key_tensor, time_tensor, reward_tensor, done_tensor

    def reset(self):
        [parent.send(("reset", None)) for parent in self.parent_connections]

        states, key, time = zip(*[parent.recv() for parent in self.parent_connections])

        state_tensor = torch.Tensor(states).to(self.device)
        key_tensor = torch.Tensor(key)
        time_tensor = torch.Tensor(time)

        return state_tensor, key_tensor, time_tensor

    def close(self):
        [parent.send(("close", None)) for parent in self.parent_connections]
        [parent.close() for parent in self.parent_connections]
