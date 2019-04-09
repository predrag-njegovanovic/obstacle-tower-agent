import agent.definitions as definitions

from agent.trainer import Trainer
from agent.tower_agent import TowerAgent
from agent.experience_memory import ExperienceMemory
from agent.parallel_environment import ParallelEnvironment
from agent.utils import create_action_space

if __name__ == "__main__":
    config = definitions.network_params

    actions = create_action_space()
    action_size = len(actions)

    env = ParallelEnvironment(definitions.OBSTACLE_TOWER_PATH, definitions.NUM_ENVS)
    memory = ExperienceMemory(definitions.NUM_ENVS, definitions.OBSERVATION_SIZE)
    agent = TowerAgent(action_size,
                       config['first_layer'],
                       config['second_layer'],
                       config['conv_output'],
                       config['hidden_state_size'])

    env.start_parallel_execution()

    trainer = Trainer(env,
                      memory,
                      agent,
                      actions,
                      definitions.NUM_ENVS,
                      definitions.OBSERVATION_SIZE,
                      definitions.BATCH_SIZE,
                      definitions.EPOCHES,
                      definitions.TIMESTAMPS)

    trainer.train()
