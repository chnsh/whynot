import numpy as np
import itertools
from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.covid19 import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    scale_factor_val = action    
    actions = [[0.9,0.75,0.5,0.0],[0.1,0.25,0.5,0.75],[0.25,0.5,0.75,0.99]]
    actioncombos = list(itertools.product(*actions))
    action_to_intervention_map = {i:actioncombos[i] for i in range(len(actioncombos))}
    action_to_intervention_map[len(action_to_intervention_map)] = (Covid19Env.config.beta_scale_factor, \
        Covid19Env.config.proportion_hospitalized, Covid19Env.config.proportion_recovered_after_hospitalization)

    beta_scale_factor,proportion_hospitalized,proportion_recovered_after_hospitalization = action_to_intervention_map[scale_factor_val]
    return Intervention(
        time=time,
        beta_scale_factor=beta_scale_factor,
        proportion_hospitalized=proportion_hospitalized,
        proportion_recovered_after_hospitalization=proportion_recovered_after_hospitalization,
        proportion_dead_after_hospitalization= 1 - proportion_recovered_after_hospitalization,
        # proportion_recovered_without_hospitalization=proportion_recovered_without_hospitalization,
        # proportion_dead_without_hospitalization=1 - proportion_recovered_without_hospitalization
    )


def get_reward(intervention, state, time):
    """Compute the reward based on the observed state and choosen intervention."""
    cost = 100000 * state.deceased + 10000 * state.hospitalized + 100 * state.exposed + 1000 * state.infected - 1 * state.recovered
    discount = 4.0 / 365
    return -cost + np.exp(discount * time)


def observation_space():
    """Return observation space.
 
    The state is (susceptible, exposed, infected, recovered).
    """
    state_dim = State.num_variables()
    state_space_low = np.zeros(state_dim)
    state_space_high = np.inf * np.ones(state_dim)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


# def action_space():
#     """Return action space.

#     There are three control variables in the model:
#         - Set sigma
#         - Set beta
#         - Set mu

#     """
#     # return spaces.Tuple((spaces.Box(np.zeros(1), np.ones(3), dtype=np.int64),
#     #                      spaces.Box(np.zeros(3), np.ones(3), dtype=np.float64)))
#     return spaces.Box(np.zeros(1),3*np.ones(1),dtype=np.int64)
    


Covid19Env = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    # action_space=action_space(),
    action_space=spaces.Discrete(65),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="COVID19-v0", entry_point=Covid19Env, max_episode_steps=150, reward_threshold=1e10,
)
