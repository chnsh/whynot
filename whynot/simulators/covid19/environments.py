import numpy as np

from whynot.gym import spaces
from whynot.gym.envs import ODEEnvBuilder, register
from whynot.simulators.covid19 import Config, Intervention, simulate, State


def get_intervention(action, time):
    """Return the intervention in the simulator required to take action."""
    sigma_scale_factor, beta_scale_factor, mu_scale_factor = action
    return Intervention(
        time=time,
        sigma_scale_factor=sigma_scale_factor,
        beta_scale_factor=beta_scale_factor,
        mu_scale_factor=mu_scale_factor,
    )


def get_reward(intervention, state, time):
    """Compute the reward based on the observed state and choosen intervention."""
    cost = state.infected - state.recovered
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


def action_space():
    """Return action space.

    There are three control variables in the model:
        - Set sigma
        - Set beta
        - Set mu

    """
    return spaces.Box(np.zeros(3), np.inf * np.ones(3), dtype=np.float64)


Covid19Env = ODEEnvBuilder(
    simulate_fn=simulate,
    config=Config(),
    initial_state=State(),
    action_space=action_space(),
    observation_space=observation_space(),
    timestep=1.0,
    intervention_fn=get_intervention,
    reward_fn=get_reward,
)

register(
    id="COVID19-v0", entry_point=Covid19Env, max_episode_steps=150, reward_threshold=1e10,
)
