import dataclasses

import numpy as np
from scipy.integrate import odeint

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameters for the simulation dynamics.

    Examples
    --------
    # Run the simulation for 200 days
    covid19.Config(duration=200)

    """

    # simulation parameters
    # exposed to infective parameter
    sigma: float = 0.2
    # susceptible to exposed parameter
    beta: float = 1.75
    # recovery parameter
    mu: float = 0.5  # on average 2.5 days to recover

    #: Simulation start time (in day)
    start_time: float = 0
    #: Simulation end time (in days)
    end_time: float = 400
    #: How frequently to measure simulator state
    delta_t: float = 1.0
    #: solver relative tolerance
    rtol: float = 1e-6
    #: solver absolute tolerance
    atol: float = 1e-6


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the COVID-19 simulator.

    The default state corresponds to an early infection state, defined by Adams
    et al. The early infection state is designed based on an unstable uninfected
    steady state by 1) adding one virus particle per ml of blood plasma, and 2)
    adding low levels of infected T-cells.
    """

    # pylint: disable-msg=invalid-name
    #: Number of susceptible
    susceptible: int = 9999
    #: Number of exposed
    exposed: int = 1
    #: Number of infected
    infected: int = 0
    #: Number of recovered
    recovered: int = 0


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the COVID-19 model.

    Examples
    --------
    >>> # Starting in step 100, set beta to 0.7 (leaving other variables unchanged)
    >>> Intervention(time=100, beta=0.7)

    """

    def __init__(self, time=100, **kwargs):
        """Specify an intervention in the dynamical system.

        Parameters
        ----------
            time: int
                Time of the intervention (days)
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def dynamics(state, time, config, intervention=None):
    """Update equations for the COVID-19 simulaton.

    Parameters
    ----------
        state:  np.ndarray, list, or tuple
            State of the dynamics
        time:   float
        config: covid19.Config
            Simulator configuration object that determines the coefficients
        intervantion: covid19.Intervention
            Simulator intervention object that determines when/how to update the
            dynamics.

    Returns
    -------
        ds_dt: list
            Derivative of the dynamics with respect to time

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    # pylint: disable-msg=invalid-name
    (
        susceptible,
        exposed,
        infected,
        recovered
    ) = state

    total_population = sum([susceptible,
                            exposed,
                            infected,
                            recovered])

    delta_susceptible = -config.beta * (susceptible * infected)
    delta_exposed = config.beta * (susceptible * infected) - config.sigma * exposed
    delta_infected = config.sigma * exposed - config.mu * infected
    delta_recovered = config.mu * infected

    ds_dt = [
        delta_susceptible, delta_exposed, delta_infected, delta_recovered
    ]
    return ds_dt


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the SEIR simulator model.

    The simulation starts at initial_state at time 0, and evolves the state
    using dynamics whose parameters are specified in config.

    Parameters
    ----------
        initial_state:  `whynot.simulators.covid19.State`
            Initial State object, which is used as x_{t_0} for the simulator.
        config:  `whynot.simulators.covid19.Config`
            Config object that encapsulates the parameters that define the dynamics.
        intervention: `whynot.simulators.covid19.Intervention`
            Intervention object that specifies what, if any, intervention to perform.
        seed: int
            Seed to set internal randomness. The simulator is deterministic, so
            the seed parameter is ignored.

    Returns
    -------
        run: `whynot.dynamics.Run`
            Rollout of the model.

    """
    # Simulator is deterministic, so seed is ignored
    # pylint: disable-msg=unused-argument
    t_eval = np.arange(
        config.start_time, config.end_time + config.delta_t, config.delta_t
    )

    solution = odeint(
        dynamics,
        y0=dataclasses.astuple(initial_state),
        t=t_eval,
        args=(config, intervention),
        rtol=config.rtol,
        atol=config.atol,
    )

    states = [initial_state] + [State(*state) for state in solution[1:]]
    return wn.dynamics.Run(states=states, times=t_eval)


if __name__ == "__main__":
    print(simulate(State(), Config()))
