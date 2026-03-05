from gymnasium.envs.registration import register

register(
    id="TelescopeGym-v0",
    entry_point="telescope_gym.envs:TelescopeGymEnv",
)

# Core environment
from telescope_gym.envs import TelescopeGymEnv

# Simulation
from telescope_gym.sim import SimulateMultiApertureTelescope, MultiAperturePSFSampler

# State containers & setters
from telescope_gym.state_containers import ActuatorState, AtmosActState, TokyoDriftState
from telescope_gym.state_setters import NormalActSetter, AtmosActSetter, TokyoDriftSetter, LoadStateSetter

# Observation & Action
from telescope_gym.obs_builders import SingleFrameObs, TokyoDriftObs
from telescope_gym.action_parsers import ActuatorActions, ReshapeActuatorActions, MaskedActuatorActions

# Transition controllers
from telescope_gym.transition_controller import ActuatorGainLoop, TokyoDriftTransition

# Rewards
from telescope_gym.rewards import (
    ImprovePupilRMS, ImprovePupilRMSmodDistance, ImprovePupilRMSDistConverge,
    ImproveStrehl, ImproveActsConverge, ImproveDarkHole,
)

# Done conditions
from telescope_gym.done_conditions import (
    TimeoutCondition,
    ActRMSConvergedCondition, ActRMSDivergedCondition,
    PupilRMSConvergedCondition, PupilRMSDivergedCondition,
    StrehlConvergedCondition, StrehlDivergedCondition,
)

# Renderers
from telescope_gym.renderers import SingleFrameObsRender, TwoFrameObsRender, DarkHoleRender
