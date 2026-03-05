# TelescopeGym

A [Gymnasium](https://gymnasium.farama.org) environment for training reinforcement learning agents to control telescope adaptive optics, built on [HCIPy](https://hcipy.org).

(Note: built on version 1.0 of [TelescopeSim](https://github.com/icunnyngham/TelescopeSim), which was my workhorse for supervised, deep learning FPWFS research for years.  Included inline because it may be updated later.)

## Overview

telescope_gym provides a modular, pluggable RL environment for telescope wavefront sensing and control problems. It wraps HCIPy optics simulations into Gymnasium's standard `reset`/`step` interface, enabling use with any RL framework.

**Key features:**
- Simulates multi-aperture telescope optics with piston-tip-tilt and deformable mirror control
- Modular component architecture: mix and match observation builders, action parsers, reward functions, and more
- Includes atmospheric turbulence simulation, photon noise, and broadband filters
- Built-in visualization/rendering for debugging and analysis
- Example CleanRL PPO training scripts for two control problems

## Installation

```bash
pip install -e .

# With example script dependencies (torch, tyro, tensorboard):
pip install -e ".[examples]"
```

**Requirements:** Python 3.9+, [HCIPy](https://hcipy.org)

## Quick Start

```python
import gymnasium as gym
import telescope_gym
from telescope_gym.sim import MultiAperturePSFSampler
from telescope_gym import (
    ReshapeActuatorActions, NormalActSetter, ActuatorGainLoop,
    SingleFrameObs, ImprovePupilRMS, TimeoutCondition,
)

# 1. Build a telescope simulator
import hcipy, numpy as np
mir_centers = hcipy.PolarGrid(
    hcipy.SeparatedCoords((np.array([1.25]), np.linspace(0, 2*np.pi, 16)[:-1]))
).as_('cartesian')

sampler = MultiAperturePSFSampler(
    mirror_config={
        'pupil_res': 128, 'piston_scale': 1e-6, 'tip_tilt_scale': 1e-6,
        'positions': mir_centers, 'aperture_config': ['circular', 0.5],
        'pupil_extent': 3.2,
    },
    filter_configs=[{
        'central_lam': 1e-6, 'focal_res': 64, 'focal_extent': 4.0,
        'frac_bandwidth': 0.05, 'num_samples': 3,
    }],
)

_, y = sampler.sample()

# 2. Create the environment
env = gym.make(
    "TelescopeGym-v0",
    action_parser=ReshapeActuatorActions(act_shape=y.shape),
    state_setter=NormalActSetter(act_shape=y.shape, error_sigma=0.5),
    transition_controller=ActuatorGainLoop(gain=-1),
    obs_builder=SingleFrameObs(telescope_sampler=sampler),
    terminated_conditions=[],
    truncated_conditions=[TimeoutCondition(max_steps=50)],
    reward_function=ImprovePupilRMS(weight=1),
)

# 3. Run episodes
obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

## Architecture

telescope_gym uses a **strategy/plugin pattern** where each aspect of the environment is a pluggable component:

| Component | Base Class | Purpose |
|-----------|-----------|---------|
| **State Container** | `StateContainer` | Holds environment state (actuator values, PSF, atmosphere) |
| **State Setter** | `StateSetter` | Initializes state on `reset()` |
| **Obs Builder** | `ObsBuilder` | Generates observations from state |
| **Action Parser** | `ActionParser` | Reshapes/masks agent actions |
| **Transition Controller** | `TransitionController` | Applies actions to state on `step()` |
| **Reward** | `Reward` | Computes scalar reward signal |
| **Done Condition** | `DoneCondition` | Determines episode termination/truncation |
| **Renderer** | `Renderer` | Matplotlib visualization |

### Built-in Components

**State Setters:** `NormalActSetter`, `AtmosActSetter`, `TokyoDriftSetter`, `LoadStateSetter`

**Obs Builders:** `SingleFrameObs` (single PSF), `TokyoDriftObs` (multi-frame history)

**Action Parsers:** `ActuatorActions`, `ReshapeActuatorActions`, `MaskedActuatorActions`

**Transition Controllers:** `ActuatorGainLoop` (integrator), `TokyoDriftTransition` (null-space)

**Rewards:** `ImprovePupilRMS`, `ImproveStrehl`, `ImproveActsConverge`, `ImproveDarkHole`

**Done Conditions:** `TimeoutCondition`, `ActRMS/PupilRMS/Strehl Converged/Diverged`

## Examples

See [examples/](examples/) for complete CleanRL PPO training scripts:

- **`atmos_ptt_ppo.py`** - Atmospheric wavefront correction with piston-tip-tilt control
- **`dark_hole_ppo.py`** - Dark hole digging (speckle suppression) with piston-only control

## Simulation Backend

The `telescope_gym.sim` subpackage bundles the telescope simulation code:

- `SimulateMultiApertureTelescope` - High-level simulator with atmosphere, noise, and CLI support
- `MultiAperturePSFSampler` - Core HCIPy-based PSF generation with PTT/DM control

Supports multi-aperture geometries (ELF, monolithic, Keck-like, custom), atmospheric turbulence (single/multi-layer), deformable mirrors, broadband filters, and detector noise.

## Acknowledgments

The environment's component architecture was inspired by [RLGym](https://github.com/lucas-emery/rocket-league-gym/).

## Author

Ian Cunnyngham, Institute for Astronomy - University of Hawai'i, 2024
