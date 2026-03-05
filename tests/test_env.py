"""
Smoke tests for telescope_gym.

Requires hcipy to be installed. These tests verify that the environment
can be created, reset, and stepped through without errors.
"""

import pytest
import numpy as np

hcipy = pytest.importorskip("hcipy")

import gymnasium as gym
import telescope_gym
from telescope_gym.sim import MultiAperturePSFSampler
from telescope_gym import (
    ReshapeActuatorActions,
    NormalActSetter,
    ActuatorGainLoop,
    SingleFrameObs,
    ImprovePupilRMS,
    ImproveStrehl,
    TimeoutCondition,
    SingleFrameObsRender,
)


@pytest.fixture
def sampler():
    """Create a small, fast telescope sampler for testing."""
    n_mir = 6
    telescope_r = 1.0
    mir_coords = hcipy.SeparatedCoords(
        (np.array([telescope_r]), np.linspace(0, 2 * np.pi, n_mir + 1)[:-1])
    )
    mir_centers = hcipy.PolarGrid(mir_coords).as_('cartesian')

    mas_setup = {
        'mirror_config': {
            'pupil_res': 64,
            'piston_scale': 1e-6,
            'tip_tilt_scale': 1e-6,
            'positions': mir_centers,
            'aperture_config': ['circular', 0.5],
            'pupil_extent': 2.6,
        },
        'filter_configs': [{
            'central_lam': 1e-6,
            'focal_res': 32,
            'focal_extent': 4.0,
            'frac_bandwidth': 0.05,
            'num_samples': 1,
        }],
    }
    return MultiAperturePSFSampler(**mas_setup)


@pytest.fixture
def env(sampler):
    """Create a basic telescope gym environment."""
    _, y = sampler.sample()
    act_shape = y.shape

    env = gym.make(
        "TelescopeGym-v0",
        action_parser=ReshapeActuatorActions(act_shape=act_shape),
        state_setter=NormalActSetter(act_shape=act_shape, error_sigma=0.5),
        transition_controller=ActuatorGainLoop(gain=-1),
        obs_builder=SingleFrameObs(telescope_sampler=sampler),
        terminated_conditions=[],
        truncated_conditions=[TimeoutCondition(max_steps=10)],
        reward_function=ImprovePupilRMS(weight=1),
    )
    yield env
    env.close()


def test_env_reset(env):
    """Test that env.reset() returns valid observation and info."""
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape


def test_env_step(env):
    """Test that env.step() returns valid outputs."""
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_full_episode(env):
    """Test running a full episode until truncation."""
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps <= 10  # max_steps is 10


def test_env_multiple_resets(env):
    """Test that the environment can be reset multiple times."""
    for _ in range(3):
        obs, info = env.reset()
        assert obs is not None
        action = env.action_space.sample()
        env.step(action)


def test_sampler_produces_psf(sampler):
    """Test that the sampler generates valid PSF outputs."""
    x, y = sampler.sample()
    assert x.ndim == 3  # (res, res, samples)
    assert y.ndim == 2  # (n_mir, 3)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
