import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TelescopeGymEnv(gym.Env):
    metadata = {"render_modes": ["notebook", "rgb_array"]}

    def __init__(
        self, 
        action_parser,
        state_setter,
        transition_controller,
        obs_builder,
        terminated_conditions,
        truncated_conditions,
        reward_function,
        renderer=None,
        render_mode=None, 
    ):
        self._action_parser = action_parser
        self._state_setter = state_setter
        self._transition_controller = transition_controller
        self._obs_builder = obs_builder
        self._terminated_conditions = terminated_conditions
        self._truncated_conditions = truncated_conditions
        self._reward_function = reward_function
        self.renderer = renderer
        self.render_mode = render_mode

        self.action_space = self._action_parser.get_action_space()

        self.observation_space = self._obs_builder.get_obs_space()

        self._state_setter.np_random = self.np_random
        self._cur_state = self._state_setter.build_state()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if self.renderer is not None:
            self.renderer.render_mode = self.render_mode
            self.renderer.start()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderer.render(self._cur_state)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._state_setter.reset(self._cur_state)

        for condition in self._truncated_conditions + self._terminated_conditions:
            condition.reset(self._cur_state)

        self._obs_builder.reset(self._cur_state)
        observation = self._obs_builder.build_obs(self._cur_state)
        
        self._reward_function.reset(self._cur_state)

        if self.renderer is not None:
            self.renderer.reset(self._cur_state)

        if self.render_mode in ["notebook"]:
            self.renderer.render(self._cur_state)

        info = self._get_info()

        return observation, info
    
    def step(self, action):
        parsed_action = self._action_parser.parse_actions(action)
        
        self._transition_controller.step(self._cur_state, parsed_action)

        observation = self._obs_builder.build_obs(self._cur_state)
        info = self._get_info()

        terminated = False
        for condition in self._terminated_conditions:
            if condition.condition_met(self._cur_state):
                terminated = True
                break
        truncated = False
        for condition in self._truncated_conditions:
            if condition.condition_met(self._cur_state):
                truncated = True
                break

        if terminated:
            # Don't use `get_final_reward` on truncated
            reward = self._reward_function.get_final_reward(self._cur_state)
        else:
            reward = self._reward_function.get_reward(self._cur_state)

        if self.render_mode in ["notebook"]:
            self.renderer.render(self._cur_state)

        return observation, reward, terminated, truncated, info
        
    def close(self):
        if self.renderer is not None:
            self.renderer.close()

