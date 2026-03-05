from telescope_gym.done_conditions import DoneCondition

class TimeoutCondition(DoneCondition):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps = 0
    
    def reset(self, initial_state):
        self.steps = 0

    def condition_met(self, current_state):
        self.steps += 1
        return self.steps >= self.max_steps