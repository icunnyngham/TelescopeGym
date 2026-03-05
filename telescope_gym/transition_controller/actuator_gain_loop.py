from telescope_gym.transition_controller import TransitionController

class ActuatorGainLoop(TransitionController):
    def __init__(self, gain=1):
        self.gain = gain
    
    def step(self, state, actions):
        state.acts += self.gain*actions