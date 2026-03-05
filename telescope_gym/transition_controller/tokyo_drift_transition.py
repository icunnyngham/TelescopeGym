from telescope_gym.transition_controller import TransitionController

class TokyoDriftTransition(TransitionController):
    def __init__(self):
        pass
    
    def step(self, state, actions):
        state.act_hist += [ actions ]
        state.acts -= actions