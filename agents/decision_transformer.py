from agents.abstractAgent import AbstractAgent
from environments.pytux import PyTux

'''
IDEAS
-----

- Gather dataset from multiple levels of AI
- Gather dataset from baseline with noise
- Train model and then generate augmented dataset and train more
'''

class TransformerController(AbstractAgent):
    def train(self):
        super().train()

    def act(self, state):
        super().act(state)
        return PyTux.Action(
            acceleration=0.7,
            brake=False,
            # steer=
            drift=False,
        )