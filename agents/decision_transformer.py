from agents.AbstractAgent import AbstractAgent
from environments.pytux import PyTux


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