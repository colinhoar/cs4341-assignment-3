"""Complete the Agent class for CS 4341 Assignment 3."""
from __future__ import annotations
from collections import defaultdict

try:
    from .reinforcement_learning import BaseAgent, EpisodeConfig
    from .tag import Action, Player, TagState
except ImportError:
    from reinforcement_learning import BaseAgent, EpisodeConfig
    from tag import Action, Player, TagState


GROUP_NAME = "replace-with-your-group-name"


class Agent(BaseAgent):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Initialize your model and hyperparameters here. Use self.rng for
        # reproducible randomness. The supplied framework runs training.

        #how much the q-table is updated
        self.lr = 0.1
        #future vs immediate reward
        self.gamma = 0.95
        #chance of picking a random action vs learned
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.q_table = defaultdict(lambda: defaultdict(float))

    def init_episode(self) -> EpisodeConfig:
        """Return EpisodeConfig(side_length=..., n_opponents=..., max_steps=...).

        All three fields are positive integers. n_opponents + 1 must fit in
        side_length ** 2 cells. max_steps counts total player turns.
        Reset episode history here and preserve your learned model.
        """
        #chance of picking a random action vs learned
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        side_length = self.rng.randint(3, 5)
        max_opponents = (side_length ** 2) -1
        n_opponents = self.rng.randint(1, max_opponents)
        max_steps = 100
        
        return EpisodeConfig(side_length=side_length, n_opponents=n_opponents, max_steps=max_steps)

    def encode_state(self, state: TagState, player: Player) -> int:
        """Convert the board from player's perspective to a stable Python int.

        Include the features your policy and model need. Keep this conversion
        free of side effects and retain encoded integers in your model and history.
        """
        raise NotImplementedError

    def calculate_reward(
        self,
        state: TagState,
        action: Action,
        next_state: TagState,
        player: Player,
        terminal: bool,
    ) -> float:
        """Return a finite float using the actual before-and-after TagState objects.

        The interval spans player's action through its next turn or game end,
        including other players' moves. terminal says whether next_state ends
        the episode. Keep state objects local to this reward calculation.
        """
        raise NotImplementedError

    def select_action(
        self, state: int, legal_actions: tuple[Action, ...], training: bool,
    ) -> Action | None:
        """Choose from legal_actions using the encoded integer state.

        Use exploration when training=True and your evaluation policy when
        training=False. Return None when legal_actions is empty.
        """
        if not legal_actions:
            return None

        #random action
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        #learned action 
        q_values = self.q_table[state]
        if not q_values:
            return self.rng.choice(legal_actions)
            
        return max(legal_actions, key=lambda a: q_values.get(a, 0.0))

    def update_model(
        self,
        state: int,
        action: Action,
        reward: float,
        next_state: int,
        next_actions: tuple[Action, ...],
        player: Player,
        terminal: bool,
    ) -> None:
        """Update self from encoded states and the supplied reward; return None.

        next_actions contains this player's next legal choices. When terminal
        is True, it is empty and the transition has zero future action value.
        """
        raise NotImplementedError