"""Complete the Agent class for CS 4341 Assignment 3."""
from __future__ import annotations

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

    def init_episode(self) -> EpisodeConfig:
        # Increase episode index
        self.episode_index += 1
        
        # Choose a random side length for the board that isn't too large
        side = self.rng.choice([4, 5, 6, 7])
        
        # Decide number of players based on the board size
        n_players = self.rng.randint(3, max(3, min(10, side * side // 3)))
        
        # Calculate max steps based on the number of players
        max_steps = 30 * n_players
        
        # Return resulting EpisodeConfig
        return EpisodeConfig(side_length=side, n_opponents=n_players - 1, max_steps=max_steps)




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
        # Determine if player was it before and after action
        was_it = state.tagged_player == player
        is_it = next_state.tagged_player == player
        
        # Determine changes in the player's status (how much they were it and how many tags they have)
        d_it = next_state.score_of(player).it_turns - state.score_of(player).it_turns
        d_tags = next_state.score_of(player).tags - state.score_of(player).tags
        
        # Reward tags, and penalize being tagged
        reward = 0.0
        reward -= d_it * 1.0
        reward += d_tags * 2.0
        if is_it and not was_it:
            reward -= 1.0
        
        
        # Function to calculate distance between player and nearest opponent
            
        
        
        
        
        
        # Distance considerations
        if was_it == is_it:
            reward += 0.3 * (self.consider_distance(next_state, player) - self.consider_distance(state, player))
        
        return reward
    
    
    def consider_distance(self, state, player):
        # Find current position of player
        me = state.position_of(player)
        scale = max(1, state.n_rows + state.n_cols - 2)
        
        # Function for calcuating distance between the current player and another player
        def dist(other):
            p = state.position_of(other)
            return abs(me[0] - p[0]) + abs(me[1] - p[1])
        
        if state.tagged_player == player:
            nearest_opponent = min(dist(other) for other in state.players if other != player)
            return -nearest_opponent / scale
        
        return dist(state.tagged_player)
        

    def select_action(
        self, state: int, legal_actions: tuple[Action, ...], training: bool,
    ) -> Action | None:
        """Choose from legal_actions using the encoded integer state.

        Use exploration when training=True and your evaluation policy when
        training=False. Return None when legal_actions is empty.
        """
        raise NotImplementedError

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
