"""Complete the Agent class for CS 4341 Assignment 3."""
from __future__ import annotations

try:
    from .reinforcement_learning import BaseAgent, EpisodeConfig
    from .tag import Action, Player, TagState
except ImportError:
    from reinforcement_learning import BaseAgent, EpisodeConfig
    from tag import Action, Player, TagState


GROUP_NAME = "CAChE"


class Agent(BaseAgent):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Initialize your model and hyperparameters here. Use self.rng for
        # reproducible randomness. The supplied framework runs training.
        self.q_table: dict[tuple[int, Action], float] = {}
        self.alpha = 0.1
        self.gamma = 0.95

    def init_episode(self) -> EpisodeConfig:
        """Return EpisodeConfig(side_length=..., n_opponents=..., max_steps=...).

        All three fields are positive integers. n_opponents + 1 must fit in
        side_length ** 2 cells. max_steps counts total player turns.
        Reset episode history here and preserve your learned model.
        """
        raise NotImplementedError

    def encode_state(self, state: TagState, player: Player) -> int:
        """Convert the board from player's perspective to a stable Python int.

        Include the features your policy and model need. Keep this conversion
        free of side effects and retain encoded integers in your model and history.
        """

        # Get player's position and role
        player_row, player_col = state.position_of(player)
        is_it = state.tagged_player == player
        just_tagged = is_it and state.just_tagged # Affects current tagger's ability to tag

        # When It...
        if is_it:

            target_player = None
            nearest_distance = None

            for other in state.players:

                if other == player:
                    continue

                other_row, other_col = state.position_of(other)

                distance = (abs(other_row - player_row) + abs(other_col - player_col))

                # Focus on tagging nearest non-tagger
                if nearest_distance is None or distance < nearest_distance:

                    nearest_distance = distance
                    target_player = other

        # When not It...
        else:

            # Focus on avoiding tagger
            target_player = state.tagged_player

        # Calculate relative position of target player
        target_row, target_col = state.position_of(target_player)
        row_difference = target_row - player_row
        col_difference = target_col - player_col
        distance = abs(row_difference) + abs(col_difference)

        # Encode overall distance
        if distance <= 1:
            distance_bucket = 0 # next to
        elif distance == 2:
            distance_bucket = 1 # very close
        elif distance <= 4:
            distance_bucket = 2 # somewhat close
        else: 
            distance_bucket = 3 # far

        # Encode row direction
        if row_difference < 0:
            row_relation = 0 # north
        elif row_difference > 0:
            row_relation = 1 # south
        else:
            row_relation = 2 # same row

        # Encode column direction
        if col_difference < 0:
            col_relation = 0 # west
        elif col_difference > 0:
            col_relation = 1 # east
        else:
            col_relation = 2 # same column

        # Encode unavailable cells
        occupied_positions = set(state.positions)

        # Bit positions:
        #   bit 0 = north
        #   bit 1 = south
        #   bit 2 = west
        #   bit 3 = east

        neighbors = (
            (player_row - 1, player_col), # north
            (player_row + 1, player_col), # south
            (player_row, player_col - 1), # west
            (player_row, player_col + 1), # east
        )

        blocked_mask = 0

        for direction, (row, col) in enumerate(neighbors):

            outside_board = (
                row < 0
                or row >= state.n_rows
                or col < 0
                or col >= state.n_cols
            )

            occupied = (row, col) in occupied_positions

            if outside_board or occupied:
                blocked_mask |= (1 << direction)

        # Return data as one integer
        encoded = int(is_it)
        encoded = encoded * 2 + int(just_tagged)
        encoded = encoded * 4 + distance_bucket
        encoded = encoded * 3 + row_relation
        encoded = encoded * 3 + col_relation
        encoded = encoded * 16 + blocked_mask

        return int(encoded)

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
        current_q = self.q_table.get((state, action), 0.0)
        if terminal or not next_actions:
            target = reward
        else:
            max_next_q = max(self.q_table.get((next_state, next_action), 0.0) for next_action in next_actions)
            target = reward + self.gamma * max_next_q

        self.q_table[(state, action)] = (current_q + self.alpha * (target - current_q))