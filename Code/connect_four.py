import numpy as np
import gym
from gym import spaces

class ConnectFourEnv(gym.Env):
    """
    Custom Gym environment for Connect Four with reward shaping.

    - 0 represents an empty cell.
    - 1 represents player 1's piece.
    - -1 represents player 2's piece.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=np.int8)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(self.rows, self.columns),
            dtype=np.int8
        )

        # Game state
        self.current_player = 1
        self.done = False
        self.last_move = None

        # Reward shaping parameters
        self.window_weights = {
            4: 100.0,  # win
            3:   5.0,  # open three
            2:   2.0   # open two
        }
        self.opponent_weight = 1.0
        # Positional control matrix (center columns are more valuable)
        self.position_weights = np.array([
            [0,  0,  1,  2,  1,  0, 0],
            [0,  0,  2,  3,  2,  0, 0],
            [1,  2,  3,  4,  3,  2, 1],
            [1,  2,  3,  4,  3,  2, 1],
            [0,  0,  2,  3,  2,  0, 0],
            [0,  0,  1,  2,  1,  0, 0],
        ], dtype=np.float32)
        self.pos_weight_scale = 0.1
        # Scale for shaping term so it doesn't overpower terminal reward
        self.shaping_scale = 0.01

    def clone(self):
        """Create a copy of the current game state."""
        new_env = ConnectFourEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.last_move = self.last_move
        return new_env

    def reset(self):
        """Reset the game board and state."""
        self.board[:] = 0
        self.current_player = 1
        self.done = False
        self.last_move = None
        return self.board.copy()

    def step(self, action):
        """
        Apply the action for the current player.

        Returns:
            state (np.ndarray), reward (float), done (bool), info (dict)
        """
        # Invalid move check
        if not self.is_valid_move(action):
            return self.board.copy(), -10.0, True, {"invalid_move": True}

        # Heuristic before move
        prev_h = self._heuristic(self.board, self.current_player)

        # Drop piece
        row = self.get_next_open_row(action)
        self.board[row, action] = self.current_player
        self.last_move = (row, action)

        # Terminal reward
        if self.check_winner(row, action):
            reward = 1.0
            self.done = True
        elif np.all(self.board != 0):
            reward = 0.0
            self.done = True
        else:
            reward = 0.0
            self.done = False

        # Shaping: change in heuristic
        new_h = self._heuristic(self.board, self.current_player)
        reward += self.shaping_scale * (new_h - prev_h)

        # Switch player if not done
        if not self.done:
            self.current_player *= -1

        return self.board.copy(), reward, self.done, {}

    def is_valid_move(self, action):
        """Return True if the column is not full."""
        return self.board[0, action] == 0

    def get_next_open_row(self, action):
        """Return the lowest empty row index in the column."""
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, action] == 0:
                return r
        raise ValueError("Column is full")

    def check_winner(self, row, col):
        """Robustly check horizontal, vertical, and diagonal for a win."""
        player = self.board[row, col]
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # check forward
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.columns and self.board[r, c] == player:
                count += 1
                r += dr; c += dc
            # check backward
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.columns and self.board[r, c] == player:
                count += 1
                r -= dr; c -= dc
            if count >= 4:
                return True
        return False

    def _score_windows(self, board: np.ndarray, player: int) -> float:
        """Score all 4-cell windows for the given player."""
        score = 0.0
        # horizontal
        for r in range(self.rows):
            for c in range(self.columns - 3):
                window = board[r, c:c+4]
                if np.count_nonzero(window == -player) == 0:
                    cnt = np.count_nonzero(window == player)
                    score += self.window_weights.get(cnt, 0)
        # vertical
        for c in range(self.columns):
            for r in range(self.rows - 3):
                window = board[r:r+4, c]
                if np.count_nonzero(window == -player) == 0:
                    cnt = np.count_nonzero(window == player)
                    score += self.window_weights.get(cnt, 0)
        # diag down-right
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                window = [board[r+i, c+i] for i in range(4)]
                if window.count(-player) == 0:
                    cnt = window.count(player)
                    score += self.window_weights.get(cnt, 0)
        # diag up-right
        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                window = [board[r-i, c+i] for i in range(4)]
                if window.count(-player) == 0:
                    cnt = window.count(player)
                    score += self.window_weights.get(cnt, 0)
        return score

    def _heuristic(self, board: np.ndarray, player: int) -> float:
        """Combine window scores and positional control."""
        own = self._score_windows(board, player)
        opp = self._score_windows(board, -player)
        pos = np.sum((board == player) * self.position_weights)
        return (own - self.opponent_weight * opp) + self.pos_weight_scale * pos

    def render(self, mode='human'):
        """Print the board (flipped vertically)."""
        print(self.board)

# Simple random-play test when run directly
if __name__ == '__main__':
    env = ConnectFourEnv()
    state = env.reset()
    env.render()
    import random
    done = False
    while not done:
        action = random.choice([c for c in range(env.columns) if env.is_valid_move(c)])
        state, reward, done, info = env.step(action)
        print(f"Move: Player {3 - env.current_player} -> col {action}, reward={reward:.3f}")
        env.render()
        if done:
            if 'invalid_move' in info:
                print("Invalid move! Game over.")
            elif reward > 0:
                print(f"Player {3 - env.current_player} wins!")
            else:
                print("Draw.")