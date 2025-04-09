# connect_four.py

import numpy as np
import gym
from gym import spaces

class ConnectFourEnv(gym.Env):
    """
    Custom Gym environment for Connect Four.
    The board is a 6x7 grid:
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
        # There are 7 possible actions (one per column)
        self.action_space = spaces.Discrete(self.columns)
        # The observation is the board state (6x7 grid with values 0, 1, or -1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows, self.columns), dtype=np.int8)
        
        self.current_player = 1
        self.done = False
        self.last_move = None
        
    def clone(self):
        """Create a copy of the current game state (more efficient than deepcopy)."""
        new_env = ConnectFourEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.last_move = self.last_move
        return new_env

    def reset(self):
        """Reset the game board and state."""
        self.board = np.zeros((self.rows, self.columns), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.last_move = None
        return self.board.copy()

    def step(self, action):
        """
        Apply the action for the current player.
        Returns:
            - new board state (np.array)
            - reward (float)
            - done (bool): whether the game has ended
            - info (dict): extra information (e.g., invalid move flag)
        """
        # Check for an invalid move (if the column is already full)
        if not self.is_valid_move(action):
            # Penalize invalid moves and end the game.
            return self.board.copy(), -10, True, {"invalid_move": True}

        # Get the row where the piece should fall
        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player
        self.last_move = (row, action)

        # Check if the move wins the game
        if self.check_winner(row, action):
            reward = 1
            self.done = True
        # Check for a draw (board is full)
        elif np.all(self.board != 0):
            reward = 0
            self.done = True
        else:
            reward = 0
            self.done = False

        # Prepare info dictionary (could include additional debugging info)
        info = {}
        # Switch player only if the game is not over
        if not self.done:
            self.current_player = -1 if self.current_player == 1 else 1

        return self.board.copy(), reward, self.done, info

    def is_valid_move(self, action):
        """Return True if the top cell in the column is empty."""
        return self.board[0][action] == 0

    def get_next_open_row(self, action):
        """Return the index of the next open row in the specified column."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][action] == 0:
                return row
        # This should not happen if is_valid_move() is called beforehand.
        raise Exception("Column is full")

    def check_winner(self, row, col):
        """
        Check whether the last move at (row, col) causes a win.
        The function checks horizontally, vertically, and on both diagonals.
        """
        player = self.board[row][col]

        # Horizontal check
        count = 0
        for c in range(max(0, col - 3), min(self.columns, col + 4)):
            if self.board[row][c] == player:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Vertical check
        count = 0
        for r in range(max(0, row - 3), min(self.rows, row + 4)):
            if self.board[r][col] == player:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Diagonal (positive slope) check
        count = 0
        for d in range(-3, 4):
            r = row + d
            c = col + d
            if 0 <= r < self.rows and 0 <= c < self.columns:
                if self.board[r][c] == player:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        # Diagonal (negative slope) check
        count = 0
        for d in range(-3, 4):
            r = row - d
            c = col + d
            if 0 <= r < self.rows and 0 <= c < self.columns:
                if self.board[r][c] == player:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        return False

    def render(self, mode='human'):
        """Print the board (flipped vertically for intuitive display)."""
        print(np.flip(self.board, 0))


# Simple test code to run the environment.
if __name__ == "__main__":
    env = ConnectFourEnv()
    state = env.reset()
    print("Initial Board:")
    env.render()

    # Run a simple random play test until the game is over.
    import random
    done = False
    while not done:
        valid_actions = [a for a in range(env.columns) if env.is_valid_move(a)]
        action = random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        print(f"\nPlayer {3 - env.current_player} played column {action}")
        env.render()
        if done:
            if reward == 1:
                print(f"\nPlayer {3 - env.current_player} wins!")
            elif "invalid_move" in info:
                print("\nInvalid move was made!")
            else:
                print("\nThe game ended in a draw.")