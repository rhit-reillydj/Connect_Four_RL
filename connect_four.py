import numpy as np

class ConnectFourGame:
    def __init__(self, rows=6, cols=7, win_length=4):
        """
        Initialize the Connect Four game.
        Args:
            rows (int): Number of rows on the board.
            cols (int): Number of columns on the board.
            win_length (int): Number of pieces in a line to win.
        """
        self.rows = rows
        self.cols = cols
        self.win_length = win_length

    def get_initial_board(self):
        """
        Returns:
            np.ndarray: An empty board, represented by a 2D numpy array of zeros.
        """
        return np.zeros((self.rows, self.cols), dtype=int)

    def get_board_size(self):
        """
        Returns:
            tuple: (rows, cols)
        """
        return (self.rows, self.cols)

    def get_action_size(self):
        """
        Returns:
            int: Number of possible actions (equal to the number of columns).
        """
        return self.cols

    def get_next_state(self, board, player, action):
        """
        Place a piece on the board and return the new board and next player.
        Args:
            board (np.ndarray): The current board state.
            player (int): The current player (1 or -1).
            action (int): The column where the piece is to be dropped.
        Returns:
            tuple: (next_board, next_player)
        """
        if board[0, action] != 0: # Column is full
            # This case should ideally be prevented by checking valid moves first
            return board, player 

        b = np.copy(board)
        for r in range(self.rows - 1, -1, -1): # Iterate from bottom row upwards
            if b[r, action] == 0:
                b[r, action] = player
                break
        next_player = -player
        return b, next_player

    def get_valid_moves(self, board):
        """
        Returns a binary vector indicating valid moves (columns that are not full).
        Args:
            board (np.ndarray): The current board state.
        Returns:
            np.ndarray: A binary vector of size self.cols. 1 if move is valid, 0 otherwise.
        """
        valid_moves = np.zeros(self.cols, dtype=int)
        for c in range(self.cols):
            if board[0, c] == 0: # Check if the top row of the column is empty
                valid_moves[c] = 1
        return valid_moves

    def _has_won(self, board, player, last_move_col=None):
        """
        Check if the given player has won. 
        If last_move_col is provided, it can optimize by checking around the last move.
        For now, it checks the entire board.
        Args:
            board (np.ndarray): The current board state.
            player (int): The player to check for a win (1 or -1).
            last_move_col (int, optional): The column of the last move. Defaults to None.
        Returns:
            bool: True if the player has won, False otherwise.
        """
        # Find the row of the last move if last_move_col is provided
        last_move_row = -1
        if last_move_col is not None:
            for r in range(self.rows):
                if board[r, last_move_col] == player:
                    last_move_row = r
                    break
            # If the column is somehow empty for this player, something is wrong
            # or it's called in a state where last_move_col isn't actually that player's last move.
            # Fallback to full board check or handle error. For now, proceed. 
            if last_move_row == -1 and board[self.rows-1, last_move_col] != 0 : # if column is full and not player, this is an issue
                 pass # Let full board check handle it or refine this logic

        # Check horizontal wins
        for r in range(self.rows):
            for c in range(self.cols - self.win_length + 1):
                if np.all(board[r, c:c+self.win_length] == player):
                    return True

        # Check vertical wins
        for c in range(self.cols):
            for r in range(self.rows - self.win_length + 1):
                if np.all(board[r:r+self.win_length, c] == player):
                    return True

        # Check positively sloped diagonals (e.g., bottom-left to top-right)
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols - self.win_length + 1):
                if np.all([board[r+i, c+i] == player for i in range(self.win_length)]):
                    return True

        # Check negatively sloped diagonals (e.g., top-left to bottom-right)
        for r in range(self.win_length - 1, self.rows):
            for c in range(self.cols - self.win_length + 1):
                if np.all([board[r-i, c+i] == player for i in range(self.win_length)]):
                    return True
        
        return False

    def get_game_ended(self, board, player):
        """
        Check if the game has ended.
        Args:
            board (np.ndarray): The current board state.
            player (int): The player who just made a move or whose turn it is.
                         The win is checked for this player.
        Returns:
            float: 1 if `player` won, -1 if `player` lost (opponent won),
                   a small positive float (e.g., 1e-4) for a draw, 
                   0 if the game is ongoing.
        """
        # last_move_col = None # For now, _has_won checks the whole board
        # If we want to optimize, we'd need to know the actual last move's column.
        # The `player` argument here is the one whose perspective we evaluate the win from.

        if self._has_won(board, player):
            return 1
        if self._has_won(board, -player): # Check if opponent has won
            return -1
        
        # Check for draw: if no empty spots left and no one has won
        if not np.any(board == 0):
            return 1e-4 # Small value for draw
            
        return 0 # Game is ongoing

    def get_canonical_form(self, board, player):
        """
        Returns the board from the perspective of the given player.
        Player's pieces are 1, opponent's are -1, empty is 0.
        Args:
            board (np.ndarray): The current board state.
            player (int): The player for whom to get the canonical form (1 or -1).
        Returns:
            np.ndarray: The board in canonical form.
        """
        # player is 1 or -1
        # If player is 1, board remains as is (1 for player, -1 for opponent).
        # If player is -1, board is multiplied by -1 (so -1 becomes 1, 1 becomes -1).
        return board * player

    def get_symmetries(self, board, pi):
        """
        Get symmetrical versions of the board and policy for data augmentation.
        For Connect Four, this includes the original and a horizontally flipped version.
        Args:
            board (np.ndarray): The current board state.
            pi (np.ndarray): The policy vector (probabilities for each action).
        Returns:
            list: A list of (symmetric_board, symmetric_pi) tuples.
        """
        assert len(pi) == self.cols

        symmetries = []
        
        # Original
        symmetries.append((np.copy(board), np.copy(pi)))
        
        # Horizontal flip
        flipped_board = np.fliplr(board)
        flipped_pi = np.flip(pi) # flip the policy vector as well
        symmetries.append((flipped_board, flipped_pi))
        
        return symmetries

    def string_representation(self, board):
        """
        Get a string representation of the board, useful for MCTS dictionary keys or debugging.
        Args:
            board (np.ndarray): The current board state.
        Returns:
            str: A string representation of the board.
        """
        return board.tobytes() 