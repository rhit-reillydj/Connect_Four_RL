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
            # If it happens, return current board, player, and an invalid row indicator like -1
            return board, player, -1

        b = np.copy(board)
        move_row = -1
        for r in range(self.rows - 1, -1, -1): # Iterate from bottom row upwards
            if b[r, action] == 0:
                b[r, action] = player
                move_row = r
                break
        next_player = -player
        return b, next_player, move_row

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

    def _has_won(self, board, player, last_move_col=None, last_move_row=None):
        """
        Check if the given player has won. 
        If last_move_col and last_move_row are provided, it optimizes by checking around the last move.
        Otherwise, it checks the entire board.
        Args:
            board (np.ndarray): The current board state.
            player (int): The player to check for a win (1 or -1).
            last_move_col (int, optional): The column of the last move.
            last_move_row (int, optional): The row of the last move.
        Returns:
            bool: True if the player has won, False otherwise.
        """
        if last_move_col is None or last_move_row is None or last_move_row == -1: # Fallback to full board check
            # Check horizontal wins (full board)
            for r_idx in range(self.rows):
                for c_idx in range(self.cols - self.win_length + 1):
                    if np.all(board[r_idx, c_idx:c_idx+self.win_length] == player):
                        return True
            # Check vertical wins (full board)
            for c_idx in range(self.cols):
                for r_idx in range(self.rows - self.win_length + 1):
                    if np.all(board[r_idx:r_idx+self.win_length, c_idx] == player):
                        return True
            # Check positively sloped diagonals (full board)
            for r_idx in range(self.rows - self.win_length + 1):
                for c_idx in range(self.cols - self.win_length + 1):
                    if np.all([board[r_idx+i, c_idx+i] == player for i in range(self.win_length)]):
                        return True
            # Check negatively sloped diagonals (full board)
            for r_idx in range(self.win_length - 1, self.rows):
                for c_idx in range(self.cols - self.win_length + 1):
                    if np.all([board[r_idx-i, c_idx+i] == player for i in range(self.win_length)]):
                        return True
            return False

        # Optimized check around (last_move_row, last_move_col)
        r_move, c_move = last_move_row, last_move_col
        W = self.win_length

        # Horizontal check: Iterate through W possible start positions for a win
        for c_start in range(c_move - W + 1, c_move + 1):
            if 0 <= c_start <= self.cols - W:
                if np.all(board[r_move, c_start : c_start + W] == player):
                    return True

        # Vertical check: Iterate through W possible start positions for a win
        for r_start in range(r_move - W + 1, r_move + 1):
            if 0 <= r_start <= self.rows - W:
                if np.all(board[r_start : r_start + W, c_move] == player):
                    return True

        # Positively sloped diagonal (/) : Iterate through W possible start positions
        # (row and col both increase or both decrease from start_diag to end_diag)
        for i in range(W):
            r_start, c_start = r_move - i, c_move - i
            if (0 <= r_start <= self.rows - W) and \
               (0 <= c_start <= self.cols - W):
                if all(board[r_start + k, c_start + k] == player for k in range(W)):
                    return True

        # Negatively sloped diagonal (\\) : Iterate through W possible start positions
        # Elements are (r_start+k, c_start-k) for k in 0..W-1
        for i in range(W):
            r_start = r_move - i
            c_start = c_move + i
            # (r_start, c_start) is the top-rightmost point of a potential win \
            # if (r_move,c_move) is the i-th element from this top-right point\
            # Check bounds for the entire diagonal segment starting from (r_start, c_start)\
            # Start of segment: (r_start, c_start)\
            # End of segment: (r_start + W - 1, c_start - (W - 1))\
            condition_row_bounds = (0 <= r_start <= self.rows - W)
            condition_col_bounds = (W - 1 <= c_start < self.cols)

            if condition_row_bounds and condition_col_bounds:\
                # Col check for start (c_start) and full length left (c_start - W + 1 >=0)\
                if all(board[r_start + k, c_start - k] == player for k in range(W)):\
                    return True
        return False

    def get_game_ended(self, board, player, last_move_col=None, last_move_row=None):
        """
        Check if the game has ended.
        Args:
            board (np.ndarray): The current board state.
            player (int): The player who just made a move or whose turn it is.
                         The win is checked for this player.
            last_move_col (int, optional): The column of the last move. For optimized check.
            last_move_row (int, optional): The row of the last move. For optimized check.
        Returns:
            float: 1 if `player` won, -1 if `player` lost (opponent won),
                   a small positive float (e.g., 1e-4) for a draw, 
                   0 if the game is ongoing.
        """
        # If last move info is provided, use it for optimized win check
        if last_move_col is not None and last_move_row is not None:
            if self._has_won(board, player, last_move_col, last_move_row):
                return 1
            if self._has_won(board, -player, last_move_col, last_move_row): # Check opponent win based on this last move
                 # This scenario (opponent winning immediately after player's move) should not occur 
                 # if player's move was valid and _has_won is called for the player who just moved.
                 # However, keeping it for robustness, or if context of call changes.
                 # For connect four, opponent cannot win on your move.
                 pass # Opponent cannot win on player's move if game rules are standard
        else: # Fallback to full board check if last move info not available
            if self._has_won(board, player): # Check win for current player (who made the last move)
                return 1
            # No need to check opponent win with full board scan here, as they couldn't have won on player's move.

        # Check if opponent would have won BEFORE player's current move (this check is tricky here)
        # This is more about checking if the *previous* state led to an opponent win.
        # For the current structure, get_game_ended is called *after* a move for player.
        # So, we check if player won. Then we check if opponent *now* has a win (should be impossible). 
        # The original code had: if self._has_won(board, -player): return -1. This meant if the current board state
        # shows the opponent has won, player loses. This is fine.
        if self._has_won(board, -player): # If opponent has a winning line on the board now
            return -1
        
        if not np.any(board == 0): # Check for draw: if no empty spots left and no one has won
            return 1e-4 
            
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

    def display(self, board):
        """
        Displays the board in a human-readable format.
        X for player 1, O for player -1 (or computer), . for empty.
        Args:
            board (np.ndarray): The current board state.
        """
        print(" -----------------------") # Top border
        for r in range(self.rows):
            row_str = "| "
            for c in range(self.cols):
                if board[r, c] == 1:
                    row_str += "X "
                elif board[r, c] == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            print(row_str + "|")
        print(" -----------------------") # Bottom border
        # Print column numbers for easier play/debug
        col_nums = "| " + " ".join(map(str, range(self.cols))) + " |"
        print(col_nums)
        print("") # Extra newline for spacing 