import numpy as np
import time

# Assuming MCTS is in mcts.py
# from mcts import MCTS # Not directly used by Arena, but players might use it.

class Arena():
    """
    An Arena class where two agents (players) compete against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Args:
            player1: Function that takes board and returns action.
            player2: Function that takes board and returns action.
            game: ThakInstance of the game class.
            display: Function that takes board and prints it (optional).
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game between two players.
        Args:
            verbose (bool): If true, prints board at each step.
        Returns:
            int: Returns 1 if player1 won, -1 if player2 won, 0 if draw.
        """
        players = [self.player2, None, self.player1] # player 1 is 1, player 2 is -1
        current_player_idx = 1 # Start with player 1
        board = self.game.get_initial_board()
        it = 0
        while self.game.get_game_ended(board, current_player_idx) == 0:
            it += 1
            if verbose:
                assert self.display
                print(f"Turn {it}, Player {current_player_idx}")
                self.display(board)
            
            canonical_board = self.game.get_canonical_form(board, current_player_idx)
            action = players[current_player_idx + 1](canonical_board) # Get action from current player

            valids = self.game.get_valid_moves(canonical_board)
            if valids[action] == 0:
                print(f"Action {action} is not valid!")
                print(f"Valids: {valids}")
                # This should not happen if players are implemented correctly (e.g. MCTS respects valid moves)
                # For robustness, could assign a loss to the current player or end game.
                assert valids[action] > 0
            
            board, current_player_idx = self.game.get_next_state(board, current_player_idx, action)
        
        if verbose:
            assert self.display
            print(f"Game over: Turn {it}, Result for Player 1: {self.game.get_game_ended(board, 1)}")
            self.display(board)
        
        return self.game.get_game_ended(board, 1) # Return result from player 1's perspective

    def play_games(self, num_games, verbose=False):
        """
        Plays num_games games and returns the number of wins for player1, player2, and draws.
        Args:
            num_games (int): The number of games to play.
            verbose (bool): If true, prints results of each game.
        Returns:
            tuple: (one_won, two_won, draws)
        """
        one_won = 0
        two_won = 0
        draws = 0
        
        num_games_half = num_games // 2

        for i in range(num_games_half):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            if verbose:
                print(f"Game {i+1}/{num_games}: P1 vs P2. Result for P1: {game_result}. Score: P1 {one_won} - P2 {two_won} - Draws {draws}")

        # Swap players for the second half of the games
        self.player1, self.player2 = self.player2, self.player1

        for i in range(num_games - num_games_half):
            game_result = self.play_game(verbose=verbose) # game_result is still from original player1's perspective
            if game_result == -1: # If original P1 (now P2) lost, it means original P2 (now P1) won
                one_won +=1 # This counts for original P1
            elif game_result == 1:
                two_won +=1 # This counts for original P2
            else:
                draws +=1
            if verbose:
                 print(f"Game {i+1+num_games_half}/{num_games}: P2 vs P1. Result for P1 (now P2): {game_result}. Score: P1 {one_won} - P2 {two_won} - Draws {draws}")

        return one_won, two_won, draws 