import numpy as np
from collections import deque
import time
import os
import random # For shuffling examples

# Assuming MCTS, ConnectFourGame, ConnectFourNNet are in these files
from mcts import MCTS
from arena import Arena # Assuming Arena is in arena.py
from connect_four import ConnectFourGame # Added import
from model import ConnectFourNNet      # Added import

class Coach():
    """
    This class executes the self-play + learning loop.
    """
    def __init__(self, game: ConnectFourGame, nnet: ConnectFourNNet, args):
        """
        Initialize the Coach.
        Args:
            game: An instance of the game class (e.g., ConnectFourGame).
            nnet: An instance of the neural network class (e.g., ConnectFourNNet).
            args: Dictionary or argparse object with hyperparameters.
                  Expected keys:
                  num_iters (int): Number of training iterations.
                  num_eps (int): Number of self-play games to generate per iteration.
                  temp_threshold (int): Number of moves after which temperature becomes 0.
                  update_threshold (float): Threshold for accepting new model in Arena.
                  max_len_of_queue (int): Maximum size of the training examples deque.
                  num_mcts_sims (int): Passed to MCTS.
                  arena_compare (int): Number of games to play in arena.
                  cpuct (float): Passed to MCTS.
                  checkpoint (str): Folder to save checkpoints.
                  load_model (bool): Whether to load a saved model.
                  load_folder_file (tuple): (folder, filename) for loading.
        """
        self.game = game
        self.nnet = nnet
        # Create a new instance of the nnet's class for pnet, using the same args
        self.pnet = self.nnet.__class__(self.game, args) 
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args) # Reusable MCTS instance
        self.train_examples_history = deque([], maxlen=self.args.get('max_len_of_queue', 20000))

        if self.args.get('load_model', False):
            load_folder, load_file = self.args.get('load_folder_file', ('checkpoint', 'best.weights.h5'))
            model_file = os.path.join(load_folder, load_file)
            if os.path.exists(model_file):
                print(f"Loading model from {model_file}...")
                self.nnet.load_checkpoint(folder=load_folder, filename=load_file)
                self.mcts.set_nnet(self.nnet) # Update MCTS with the loaded nnet
                loaded_hist = self.load_train_examples() 
                if loaded_hist:
                    self.train_examples_history = loaded_hist
                    print(f"Loaded {len(self.train_examples_history)} training examples.")
            else:
                print(f"No model found at {model_file}, starting from scratch.")
        else:
            print("Starting new model training from scratch.")

    def execute_episode(self):
        """
        Executes one episode of self-play, generating training examples.
        Returns:
            list: A list of training examples, where each example is:
                  (canonical_board, current_player, mcts_policy, game_result_from_current_player_perspective)
                  The MCTS policy is for the canonical_board.
                  The game_result is +1 if current_player eventually wins, -1 if loses, 0 for draw.
        """
        train_examples = []
        board = self.game.get_initial_board()
        current_player = 1
        episode_step = 0
        
        self.mcts.set_nnet(self.nnet) # Ensure MCTS is using the primary nnet for self-play

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temp = int(episode_step < self.args.get('temp_threshold', 15))
            
            self.mcts.reset_search_state() # Reset MCTS state for new move
            pi = self.mcts.getActionProb(canonical_board, temp=temp)
            
            sym = self.game.get_symmetries(canonical_board, pi)
            for b_sym, p_sym in sym:
                train_examples.append([b_sym, current_player, p_sym, None]) 

            action = np.random.choice(len(pi), p=pi)
            
            if not self.game.get_valid_moves(canonical_board)[action]:
                print(f"Warning: MCTS chose an invalid action {action} with pi {pi}. Board:\n{canonical_board}")
                valid_actions = np.where(self.game.get_valid_moves(canonical_board) == 1)[0]
                if len(valid_actions) == 0: 
                    print("Error: No valid moves left but game not ended by MCTS.")
                    game_result = self.game.get_game_ended(board, current_player)
                    break 
                action = np.random.choice(valid_actions)
                
            board, next_player_val, move_row = self.game.get_next_state(board, current_player, action)
            game_result = self.game.get_game_ended(board, current_player, last_move_col=action, last_move_row=move_row)

            if game_result != 0:
                final_examples = []
                for hist_board, hist_player, hist_pi, _ in train_examples:
                    if hist_player == current_player:
                        final_examples.append((hist_board, hist_pi, game_result))
                    else: 
                        final_examples.append((hist_board, hist_pi, -game_result))
                return final_examples
            current_player = next_player_val
        
        final_examples = []
        if game_result !=0: 
            for hist_board, hist_player, hist_pi, _ in train_examples:
                if hist_player == current_player:
                    final_examples.append((hist_board, hist_pi, game_result))
                else: 
                    final_examples.append((hist_board, hist_pi, -game_result))
        return final_examples

    def learn(self):
        for i in range(1, self.args.get('num_iters', 100) + 1):
            print(f'------ ITERATION {i} ------')
            print("Starting Self-Play Phase...")
            iteration_train_examples = deque([])
            num_eps_to_run = self.args.get('num_eps', 50)
            for eps in range(num_eps_to_run):
                start_time = time.time()
                # self.mcts.set_nnet(self.nnet) # Already set at start of execute_episode or after training
                new_examples = self.execute_episode()
                if new_examples: 
                    iteration_train_examples.extend(new_examples)
                print(f"Self-Play Episode {eps+1}/{num_eps_to_run} completed in {time.time()-start_time:.2f}s. Examples: {len(new_examples if new_examples else [])}")
            
            self.train_examples_history.extend(iteration_train_examples)
            if i % self.args.get('save_examples_freq', 5) == 0: 
                self.save_train_examples(i)
            

            if not self.train_examples_history:
                print("No training examples available. Skipping training and arena phase.")
                continue

            print("Starting Training Phase...")
            checkpoint_folder = self.args.get('checkpoint', './temp/')
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            
            # Save current nnet (which is the best nnet so far) to pnet.weights.h5 to serve as the challenger
            self.nnet.save_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5')
            # pnet loads these weights
            self.pnet.load_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5')
            
            train_data = list(self.train_examples_history)
            random.shuffle(train_data)
            
            print(f"Training nnet on {len(train_data)} examples...")
            self.nnet.train(train_data) # nnet is trained in-place
            self.mcts.set_nnet(self.nnet) # Update self-play MCTS with newly trained nnet

            print("Starting Arena Comparison Phase...")
            def nnet_player(board_state):
                self.mcts.set_nnet(self.nnet) # Use the new nnet
                self.mcts.reset_search_state()
                pi = self.mcts.getActionProb(board_state, temp=0)
                return np.argmax(pi)

            def pnet_player(board_state):
                self.mcts.set_nnet(self.pnet) # Use the challenger pnet
                self.mcts.reset_search_state()
                pi = self.mcts.getActionProb(board_state, temp=0)
                return np.argmax(pi)
            
            display_fn = self.game.display if hasattr(self.game, 'display') else None

            arena = Arena(nnet_player, pnet_player, self.game, display=display_fn)
            arena_compare_games = self.args.get('arena_compare', 20)
            print(f"Playing {arena_compare_games} games in Arena...")
            n_wins, p_wins, draws = arena.play_games(arena_compare_games, verbose=self.args.get('arena_verbose', False))
            print(f"ARENA RESULTS: NewNet wins: {n_wins}, PrevNet wins: {p_wins}, Draws: {draws}")

            total_played = n_wins + p_wins
            if total_played == 0: 
                win_rate = 0
            else:
                win_rate = float(n_wins) / total_played

            if win_rate >= self.args.get('update_threshold', 0.50):
                print(f"ACCEPTING NEW MODEL (Win rate: {win_rate:.3f})")
                self.nnet.save_checkpoint(folder=checkpoint_folder, filename='best.weights.h5')
                # self.mcts.set_nnet(self.nnet) already done after training
            else:
                print(f"REJECTING NEW MODEL (Win rate: {win_rate:.3f})")
                self.nnet.load_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5') 
                self.mcts.set_nnet(self.nnet) # Update MCTS with the reverted nnet
            print("------------------------")

    def save_train_examples(self, iteration):
        folder = self.args.get('checkpoint', './temp/')
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "train_examples_history.pkl") 
        import pickle
        try:
            with open(filename, "wb+") as f:
                pickle.dump(self.train_examples_history, f)
            print(f"Saved {len(self.train_examples_history)} training examples to {filename}")
        except Exception as e:
            print(f"Error saving training examples: {e}")

    def load_train_examples(self, iteration=None): 
        folder = self.args.get('checkpoint', './temp/')
        example_file = os.path.join(folder, "train_examples_history.pkl")
        
        if os.path.exists(example_file):
            import pickle
            try:
                with open(example_file, "rb") as f:
                    loaded_deque = pickle.load(f)
                    if not isinstance(loaded_deque, deque):
                        print("Warning: Loaded examples not a deque, converting.")
                        return deque(list(loaded_deque), maxlen=self.args.get('max_len_of_queue', 20000))
                    if loaded_deque.maxlen != self.args.get('max_len_of_queue', 20000):
                        print("Warning: Maxlen of loaded deque differs from args. Re-creating deque.")
                        return deque(list(loaded_deque), maxlen=self.args.get('max_len_of_queue', 20000))
                    return loaded_deque
            except Exception as e:
                print(f"Error loading training examples from {example_file}: {e}")
                return deque([], maxlen=self.args.get('max_len_of_queue', 20000))
        return deque([], maxlen=self.args.get('max_len_of_queue', 20000))

# Placeholder for a main execution script or function
if __name__ == '__main__':
    # This is where you would set up the game, nnet, args, and start the coach.
    # from connect_four import ConnectFourGame
    # from model import ConnectFourNNet

    # print("Example: Initializing and running the Coach")
    # game = ConnectFourGame()
    # args_dict = {
    #     'num_iters': 3,
    #     'num_eps': 5, # Number of self-play games per iteration
    #     'temp_threshold': 15,
    #     'update_threshold': 0.55, # Win rate needed to accept new model
    #     'max_len_of_queue': 200000,
    #     'num_mcts_sims': 25, # For ConnectFour, can be lower than Go
    #     'arena_compare': 10, # Number of games for comparison
    #     'cpuct': 1.0,
    #     'checkpoint': './temp_connect_four/',
    #     'load_model': False,
    #     'load_folder_file': ('./temp_connect_four/', 'best.h5'),
    #     'lr': 0.001,
    #     'epochs': 10,
    #     'batch_size': 64,
    #     'save_examples_freq': 1, # How often to save the examples deque
    #     'arena_verbose': False # Print arena game details
    # }
    # # Convert dict to a Namespace-like object if your classes expect attribute access
    # from argparse import Namespace
    # args_ns = Namespace(**args_dict)

    # nnet = ConnectFourNNet(game, args_ns)
    # coach = Coach(game, nnet, args_ns)
    # coach.learn()
    pass 