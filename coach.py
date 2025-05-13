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
        # self.mcts = MCTS(self.game, self.nnet, self.args) # Main MCTS not strictly needed if created fresh per move
        self.train_examples_history = deque([], maxlen=self.args.get('max_len_of_queue', 20000))
        self.skip_first_self_play = False

        if self.args.get('load_model', False):
            load_folder, load_file = self.args.get('load_folder_file', ('checkpoint', 'best.weights.h5'))
            model_file = os.path.join(load_folder, load_file)
            if os.path.exists(model_file):
                print(f"Loading model from {model_file}...")
                self.nnet.load_checkpoint(folder=load_folder, filename=load_file)
                # Try to load corresponding training examples if a convention is established
                # example_iter_to_load = self.args.get('load_examples_iter', 0) # Example arg
                # if example_iter_to_load > 0:
                #     self.load_train_examples(example_iter_to_load)
                # else: # Or load a default named history file
                loaded_hist = self.load_train_examples() # Tries to load default name
                if loaded_hist:
                    self.train_examples_history = loaded_hist
                    print(f"Loaded {len(self.train_examples_history)} training examples.")
                self.skip_first_self_play = True 
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
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temp = int(episode_step < self.args.get('temp_threshold', 15))
            
            # Create a new MCTS instance for each move decision to ensure a fresh tree search
            episode_mcts = MCTS(self.game, self.nnet, self.args)
            pi = episode_mcts.getActionProb(canonical_board, temp=temp)
            
            sym = self.game.get_symmetries(canonical_board, pi)
            for b_sym, p_sym in sym:
                train_examples.append([b_sym, current_player, p_sym, None]) 

            action = np.random.choice(len(pi), p=pi)
            # Ensure action is valid (MCTS should only return valid policies, but good to be safe if pi can be noisy)
            if not self.game.get_valid_moves(canonical_board)[action]:
                print(f"Warning: MCTS chose an invalid action {action} with pi {pi}. Board:\n{canonical_board}")
                # Fallback: choose a random valid action if MCTS fails (should be rare)
                valid_actions = np.where(self.game.get_valid_moves(canonical_board) == 1)[0]
                if len(valid_actions) == 0: # Should be a draw or error
                    print("Error: No valid moves left but game not ended by MCTS.")
                    game_result = self.game.get_game_ended(board, current_player)
                    # Proceed to reward assignment with this game_result
                    break # Exit while loop
                action = np.random.choice(valid_actions)
                
            board, next_player_val = self.game.get_next_state(board, current_player, action)
            game_result = self.game.get_game_ended(board, current_player) 

            if game_result != 0:
                final_examples = []
                for hist_board, hist_player, hist_pi, _ in train_examples:
                    if hist_player == current_player:
                        final_examples.append((hist_board, hist_pi, game_result))
                    else: 
                        final_examples.append((hist_board, hist_pi, -game_result))
                return final_examples
            current_player = next_player_val
        
        # This part is reached if loop exited due to no valid moves error (should be rare)
        # Assign rewards based on the game_result obtained before break
        final_examples = []
        if game_result !=0: # Ensure game_result was set
            for hist_board, hist_player, hist_pi, _ in train_examples:
                if hist_player == current_player:
                    final_examples.append((hist_board, hist_pi, game_result))
                else: 
                    final_examples.append((hist_board, hist_pi, -game_result))
        return final_examples # Might be empty if game loop had issues early

    def learn(self):
        for i in range(1, self.args.get('num_iters', 100) + 1):
            print(f'------ ITERATION {i} ------')
            if not self.skip_first_self_play or i > 1:
                print("Starting Self-Play Phase...")
                iteration_train_examples = deque([])
                num_eps_to_run = self.args.get('num_eps', 50)
                for eps in range(num_eps_to_run):
                    start_time = time.time()
                    new_examples = self.execute_episode()
                    if new_examples: # Only extend if new examples were generated
                        iteration_train_examples.extend(new_examples)
                    print(f"Self-Play Episode {eps+1}/{num_eps_to_run} completed in {time.time()-start_time:.2f}s. Examples: {len(new_examples if new_examples else [])}")
                
                self.train_examples_history.extend(iteration_train_examples)
                if i % self.args.get('save_examples_freq', 5) == 0: # Save examples every few iterations
                    self.save_train_examples(i)
            else:
                print("Skipping self-play for the first iteration as per config.")
                self.skip_first_self_play = False

            if not self.train_examples_history:
                print("No training examples available. Skipping training and arena phase.")
                continue

            print("Starting Training Phase...")
            # Save current nnet (which is the best nnet so far) to pnet
            # This effectively makes pnet the model from the start of this iteration
            checkpoint_folder = self.args.get('checkpoint', './temp/')
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            
            self.nnet.save_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5')
            self.pnet.load_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5')
            # pmcts = MCTS(self.game, self.pnet, self.args) # Not needed for player lambda def

            train_data = list(self.train_examples_history)
            random.shuffle(train_data)
            
            print(f"Training nnet on {len(train_data)} examples...")
            self.nnet.train(train_data) # nnet is trained in-place

            # nmcts = MCTS(self.game, self.nnet, self.args) # Not needed for player lambda def
            # self.nnet.save_checkpoint(folder=checkpoint_folder, filename='nnet_temp.h5') 

            print("Starting Arena Comparison Phase...")
            def nnet_player(board_state):
                m = MCTS(self.game, self.nnet, self.args)
                pi = m.getActionProb(board_state, temp=0)
                return np.argmax(pi)

            def pnet_player(board_state):
                m = MCTS(self.game, self.pnet, self.args)
                pi = m.getActionProb(board_state, temp=0)
                return np.argmax(pi)
            
            # display_fn = self.game.display if hasattr(self.game, 'display') else None
            display_fn = None 

            arena = Arena(nnet_player, pnet_player, self.game, display=display_fn)
            arena_compare_games = self.args.get('arena_compare', 20)
            print(f"Playing {arena_compare_games} games in Arena...")
            n_wins, p_wins, draws = arena.play_games(arena_compare_games, verbose=self.args.get('arena_verbose', False))
            print(f"ARENA RESULTS: NewNet wins: {n_wins}, PrevNet wins: {p_wins}, Draws: {draws}")

            total_played = n_wins + p_wins
            if total_played == 0: # Avoid division by zero if all games are draws
                win_rate = 0
            else:
                win_rate = float(n_wins) / total_played

            if win_rate > self.args.get('update_threshold', 0.55):
                print(f"ACCEPTING NEW MODEL (Win rate: {win_rate:.3f})")
                self.nnet.save_checkpoint(folder=checkpoint_folder, filename='best.weights.h5')
            else:
                print(f"REJECTING NEW MODEL (Win rate: {win_rate:.3f})")
                self.nnet.load_checkpoint(folder=checkpoint_folder, filename='pnet.weights.h5') # MODIFIED - Revert to pnet weights (previous best for this iter)
            print("------------------------")

    def save_train_examples(self, iteration):
        folder = self.args.get('checkpoint', './temp/')
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Save the whole deque
        filename = os.path.join(folder, "train_examples_history.pkl") 
        # filename_iter = os.path.join(folder, f"train_examples_iter_{iteration}.pkl") # Optionally save per iteration
        import pickle
        try:
            with open(filename, "wb+") as f:
                pickle.dump(self.train_examples_history, f)
            print(f"Saved {len(self.train_examples_history)} training examples to {filename}")
        except Exception as e:
            print(f"Error saving training examples: {e}")

    def load_train_examples(self, iteration=None): # iteration arg not used currently for general load
        folder = self.args.get('checkpoint', './temp/')
        example_file = os.path.join(folder, "train_examples_history.pkl")
        # if iteration is not None:
        #     example_file = os.path.join(folder, f"train_examples_iter_{iteration}.pkl")
        
        if os.path.exists(example_file):
            import pickle
            try:
                with open(example_file, "rb") as f:
                    loaded_deque = pickle.load(f)
                    # Ensure it's a deque with the correct maxlen from args
                    if not isinstance(loaded_deque, deque):
                        print("Warning: Loaded examples not a deque, converting.")
                        return deque(list(loaded_deque), maxlen=self.args.get('max_len_of_queue', 20000))
                    # If maxlen changed in args, re-create deque
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