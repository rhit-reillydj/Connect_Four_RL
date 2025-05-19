import numpy as np
from collections import deque
import time
import os
import random # For shuffling examples
import multiprocessing # Added for parallel processing
from functools import partial # Added for passing args to worker
import signal # For more graceful handling
import sys    # For sys.exit() if needed

# Assuming MCTS, ConnectFourGame, ConnectFourNNet are in these files
from mcts import MCTS
from arena import Arena # Assuming Arena is in arena.py
from connect_four import ConnectFourGame # Added import
from model import ConnectFourNNet      # Added import

# Global holder for the shutdown event for worker processes
SHARED_SHUTDOWN_EVENT_HOLDER = {}

def init_worker_event(main_shutdown_event):
    """Initializer for pool workers to receive the shutdown event."""
    SHARED_SHUTDOWN_EVENT_HOLDER['event'] = main_shutdown_event
    print(f"[Worker {os.getpid()}] Initialized with shutdown event.")

# Worker function for parallel self-play. Must be at top-level for pickling.
def _execute_episode_worker(worker_args_tuple):
    """
    Worker function to execute a single episode of self-play.
    Args:
        worker_args_tuple: A tuple containing (game_class, nnet_class, coach_args, current_nnet_weights_path, episode_num)
                           NOTE: shutdown_event is now accessed globally via SHARED_SHUTDOWN_EVENT_HOLDER
    Returns:
        A list of training examples from the episode, or None if an error occurs.
    """
    # print(f"[SelfPlayWorker {os.getpid()}] Entered _execute_episode_worker. Args tuple is present: {worker_args_tuple is not None}") # DIAGNOSTIC PRINT ADDED
    game_class, nnet_class, coach_args, current_nnet_weights_path, episode_idx = worker_args_tuple
    shutdown_event_worker = SHARED_SHUTDOWN_EVENT_HOLDER.get('event')
    
    print(f"SP_W {os.getpid()}: Ep {episode_idx} starting")
    start_time_worker = time.time()

    game_instance = game_class() # Instantiate the game
    nnet_instance = nnet_class(game_instance, coach_args) # Instantiate the nnet
    
    try:
        # nnet_instance.load_checkpoint(filepath=current_nnet_weights_path)
        folder, filename = os.path.split(current_nnet_weights_path)
        nnet_instance.load_checkpoint(folder=folder, filename=filename)
    except Exception as e:
        print(f"SP_W {os.getpid()}: Ep {episode_idx} Error loading nnet weights from {current_nnet_weights_path}: {e}")
        return None # Or raise error

    mcts_instance = MCTS(game_instance, nnet_instance, coach_args)
    
    # --- Replicated logic from Coach.execute_episode --- 
    train_examples_episode = []
    board = game_instance.get_initial_board()
    current_player = 1
    episode_step = 0

    mcts_instance.set_nnet(nnet_instance) # Should be redundant if passed in constructor correctly
    mcts_instance.reset_search_state() # Reset MCTS state once at the start of the episode

    while True:
        if shutdown_event_worker and shutdown_event_worker.is_set():
            print(f"SP_W {os.getpid()}: Ep {episode_idx} Shutdown signal detected, aborting self-play episode.")
            return None # Or an empty list to indicate abortion

        episode_step += 1
        canonical_board = game_instance.get_canonical_form(board, current_player)
        temp = int(episode_step < coach_args.get('temp_threshold', 15))
        
        pi = mcts_instance.getActionProb(canonical_board, temp=temp)
        
        # Ensure pi is a valid probability distribution (e.g. sums to 1)
        # This can be an issue if MCTS returns all zeros due to some edge case
        if not np.any(pi) or np.sum(pi) == 0:
            print(f"SP_W {os.getpid()}: Ep {episode_idx} Warning: MCTS returned all zero policy. Board:\n{canonical_board}")
            valids = game_instance.get_valid_moves(canonical_board)
            if np.sum(valids) > 0:
                pi = valids / np.sum(valids) # Fallback to uniform random over valid moves
            else: # No valid moves, game should have ended
                print(f"SP_W {os.getpid()}: Ep {episode_idx} Error: No valid moves and zero policy. Game should be over.")
                # Attempt to get game result as is
                game_result = game_instance.get_game_ended(board, current_player)
                break
        elif abs(np.sum(pi) - 1.0) > 1e-6 : # Check if pi sums to 1
            # print(f"[Worker {os.getpid()}] Normalizing pi in worker for episode {episode_idx} as sum was {np.sum(pi)}.")
            pi = pi / np.sum(pi)

        sym = game_instance.get_symmetries(canonical_board, pi)
        for b_sym, p_sym in sym:
            train_examples_episode.append([b_sym, current_player, p_sym, None]) 

        try:
            action = np.random.choice(len(pi), p=pi)
        except ValueError as e:
             print(f"SP_W {os.getpid()}: Ep {episode_idx} Error choosing action with pi {pi} (sum: {np.sum(pi)}): {e}. Board:\n{canonical_board}")
             # Attempt to recover if possible, e.g. by choosing a valid random move
             valids = game_instance.get_valid_moves(canonical_board)
             if np.sum(valids) > 0:
                 action = np.random.choice(np.where(valids == 1)[0])
             else:
                 game_result = game_instance.get_game_ended(board, current_player)
                 break # No valid moves, break

        if not game_instance.get_valid_moves(canonical_board)[action]:
            # This should ideally not happen if pi is correctly masked by MCTS
            print(f"SP_W {os.getpid()}: Ep {episode_idx} Warning: MCTS chose invalid action {action} with pi {pi}. Board:\n{canonical_board}")
            valid_actions = np.where(game_instance.get_valid_moves(canonical_board) == 1)[0]
            if len(valid_actions) == 0: 
                print(f"SP_W {os.getpid()}: Ep {episode_idx} Error: No valid moves left but game not ended by MCTS.")
                game_result = game_instance.get_game_ended(board, current_player)
                break 
            action = np.random.choice(valid_actions)
            
        board, next_player_val, move_row = game_instance.get_next_state(board, current_player, action)
        # Pass last_move_col and last_move_row to get_game_ended if your game implementation needs it
        game_result = game_instance.get_game_ended(board, current_player, last_move_col=action, last_move_row=move_row)

        if game_result != 0:
            break # Game ended
        current_player = next_player_val
    
    # Assign results to training examples
    final_examples_for_episode = []
    if game_result !=0: 
        for hist_board, hist_player, hist_pi, _ in train_examples_episode:
            # Perspective of the player who made the move for that board state
            if hist_player == current_player: # Game ended on current_player's turn (they lost or drew)
                final_examples_for_episode.append((hist_board, hist_pi, game_result if game_result != hist_player else -game_result))
            else: # Game ended on opponent's turn (hist_player won or drew)
                final_examples_for_episode.append((hist_board, hist_pi, -game_result if game_result != hist_player else game_result))
    
    print(f"SP_W {os.getpid()}: Ep {episode_idx} finished. Examples: {len(final_examples_for_episode)}")
    return final_examples_for_episode

# Worker function for parallel Arena games. Must be at top-level for pickling.
def _play_arena_game_worker(worker_args_tuple):
    """
    Worker function to play a single Arena game.
    Args:
        worker_args_tuple: A tuple containing 
                           (game_class, nnet_class, coach_args, 
                            p1_nnet_weights_path, p2_nnet_weights_path, 
                            game_idx, verbose_arena)
                           NOTE: shutdown_event is now accessed globally via SHARED_SHUTDOWN_EVENT_HOLDER
    Returns:
        int: Result of the game from player1's perspective (1 if P1 won, -1 if P2 won, 0 for draw), 
             or None if shutdown was triggered.
    """
    game_class, nnet_class, coach_args, p1_weights_path, p2_weights_path, game_idx, verbose = worker_args_tuple
    shutdown_event_worker = SHARED_SHUTDOWN_EVENT_HOLDER.get('event')
    
    print(f"AW {os.getpid()}: Game {game_idx} starting")

    game_instance = game_class()
    
    # Player 1 (e.g., new nnet)
    nnet1 = nnet_class(game_instance, coach_args)
    try:
        # nnet1.load_checkpoint(filepath=p1_weights_path)
        folder, filename = os.path.split(p1_weights_path)
        nnet1.load_checkpoint(folder=folder, filename=filename)
    except Exception as e:
        print(f"AW {os.getpid()}: Game {game_idx} Error loading P1 nnet weights from {p1_weights_path}: {e}")
        return 0 # Default to draw or handle error appropriately
    mcts1 = MCTS(game_instance, nnet1, coach_args)
    player1_func = lambda board_state: np.argmax(mcts1.getActionProb(board_state, temp=0))

    # Player 2 (e.g., previous best nnet)
    nnet2 = nnet_class(game_instance, coach_args)
    try:
        # nnet2.load_checkpoint(filepath=p2_weights_path)
        folder, filename = os.path.split(p2_weights_path)
        nnet2.load_checkpoint(folder=folder, filename=filename)
    except Exception as e:
        print(f"AW {os.getpid()}: Game {game_idx} Error loading P2 nnet weights from {p2_weights_path}: {e}")
        return 0 # Default to draw or handle error appropriately
    mcts2 = MCTS(game_instance, nnet2, coach_args)
    player2_func = lambda board_state: np.argmax(mcts2.getActionProb(board_state, temp=0))

    # Arena game logic (simplified from Arena class)
    players = [player2_func, None, player1_func] # P1 is at index 2 (for current_player_idx=1), P2 at index 0 (for current_player_idx=-1)
    current_player_idx = 1 # Start with player 1
    board = game_instance.get_initial_board()
    it = 0
    
    # Ensure MCTS search state is reset for each player at the start of their turn in an arena game
    # This is typically handled by how MCTS is used by playerX_func, 
    # but explicit reset inside playerX_func might be safer if MCTS instances are reused across games (not the case here per game).
    # The playerX_func created above uses a fresh MCTS instance for each game implicitly.
    # If MCTS instances were shared, they would need mctsX.reset_search_state() before getActionProb.
    # The nnet_player/pnet_player in original Coach.learn did this explicitly.
    # For this worker, since mcts1 and mcts2 are local to this game, their state is fresh.
    # We need to ensure that getActionProb inside the player functions doesn't carry over state from previous calls *within the same game* if not desired for arena.
    # However, for Arena, it is standard to reset MCTS for *each move decision* to get an independent assessment.
    # Modifying playerX_func slightly:
    
    def get_player_action(mcts_player, board_state_canonical):
        mcts_player.reset_search_state() # Reset for each move in Arena for fair comparison
        pi = mcts_player.getActionProb(board_state_canonical, temp=0)
        return np.argmax(pi)

    player1_final_func = partial(get_player_action, mcts1)
    player2_final_func = partial(get_player_action, mcts2)
    players = [player2_final_func, None, player1_final_func]

    while game_instance.get_game_ended(board, current_player_idx) == 0:
        if shutdown_event_worker and shutdown_event_worker.is_set():
            print(f"AW {os.getpid()}: Game {game_idx} Shutdown signal detected, aborting arena game.")
            return None # Indicate game was aborted

        it += 1
        if verbose: print(f"AW {os.getpid()}: Game {game_idx}, Turn {it}, Player {current_player_idx}")
        
        canonical_board = game_instance.get_canonical_form(board, current_player_idx)
        action = players[current_player_idx + 1](canonical_board)

        valids = game_instance.get_valid_moves(canonical_board)
        if not valids[action]:
            print(f"AW {os.getpid()}: Game {game_idx} Action {action} is not valid! Valids: {valids}. Board:\n{canonical_board}")
            # This indicates an issue with the MCTS or player logic if it picks an invalid move.
            # Force a loss for the current player.
            return -current_player_idx # P1 loses if current_player_idx is 1, P2 loses if current_player_idx is -1
            
        board, new_current_player_idx, move_row = game_instance.get_next_state(board, current_player_idx, action)
        # Check game end using the player who just moved (current_player_idx) and their move (action, move_row)
        game_end_status = game_instance.get_game_ended(board, current_player_idx, last_move_col=action, last_move_row=move_row)
        
        if game_end_status != 0:
            result_p1_perspective = game_end_status * current_player_idx
            print(f"AW {os.getpid()}: Game {game_idx} ended. P1 Result: {result_p1_perspective}")
            if verbose: print(f"    (Verbose: Game {game_idx} ended with status {game_end_status} for player {current_player_idx})")
            return result_p1_perspective # Convert to P1's perspective
        
        current_player_idx = new_current_player_idx
        if it > game_instance.get_max_game_len(): # Safety break for excessively long games
            print(f"AW {os.getpid()}: Game {game_idx} exceeded max length. Declaring draw.")
            return 0

    # Fallback if loop terminates unexpectedly (should be caught by get_game_ended check)
    final_game_status = game_instance.get_game_ended(board, 1) # Check from P1's perspective
    print(f"AW {os.getpid()}: Game {game_idx} loop ended unexpectedly or by max length. Final result for P1 perspective: {final_game_status}")
    return final_game_status

class Coach():
    """
    This class executes the self-play + learning loop.
    """
    def __init__(self, game: ConnectFourGame, nnet: ConnectFourNNet, args, shutdown_event_main: multiprocessing.Event):
        """
        Initialize the Coach.
        Args:
            game: An instance of the game class (e.g., ConnectFourGame).
            nnet: An instance of the neural network class (e.g., ConnectFourNNet).
            args: Dictionary or argparse object with hyperparameters.
            shutdown_event_main: A multiprocessing.Event to signal shutdown.
        """
        self.game = game
        self.nnet = nnet
        self.shutdown_event = shutdown_event_main # Store the shutdown event
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
        self.mcts.reset_search_state() # Reset MCTS state once at the start of the episode

        while True:
            if self.shutdown_event and self.shutdown_event.is_set():
                print(f"Coach.execute_episode: Shutdown signal detected, aborting episode.")
                return None # Or an empty list

            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, current_player)
            temp = int(episode_step < self.args.get('temp_threshold', 15))
            
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
        current_nnet_weights_path = os.path.join(self.args.get('checkpoint', './src/temp_connect_four/'), 'current_selfplay_nnet.keras')
        # Paths for arena players
        pnet_arena_weights_path = os.path.join(self.args.get('checkpoint', './src/temp_connect_four/'), 'pnet_arena.keras')
        nnet_arena_weights_path = os.path.join(self.args.get('checkpoint', './src/temp_connect_four/'), 'nnet_arena.keras')

        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler_coach(signum, frame):
            print("\nSIGINT received by Coach main process. Setting shutdown event...", flush=True)
            self.shutdown_event.set()
            # Do not do more here; let the loops check the event.
            # Restoring handler immediately might allow multiple interrupts to bypass event logic.

        signal.signal(signal.SIGINT, sigint_handler_coach)

        try:
            for i in range(1, self.args.get('num_iters', 100) + 1):
                    if self.shutdown_event.is_set():
                        print(f"Coach: Iteration {i} loop start: Shutdown event already set. Breaking from learn loop.", flush=True)
                        break

                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ------ ITERATION {i} ------")
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Self-Play Phase...")
                    
                    current_folder, current_filename = os.path.split(current_nnet_weights_path)
                    self.nnet.save_checkpoint(folder=current_folder, filename=current_filename)

                    iteration_train_examples_collected = []
                    num_eps_to_run = self.args.get('num_eps', 50)
                    
                    num_parallel_workers = self.args.get('num_parallel_self_play_workers', os.cpu_count())
                    actual_workers = min(num_parallel_workers, num_eps_to_run, os.cpu_count() if os.cpu_count() else 1)
                    print(f"Running {num_eps_to_run} self-play episodes using {actual_workers} parallel worker(s)...")

                    worker_arg_list = [
                        (self.game.__class__, self.nnet.__class__, self.args, current_nnet_weights_path, eps_idx) 
                        for eps_idx in range(num_eps_to_run)
                    ]

                    if actual_workers > 1:
                        try:
                            # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Self-Play: Initializing multiprocessing pool with {actual_workers} workers...")
                            with multiprocessing.Pool(processes=actual_workers, initializer=init_worker_event, initargs=(self.shutdown_event,)) as pool:
                                async_results = pool.map_async(_execute_episode_worker, worker_arg_list)
                                
                                while not async_results.ready():
                                    try:
                                        async_results.wait(timeout=0.5)
                                    except multiprocessing.TimeoutError:
                                        pass 
                                    if self.shutdown_event.is_set():
                                        print("Coach: Shutdown signal detected during self-play result wait. Terminating pool...", flush=True)
                                        pool.terminate() 
                                        pool.join()      
                                        break 
                                
                                if self.shutdown_event.is_set():
                                    print("Coach: Self-play pool processing was aborted due to shutdown.", flush=True)
                                else:
                                    try:
                                        results = async_results.get(timeout=5.0) 
                                        for res_list in results:
                                            if res_list is not None: 
                                                iteration_train_examples_collected.extend(res_list)
                                        print(f"Self-Play Phase: Collected {len(iteration_train_examples_collected)} examples from {num_eps_to_run} episodes.")
                                    except multiprocessing.TimeoutError:
                                        print("Coach: Timeout waiting for self-play results from pool. Shutdown event may not have been processed by workers in time.", flush=True)
                                    except Exception as e:
                                        print(f"Error retrieving self-play results after pool: {e}", flush=True)
                        except Exception as e:
                            print(f"Error during parallel self-play setup or outer pool operations: {e}. Self-play examples may be incomplete.", flush=True)
                            if self.shutdown_event.is_set():
                                 print("Shutdown event was set during this exception.", flush=True)
                            iteration_train_examples_collected.clear()
                            for eps_num_seq in range(num_eps_to_run):
                                start_time_seq = time.time()
                                new_examples_seq = self.execute_episode()
                                if new_examples_seq: 
                                    iteration_train_examples_collected.extend(new_examples_seq)
                                print(f"Self-Play Episode {eps_num_seq+1}/{num_eps_to_run} (sequential fallback) completed in {time.time()-start_time_seq:.2f}s. Examples: {len(new_examples_seq if new_examples_seq else [])}")
                    else: 
                        print("Running self-play sequentially (1 worker).")
                        for eps_num_seq in range(num_eps_to_run):
                            if self.shutdown_event.is_set():
                                print("Coach: Shutdown during sequential self-play.", flush=True)
                                break
                            start_time_seq = time.time()
                            new_examples_seq = self.execute_episode()
                            if new_examples_seq: 
                                iteration_train_examples_collected.extend(new_examples_seq)
                            print(f"Self-Play Episode {eps_num_seq+1}/{num_eps_to_run} (sequential) completed in {time.time()-start_time_seq:.2f}s. Examples: {len(new_examples_seq if new_examples_seq else [])}")

                    self.train_examples_history.extend(iteration_train_examples_collected)
                    
                    if self.shutdown_event.is_set():
                        print(f"Coach: Iteration {i} post self-play: Shutdown event detected. Breaking from learn loop.", flush=True)
                        break

                    if i % self.args.get('save_examples_freq', 5) == 0: 
                        self.save_train_examples(i)

                    if not self.train_examples_history:
                        print("No training examples available. Skipping training and arena phase.")
                        continue

                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Training Phase...")
                    checkpoint_folder = self.args.get('checkpoint', './src/temp_connect_four/')
                    if not os.path.exists(checkpoint_folder):
                        os.makedirs(checkpoint_folder)
                
                    pnet_arena_folder, pnet_arena_filename = os.path.split(pnet_arena_weights_path)
                    self.nnet.save_checkpoint(folder=pnet_arena_folder, filename=pnet_arena_filename)
                    self.pnet.load_checkpoint(folder=pnet_arena_folder, filename=pnet_arena_filename)
                
                    train_data = list(self.train_examples_history)
                    random.shuffle(train_data)
                
                    print(f"Training nnet on {len(train_data)} examples...")
                    try:
                        self.nnet.train(train_data)
                    except KeyboardInterrupt:
                        print("\nCoach: KeyboardInterrupt during model training. Ensuring shutdown event is set.", flush=True)
                        self.shutdown_event.set()
                    
                    if self.shutdown_event.is_set():
                        print(f"Coach: Iteration {i} post training: Shutdown event detected. Breaking from learn loop.", flush=True)
                        break

                    # Save the newly trained nnet for the arena
                    nnet_arena_folder, nnet_arena_filename = os.path.split(nnet_arena_weights_path)
                    self.nnet.save_checkpoint(folder=nnet_arena_folder, filename=nnet_arena_filename)

                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Starting Arena Comparison Phase...")
                    arena_compare_games = self.args.get('arena_compare', 20)
                    arena_verbose = self.args.get('arena_verbose', False)
                    num_parallel_arena_workers = self.args.get('num_parallel_arena_workers', os.cpu_count())
                    actual_arena_workers = min(num_parallel_arena_workers, arena_compare_games, os.cpu_count() if os.cpu_count() else 1)
                    
                    print(f"Playing {arena_compare_games} games in Arena using {actual_arena_workers} parallel worker(s)...")

                    n_wins, p_wins, draws = 0, 0, 0
                    games_to_play_first_half = arena_compare_games // 2
                    games_to_play_second_half = arena_compare_games - games_to_play_first_half

                    arena_worker_arg_list_p1_vs_p2 = [
                        (self.game.__class__, self.nnet.__class__, self.args, 
                         nnet_arena_weights_path, pnet_arena_weights_path, game_idx, arena_verbose) 
                        for game_idx in range(games_to_play_first_half)
                    ]
                    arena_worker_arg_list_p2_vs_p1 = [
                        (self.game.__class__, self.nnet.__class__, self.args, 
                         pnet_arena_weights_path, nnet_arena_weights_path, game_idx + games_to_play_first_half, arena_verbose) 
                        for game_idx in range(games_to_play_second_half)
                    ]

                    game_results_p1_perspective = []

                    if actual_arena_workers > 1:
                        try:
                            # print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Arena: Initializing multiprocessing pool with {actual_arena_workers} workers...")
                            with multiprocessing.Pool(processes=actual_arena_workers, initializer=init_worker_event, initargs=(self.shutdown_event,)) as pool:
                                game_results_p1_perspective_async = []
                                if games_to_play_first_half > 0:
                                    async_res_p1_vs_p2 = pool.map_async(_play_arena_game_worker, arena_worker_arg_list_p1_vs_p2)
                                    game_results_p1_perspective_async.append(async_res_p1_vs_p2)
                                
                                if games_to_play_second_half > 0 and not self.shutdown_event.is_set():
                                    async_res_p2_vs_p1 = pool.map_async(_play_arena_game_worker, arena_worker_arg_list_p2_vs_p1)
                                    game_results_p1_perspective_async.append(async_res_p2_vs_p1)
                                
                                for async_res_group_idx, async_res in enumerate(game_results_p1_perspective_async):
                                    if self.shutdown_event.is_set(): break
                                    while not async_res.ready():
                                        try:
                                            async_res.wait(timeout=0.5)
                                        except multiprocessing.TimeoutError:
                                            pass
                                        if self.shutdown_event.is_set():
                                            print("Coach: Shutdown signal detected during arena result wait. Terminating pool...", flush=True)
                                            pool.terminate()
                                            pool.join()
                                            break 
                                    if self.shutdown_event.is_set(): break

                                    if not self.shutdown_event.is_set():
                                        try:
                                            raw_results = async_res.get(timeout=5.0)
                                            if async_res_group_idx == 0:
                                                game_results_p1_perspective.extend(r for r in raw_results if r is not None)
                                            else: 
                                                game_results_p1_perspective.extend([-res for res in raw_results if res is not None])
                                        except multiprocessing.TimeoutError:
                                            print("Coach: Timeout waiting for arena results from pool.", flush=True)
                                        except Exception as e:
                                             print(f"Error retrieving arena results: {e}", flush=True)
                        except Exception as e:
                            print(f"Error during parallel arena games: {e}. Arena results might be incomplete.", flush=True)
                    else: 
                        print("Running arena games sequentially (1 worker).")
                        for args_tuple in arena_worker_arg_list_p1_vs_p2:
                            if self.shutdown_event.is_set(): break
                            res = _play_arena_game_worker(args_tuple)
                            if res is not None: game_results_p1_perspective.append(res)
                        if not self.shutdown_event.is_set():
                            for args_tuple in arena_worker_arg_list_p2_vs_p1:
                                if self.shutdown_event.is_set(): break
                                raw_res = _play_arena_game_worker(args_tuple)
                                if raw_res is not None: game_results_p1_perspective.append(-raw_res)
                    
                    if self.shutdown_event.is_set():
                        print(f"Coach: Iteration {i} main loop end: Shutdown event detected. Breaking from learn loop.", flush=True)
                        break

                    for game_result in game_results_p1_perspective:
                        if game_result == 1: n_wins +=1
                        elif game_result == -1: p_wins +=1
                        else: draws +=1
                    
                    print(f"ARENA RESULTS: NewNet (nnet) wins: {n_wins}, PrevNet (pnet) wins: {p_wins}, Draws: {draws}")

                    total_played = n_wins + p_wins
                    if total_played == 0: 
                        win_rate = 0
                    else:
                        win_rate = float(n_wins) / total_played

                    if win_rate >= self.args.get('update_threshold', 0.50):
                        print(f"ACCEPTING NEW MODEL (Win rate: {win_rate:.3f})")
                        self.nnet.save_checkpoint(folder=checkpoint_folder, filename='best.keras')
                        self.mcts.set_nnet(self.nnet)
                    else:
                        print(f"REJECTING NEW MODEL (Win rate: {win_rate:.3f})")
                        pnet_arena_folder, pnet_arena_filename = os.path.split(pnet_arena_weights_path)
                        self.nnet.load_checkpoint(folder=pnet_arena_folder, filename=pnet_arena_filename)
                        self.mcts.set_nnet(self.nnet)
                    print("------------------------")

                    if self.shutdown_event.is_set():
                        print(f"Coach: Iteration {i} main loop end: Shutdown event detected. Breaking from learn loop.", flush=True)
                        break

            if self.shutdown_event.is_set():
                 print(f"Coach: Iteration main loop end outside for: Shutdown event detected. Breaking from learn loop.", flush=True)
        
        except KeyboardInterrupt: 
            print("\nCoach: KeyboardInterrupt caught in outer learn try-except. Ensuring shutdown event is set.", flush=True)
            self.shutdown_event.set()
        
        finally:
            print("Coach: learn() method finishing. Restoring original SIGINT handler.", flush=True)
            signal.signal(signal.SIGINT, original_sigint_handler) 
            if self.shutdown_event.is_set():
                print("Coach: learn() method terminated because shutdown signal was active at the end.", flush=True)

    def save_train_examples(self, iteration):
        folder = self.args.get('checkpoint', './src/temp_connect_four/')
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
        folder = self.args.get('checkpoint', './src/temp_connect_four/')
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
    #     'checkpoint': './src/temp_connect_four/',
    #     'load_model': False,
    #     'load_folder_file': ('./src/temp_connect_four/', 'best.h5'),
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