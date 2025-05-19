import numpy as np
import math

class MCTS():
    """
    This class handles the Monte Carlo Tree Search.
    """

    def __init__(self, game, nnet, args):
        """
        Initialize MCTS.
        Args:
            game: Game ThakInstance of the game class.
            nnet: Neural network instance to guide the search and evaluate states.
            args: Dictionary or argparse object with MCTS hyperparameters.
                  Expected keys:
                  num_mcts_sims (int): Number of MCTS simulations per move.
                  cpuct (float): Exploration constant for PUCT formula.
        """
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa = {}  # Stores Q values for s,a pairs (mean action value)
        self.Nsa = {}  # Stores N values for s,a pairs (edge visit count)
        self.Ns = {}   # Stores N values for s (board visit count)
        self.Ps = {}   # Stores P values for s (initial policy from nnet)

        self.Es = {}   # Stores game_ended_value for board s (1, -1, 1e-4, or 0)
        self.Vs = {}   # Stores valid_moves for board s

    def set_nnet(self, nnet):
        """
        Set the neural network for the MCTS.
        Args:
            nnet: The neural network instance.
        """
        self.nnet = nnet

    def reset_search_state(self):
        """
        Reset the internal search state (Q, N, P, E, V dictionaries).
        This should be called before starting a new search for a new root node (e.g., for a new move).
        """
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def getActionProb(self, canonical_board, temp=1):
        """
        Performs num_mcts_sims simulations starting from canonical_board to get action probabilities.
        Args:
            canonical_board (np.ndarray): The current board state from the perspective of the current player.
            temp (float): Temperature parameter. If temp=0, always picks the best move.
                          For temp > 0, samples from the distribution.
                          During self-play, temp is usually 1 for early moves, then decays to 0.
        Returns:
            list: A policy vector where the probability of an action is proportional to Nsa[(s,a)]**(1./temp).
        """
        num_sims = self.args.get('num_mcts_sims', 50)
        for i in range(num_sims):
            self.search(canonical_board, is_root_node=(i == 0)) # Pass is_root_node=True only for the first simulation

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0: # Should not happen if there are valid moves and sims are run
            # This can happen if all counts are 0, e.g. if a terminal state is passed right at the start.
            # Or if num_mcts_sims is 0.
            # Fallback to a uniform distribution over valid moves if needed or raise error.
            print("Warning: Sum of MCTS counts is zero. Using uniform distribution over valid moves.")
            valids = self.game.get_valid_moves(canonical_board)
            probs = valids / np.sum(valids) # Uniform over valid moves
            return list(probs)
            
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonical_board, is_root_node=False):
        """
        Performs one MCTS simulation: selection, expansion, simulation (via nnet), and backpropagation.
        Args:
            canonical_board (np.ndarray): The board state from the perspective of the current player.
            is_root_node (bool): True if this is the root of the current MCTS search for a move decision.
        Returns:
            float: The negative of the value of the current board state from the perspective of the current player.
                   (i.e., the value from the perspective of the opponent if it were their turn next from this state).
                   This is because the value v returned by the NNet is for the current player at canonical_board.
                   When backing up, if player P made a move to get to state S, the value of S for P is v(S).
                   The parent state S_parent from which P made the move, its Q(S_parent, move_to_S) should be updated with v(S).
                   If the search function is recursive, and it returns the value for the *next* player, then this negation is natural.
        """
        s = self.game.string_representation(canonical_board)

        # Check if game ended (leaf node)
        if s not in self.Es:
            # Pass current player (1 for canonical) and last move info if available/needed by game
            # For generic MCTS, assuming get_game_ended can work with just board and current player perspective
            self.Es[s] = self.game.get_game_ended(canonical_board, 1) 
        if self.Es[s] != 0: # Game ended
            return -self.Es[s] # Return value from opponent's perspective

        # Expansion: if node not visited before
        if s not in self.Ps:
            # Predict policy and value using the neural network
            policy, v = self.nnet.predict(canonical_board)

            if is_root_node and self.args.get('add_dirichlet_noise', False):
                alpha = self.args.get('dirichlet_alpha', 0.3)
                epsilon = self.args.get('epsilon_noise', 0.25)
                noise = np.random.dirichlet([alpha] * self.game.get_action_size())
                policy = (1 - epsilon) * np.array(policy).flatten() + epsilon * noise
                # Ensure policy is a flat array if it wasn't already from predict()

            self.Ps[s] = policy
            valids = self.game.get_valid_moves(canonical_board)
            self.Ps[s] = self.Ps[s] * valids  # Mask invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # Renormalize
            else:
                # This can happen if all valid moves have 0 probability from NNet (even after noise).
                # Or if there are no valid moves (terminal state, should have been caught by Es[s] != 0)
                print("Warning: All valid moves were masked after policy calculation (including noise), re-normalizing Ps to uniform over valids.")
                # Log the state or policy for debugging if this happens often.
                self.Ps[s] = valids # Fallback to uniform distribution over valid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s
                else:
                    # This case implies no valid moves, but Es[s] was 0. This is problematic.
                    print("Error: No valid moves and not a terminal state identified by Es[s], even after fallback in Ps calculation.")
                    self.Es[s] = 1e-7 # Mark as an error state / draw
                    return -self.Es[s]

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v # Return value from opponent's perspective

        # Selection: Choose action with highest UCB value
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # UCB calculation
        for a in range(self.game.get_action_size()):
            if valids[a]:
                q_sa = self.Qsa.get((s, a), 0)
                n_sa = self.Nsa.get((s, a), 0)
                u = q_sa + self.args.get('cpuct', 1.0) * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + n_sa)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        if a == -1: # Should not happen if there are valid moves
            print("Error: No valid action selected in MCTS search. This might mean no valid moves.")
            # This could happen if valids is all zeros, but Ps[s] was non-zero initially and then Es[s] was 0.
            # This path indicates a bug or an unhandled terminal state. Assign a value (e.g. draw) and return.
            # Or, if num_mcts_sims is very low and Ns[s] is 0 for the very first pass for a new node.
            # Ensure Ns[s] is initialized to at least 1 after expansion if sqrt(0) is an issue, but usually sqrt(0) is fine.
            # The issue is more likely if Ns[s] is 0, sqrt(Ns[s]) is 0, so U is just Q. If all Q are 0 or negative... 
            # For now, let's assume a draw if no action can be chosen.
            return 0 

        # next_s_board, next_player = self.game.get_next_state(canonical_board, 1, a) # player=1 for canonical_board, ignore move_row
        next_s_board, next_player, _ = self.game.get_next_state(canonical_board, 1, a) # player=1 for canonical_board, ignore move_row
        next_s_canonical = self.game.get_canonical_form(next_s_board, next_player)
        
        v = self.search(next_s_canonical, is_root_node=False) # Recursive call

        # Backpropagation
        # Update Qsa and Nsa for the (s,a) pair
        q_sa = self.Qsa.get((s, a), 0.0)
        n_sa = self.Nsa.get((s, a), 0)

        self.Qsa[(s, a)] = (n_sa * q_sa + v) / (n_sa + 1) # v is from next player's perspective, so this is correct
        self.Nsa[(s, a)] = n_sa + 1
        self.Ns[s] += 1
        return -v # Return value from opponent's perspective (current player of canonical_board) 