# ReinforcementLearning.py

import os
import copy
import random
import math
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count

from connect_four import ConnectFourEnv
from CNNModel import ConnectFourCNN

torch.backends.cudnn.benchmark = True  # Optimizes CNN ops

# ===============================
# Enhanced Helper Functions
# ===============================
def clone_env(env):
    """Optimized environment cloning"""
    return env.clone()

# In ReinforcementLearning.py
def state_to_tensor(env):
    """Simplified 3-channel state representation"""
    board = env.board.copy()
    current_player = env.current_player
    opponent = -1 if current_player == 1 else 1
    
    return torch.tensor(np.stack([
        (board == current_player).astype(np.float32),  # Channel 0: current player
        (board == opponent).astype(np.float32),        # Channel 1: opponent
        np.zeros_like(board, dtype=np.float32)         # Channel 2: empty
    ], axis=0)).unsqueeze(0)

def augment_data(state, pi):
    """
    Augment data by horizontal flip:
      - state: numpy array of shape [2, 6, 7]
      - pi: target policy vector of length 7 (one probability per column)
    Returns both the original and the horizontally flipped version.
    """
    # Flip the board along the horizontal axis (flip columns)
    state_flipped = np.flip(state, axis=2)
    pi_flipped = np.flip(pi)
    return [(state, pi), (state_flipped, pi_flipped)]

# ===============================
# MCTS Components
# ===============================
class MCTSNode:
    def __init__(self, env, parent=None, prior=0.0, action_taken=None):
        self.env = env
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior
        self.action_taken = action_taken
        self.is_terminal = env.done
    
    @property
    def q_value(self):
        return self.total_value / self.visits if self.visits else 0

class MCTS:
    def __init__(self, network, num_simulations=100, c_puct=1.0, device='cpu'):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.node_cache = {}

    def search(self, root):
        # Collect all leaf nodes for batch evaluation
        leaf_nodes = []
        search_paths = []
        
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            
            while node.children and not node.is_terminal:
                action, node = self.select_child(node)
                path.append(node)
            
            if node not in self.node_cache:
                leaf_nodes.append(node)
            search_paths.append((path, node))
        
        # Batch evaluate all unique leaf nodes
        if leaf_nodes:
            policies, values = self.evaluate_batch(leaf_nodes)
            for node, policy, value in zip(leaf_nodes, policies, values):
                self.node_cache[node] = (policy, value)
        
        # Backpropagate results
        for path, leaf_node in search_paths:
            policy, value = self.node_cache.get(leaf_node, (None, 0))
            if policy is not None:
                self.expand_node(leaf_node, policy)
            self.backpropagate(path, value)
        
        # Return final policy
        counts = np.zeros(root.env.columns)
        for action, child in root.children.items():
            counts[action] = child.visits
        return counts / counts.sum() if counts.sum() > 0 else np.ones(7)/7

    def evaluate_batch(self, nodes):
        states = [state_to_tensor(n.env) for n in nodes]
        states = torch.cat(states).to(self.device)
        
        with torch.no_grad():
            policy_logits, values = self.network(states)
            
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        return policies, values.squeeze(1).cpu().numpy()

    def expand_node(self, node, policy):
        valid_moves = [a for a in range(node.env.columns) 
                      if node.env.is_valid_move(a)]
        
        # Normalize policy
        policy_mask = np.zeros_like(policy)
        policy_mask[valid_moves] = 1
        policy = policy * policy_mask
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            policy = np.ones_like(policy)/len(policy)
        
        # Create children
        for action in valid_moves:
            if action not in node.children:
                new_env = clone_env(node.env)
                new_env.step(action)
                node.children[action] = MCTSNode(new_env, parent=node, 
                                                prior=policy[action], 
                                                action_taken=action)
    
    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visits += 1
            node.total_value += value
            value = -value

    def select_child(self, node):
        best_score = -float("inf")
        best_action = None
        best_child = None
        total_visits = node.visits
        for action, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            score = child.q_value + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

# ===============================
# Parallel Self-Play Pipeline
# ===============================
def self_play_worker(args):
    """Parallel self-play worker"""
    state_dict, mcts_simulations, device_str, temperature = args
    device = torch.device(device_str)
    model = ConnectFourCNN(input_channels=3).to(device)  # Changed from 5 to 3    model.load_state_dict(state_dict)
    model.eval()
    max_moves = 42  # Absolute maximum possible moves
    move_count = 0
    
    env = ConnectFourEnv()
    examples = []
    
    try:
        while not env.done and move_count < max_moves:
            move_count += 1
            root = MCTSNode(env.clone())
            mcts = MCTS(model, mcts_simulations, device=device)
            policy = mcts.search(root)

            # Validate policy probabilities
            if np.any(np.isnan(policy)) or not np.isclose(policy.sum(), 1.0, atol=1e-6):
                policy = np.ones(7)/7

            # Apply temperature
            if temperature != 1.0:
                policy = np.power(policy, 1/temperature)
                policy /= policy.sum()

            examples.append((
                state_to_tensor(env).squeeze(0).cpu().numpy(),
                policy,
                env.current_player
            ))
            action = np.random.choice(7, p=policy)
            _, reward, done, _ = env.step(action)  # Capture reward from last move

        # Determine final outcome
        if reward == 1:
            winning_player = env.current_player  # Last move was a win by current_player
            outcome = 1 if winning_player == 1 else -1
        else:
            outcome = 0  # Draw

        return [
            (s, p, outcome if c == winning_player else -outcome)
            if reward == 1 else 
            (s, p, 0)
            for s, p, c in examples
        ]
        
    except Exception as e:
        print(f"Worker error: {str(e)}")
        traceback.print_exc()
        return []

# ===============================
# Training and Evaluation
# ===============================
def train(network, optimizer, replay_buffer, batch_size, device='cpu'):
    if len(replay_buffer) < batch_size:
        return None

    batch = random.sample(replay_buffer, batch_size)
    states, target_pis, target_values = zip(*batch)
    
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    target_pis = torch.tensor(np.array(target_pis), dtype=torch.float32).to(device)
    target_values = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1).to(device)

    optimizer.zero_grad()
    policy_logits, values = network(states)
    
    loss = F.cross_entropy(policy_logits, target_pis) + F.mse_loss(values, target_values)
    loss.backward()
    optimizer.step()
    return loss.item()

def play_game(net1, net2, device='cpu', verbose=False):
    """
    Simulate a game between two networks.
    net1 plays as Player 1, net2 as Player 2.
    Returns: 1 if net1 wins, -1 if net2 wins, 0 for draw
    """
    env = ConnectFourEnv()
    current_player = 1
    mcts = MCTS(net1 if current_player == 1 else net2, num_simulations=25, device=device)
    
    while not env.done:
        root = MCTSNode(env.clone())
        mcts.network = net1 if current_player == 1 else net2
        policy = mcts.search(root)
        action = np.random.choice(env.columns, p=policy)
        _, reward, done, _ = env.step(action)
        
        if verbose:
            print(f"\nPlayer {current_player} played column {action}")
            env.render()
            
        current_player = -1 if current_player == 1 else 1

    # Return result from net1's perspective
    if reward == 1:
        return 1 if env.current_player == -1 else -1
    return 0

def evaluate_against_baseline(current_net, baseline_net, num_games=20, device='cpu'):
    wins = 0
    for game in range(num_games):
        if game % 2 == 0:
            result = play_game(current_net, baseline_net, device=device)
        else:
            result = -play_game(baseline_net, current_net, device=device)
        wins += 1 if result == 1 else 0
    win_rate = wins / num_games
    print(f"Evaluation: Current network win rate = {win_rate * 100:.2f}%")
    return win_rate

# ===============================
# Main Training Loop
# ===============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize networks
    network = ConnectFourCNN(input_channels=3).to(device)
    best_network = ConnectFourCNN(input_channels=3).to(device)
    
    # Try loading existing best model
    if os.path.exists('best_model.pt'):
        print("Loading existing best model...")
        try:
            checkpoint = torch.load('best_model.pt', map_location=device)
            network.load_state_dict(checkpoint)
            best_network.load_state_dict(checkpoint)
        except:
            print("Couldn't load existing model, initializing new one")
            network.apply(network.init_weights)
            best_network.load_state_dict(network.state_dict())
    else:
        print("No existing model found, initializing new one")
        network.apply(network.init_weights)
        best_network.load_state_dict(network.state_dict())
    
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    num_iterations = 1000
    
    params = {
        'num_iterations': num_iterations,
        'num_self_play': 50,
        'mcts_simulations': 42,
        'batch_size': 512,
        'eval_interval': 50,
        'temperature': np.linspace(0.7, 0.5, num_iterations)
    }
    
    replay_buffer = []
    
    for iteration in range(1, params['num_iterations']+1):
        start_time = time.time()
        temperature = params['temperature'][iteration-1]
        
        # Self-play
        with Pool(cpu_count()) as pool:
            args = [(network.state_dict(), 
                    params['mcts_simulations'],
                    device.type,
                    temperature)] * params['num_self_play']
            results = pool.map(self_play_worker, args)
            replay_buffer.extend([ex for res in results for ex in res])
            replay_buffer = replay_buffer[-50000:]  # Smaller buffer
        
        # Training
        losses = []
        for _ in range(200):
            loss = train(network, optimizer, replay_buffer, params['batch_size'], device)
            if loss is not None:
                losses.append(loss)
        
        scheduler.step()
        
        # Evaluation against previous best
        if iteration % params['eval_interval'] == 0:
            current_win_rate = evaluate_against_baseline(network, best_network, device=device)
            
            if current_win_rate >= 0.5:  # Only need >50% win rate
                print(f"🔥 New best model! Win rate: {current_win_rate*100:.1f}%")
                best_network.load_state_dict(network.state_dict())
                torch.save(best_network.state_dict(), "best_model.pt")
        
        print(f"Iteration {iteration} | Loss: {np.mean(losses) if losses else 0:.4f} | "
              f"Time: {time.time()-start_time:.2f}s")

    torch.save(network.state_dict(), "final_model.pt")
    print("Training complete")

if __name__ == "__main__":
    main()