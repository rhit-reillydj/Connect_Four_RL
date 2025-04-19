# play_connect_four.py

import sys
import math
import time
import copy
import random
import numpy as np
import torch
import pygame

from connect_four import ConnectFourEnv
from cnn_model import ConnectFourCNN
from reinforcement_learning import MCTS, MCTSNode, clone_env

# Define colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)       # Human (Player 1)
YELLOW = (255, 255, 0)  # AI (Player 2)
WHITE = (255, 255, 255)

# Board parameters
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE/2 - 5)

# Window dimensions
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE  # Extra top row for piece drop
size = (width, height)

def draw_board(screen, board):
    """Draws the Connect Four board on the given pygame screen."""
    # Draw board background (blue rectangle)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, (r+1)*SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # Draw empty slots as black circles.
            pygame.draw.circle(screen, BLACK, 
                               (int(c*SQUARESIZE + SQUARESIZE/2), int((r+1)*SQUARESIZE + SQUARESIZE/2)), 
                               RADIUS)
    
    # Draw pieces (if any)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                # Human piece (red)
                pygame.draw.circle(screen, RED, 
                                   (int(c*SQUARESIZE + SQUARESIZE/2), int((r+1)*SQUARESIZE + SQUARESIZE/2)), 
                                   RADIUS)
            elif board[r][c] == -1:
                # AI piece (yellow)
                pygame.draw.circle(screen, YELLOW, 
                                   (int(c*SQUARESIZE + SQUARESIZE/2), int((r+1)*SQUARESIZE + SQUARESIZE/2)), 
                                   RADIUS)
    pygame.display.update()

def animate_drop(screen, board, col, row, color):
    """
    Animate the dropping of a piece in the given column until it lands at the given row.
    """
    x_pos = int(col * SQUARESIZE + SQUARESIZE/2)
    # Start from top (outside board) and drop down.
    for y in range(int(SQUARESIZE/2), height - row*SQUARESIZE - int(SQUARESIZE/2), 10):
        # Redraw board to erase previous animation frame.
        screen.fill(BLACK)
        draw_board(screen, board)
        pygame.draw.circle(screen, color, (x_pos, y), RADIUS)
        pygame.display.update()
        pygame.time.delay(10)

def get_ai_move(model, env, device):
    """
    Use a reduced MCTS search to get the AI move.
    """
    root = MCTSNode(clone_env(env))
    # Use fewer simulations for interactive speed.
    mcts = MCTS(model, num_simulations=25, device=device)
    mcts_policy = mcts.search(root)
    # Choose move stochastically from the MCTS policy.
    move = np.random.choice(env.columns, p=mcts_policy)
    return move

def main():
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect Four: Human (Red) vs AI (Yellow)")
    font = pygame.font.SysFont("monospace", 50)

    # Initialize environment and board.
    env = ConnectFourEnv()
    board = env.board.copy()

    # Load trained CNN model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConnectFourCNN().to(device)
    model.load_state_dict(torch.load("Models/best_model.pt", map_location=device))
    model.eval()

    # Initial draw
    screen.fill(BLACK)
    draw_board(screen, board)

    game_over = False
    turn = env.current_player  # Assume human is Player 1 (Red), AI is Player 2 (Yellow)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            # Human turn: wait for mouse click input.
            if turn == 1 and event.type == pygame.MOUSEBUTTONDOWN:
                # Get x coordinate of click.
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))
                if env.is_valid_move(col):
                    # Animate drop for human piece.
                    row = env.get_next_open_row(col)
                    animate_drop(screen, board, col, ROW_COUNT - row, RED)
                    # Take step.
                    board, reward, done, info = env.step(col)
                    draw_board(screen, board)
                    if done:
                        game_over = True
                    turn = env.current_player

        # AI turn
        if turn == -1 and not game_over:
            # Small delay for better UX.
            pygame.time.delay(500)
            col = get_ai_move(model, env, device)
            if env.is_valid_move(col):
                row = env.get_next_open_row(col)
                animate_drop(screen, board, col, ROW_COUNT - row, YELLOW)
                board, reward, done, info = env.step(col)
                draw_board(screen, board)
                if done:
                    game_over = True
                turn = env.current_player

        if game_over:
            # Display win/draw message.
            label = None
            if reward == 1:
                # The last move winner is the player who just played.
                # Because our env switches turn after move (if not game over),
                # we can infer the winner by checking env.current_player.
                # (Remember, the env doesn't switch if game is over.)
                if env.current_player != 1:
                    # Last move was by player 2 (AI) because we didn't switch.
                    label = font.render("AI wins!", 1, YELLOW)
                else:
                    label = font.render("You win!", 1, RED)
            else:
                label = font.render("Draw!", 1, WHITE)
            screen.blit(label, (40,10))
            pygame.display.update()
            pygame.time.delay(3000)

if __name__ == "__main__":
    main()