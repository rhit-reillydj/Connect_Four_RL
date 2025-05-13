import pygame
import sys
import numpy as np
import math
import os

from connect_four import ConnectFourGame
from model import ConnectFourNNet
from mcts import MCTS
from utils import dotdict

# --- Constants ---
SQUARESIZE = 100
RADIUS = int(SQUARESIZE * 0.4) # Slightly smaller radius for better padding
BOARD_BLUE = (20, 80, 180) # Darker, richer blue for the board
BACKGROUND_BLACK = (10, 10, 10) # Background for empty slots
RED_PLAYER = (220, 50, 50) # Brighter Red
YELLOW_AI = (255, 200, 0) # Brighter Yellow

MESSAGE_AREA_COLOR = (220, 220, 220) # Light grey for message area
MESSAGE_TEXT_COLOR = (30, 30, 30)
HOVER_ALPHA = 128 # Alpha for hover piece

PLAYER_PIECE = 1
AI_PIECE = -1

# --- Pygame Helper Functions ---
def draw_board_slots_background(screen, game_rows, game_cols):
    """Draws the background for the board slots (the black circles)."""
    pygame.draw.rect(screen, BOARD_BLUE, (0, SQUARESIZE, game_cols * SQUARESIZE, game_rows * SQUARESIZE))
    for c in range(game_cols):
        for r in range(game_rows):
            pygame.draw.circle(screen, BACKGROUND_BLACK, 
                               (int(c * SQUARESIZE + SQUARESIZE / 2), 
                                int((r * SQUARESIZE) + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

def draw_board_grid_overlay(screen, game_rows, game_cols):
    """Draws the blue grid lines on top of pieces."""
    grid_line_color = BOARD_BLUE 
    line_thickness = 2 # Thinner lines for a crisper grid

    # Draw horizontal lines
    for r_idx in range(game_rows + 1):
        y_pos = (r_idx * SQUARESIZE) + SQUARESIZE
        start_pos = (0, y_pos)
        end_pos = (game_cols * SQUARESIZE, y_pos)
        pygame.draw.line(screen, grid_line_color, start_pos, end_pos, line_thickness)

    # Draw vertical lines
    for c_idx in range(game_cols + 1):
        x_pos = c_idx * SQUARESIZE
        start_pos = (x_pos, SQUARESIZE) 
        end_pos = (x_pos, (game_rows * SQUARESIZE) + SQUARESIZE) 
        pygame.draw.line(screen, grid_line_color, start_pos, end_pos, line_thickness)

def draw_pieces(screen, board_array, game_rows, game_cols):
    """Draws all the pieces currently on the board_array."""
    for c in range(game_cols):
        for r in range(game_rows):
            if board_array[r][c] == PLAYER_PIECE:
                color = RED_PLAYER
            elif board_array[r][c] == AI_PIECE:
                color = YELLOW_AI
            else:
                continue
            center_x = int(c * SQUARESIZE + SQUARESIZE / 2)
            center_y = int((r * SQUARESIZE) + SQUARESIZE + SQUARESIZE / 2)
            pygame.draw.circle(screen, color, (center_x, center_y), RADIUS)

def draw_animated_piece(screen, player, col, y_pos):
    """Draws a single piece at a specific y position, used for animation."""
    color = RED_PLAYER if player == PLAYER_PIECE else YELLOW_AI
    center_x = int(col * SQUARESIZE + SQUARESIZE / 2)
    pygame.draw.circle(screen, color, (center_x, int(y_pos)), RADIUS)

def draw_column_indicator(screen, player, current_mouse_x):
    """Draws a piece at the top indicating the drop column."""
    if player == PLAYER_PIECE: # Only show for human player
        color = RED_PLAYER
        col = int(math.floor(current_mouse_x / SQUARESIZE))
        # Ensure indicator stays within board bounds if mouse goes off-screen (optional)
        col = max(0, min(col, (screen.get_width() // SQUARESIZE) - 1))
        center_x = int(col * SQUARESIZE + SQUARESIZE / 2)
        pygame.draw.circle(screen, color, (center_x, int(SQUARESIZE/2)), RADIUS)

def display_message(screen, message, game_cols, font_size=35, color=MESSAGE_TEXT_COLOR):
    width = game_cols * SQUARESIZE
    font = pygame.font.SysFont("Arial", font_size, bold=True)
    text_surface = font.render(message, True, color)
    text_rect = text_surface.get_rect(center=(width / 2, SQUARESIZE / 2))
    
    # Clear message area
    screen.fill(MESSAGE_AREA_COLOR, (0,0, width, SQUARESIZE))
    screen.blit(text_surface, text_rect)
    # No pygame.display.update() here, caller should handle it

# --- Main Game Function ---
def play_game_with_ui():
    game = ConnectFourGame()
    board_rows, board_cols = game.get_board_size()

    ai_args = dotdict({
        'cpuct': 1.0,
        'num_mcts_sims': 50, 
        'lr': 0.001, 'epochs': 1, 'batch_size': 32
    })

    print("Loading AI model...")
    nnet = ConnectFourNNet(game, ai_args)
    model_folder = './temp_connect_four/' 
    model_filename = 'best.weights.h5'
    
    if os.path.exists(os.path.join(model_folder, model_filename)):
        nnet.load_checkpoint(folder=model_folder, filename=model_filename)
        print(f"Model {model_filename} loaded.")
    else:
        print(f"Model {model_filename} not found. AI cannot play.")
        return
    ai_mcts = MCTS(game, nnet, ai_args)

    pygame.init()
    pygame.font.init() # Initialize font module

    width = board_cols * SQUARESIZE
    height = (board_rows + 1) * SQUARESIZE
    size = (width, height)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Connect Four - Human (Red) vs AI (Yellow)")
    
    board_state = game.get_initial_board()
    game_over = False
    turn = PLAYER_PIECE
    human_can_move = True # Control when human can input

    # Animation variables
    is_animating = False
    animating_piece_y = 0
    animating_piece_col = -1
    animating_piece_player = 0
    animation_speed = 25 # Increased speed
    target_row_for_animation = -1

    clock = pygame.time.Clock() # For controlling frame rate

    # Initial draw
    screen.fill(MESSAGE_AREA_COLOR) # Fill entire screen initially or just top bar
    draw_board_slots_background(screen, board_rows, board_cols)
    draw_pieces(screen, board_state, board_rows, board_cols)
    display_message(screen, "Your (Red) Turn - Click a Column", board_cols)
    pygame.display.flip() # Use flip for full screen update once

    # --- Game Loop ---
    while True: # Main loop, sys.exit() will quit
        current_mouse_pos_x = pygame.mouse.get_pos()[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not is_animating and human_can_move and turn == PLAYER_PIECE and not game_over:
                col = int(math.floor(current_mouse_pos_x / SQUARESIZE))
                if 0 <= col < board_cols and game.get_valid_moves(board_state)[col]:
                    human_can_move = False # Prevent multiple clicks during animation
                    is_animating = True
                    animating_piece_player = PLAYER_PIECE
                    animating_piece_col = col
                    animating_piece_y = SQUARESIZE / 2 # Start at top of message bar
                    
                    # Find target row for animation (lowest empty row)
                    for r_idx in range(board_rows - 1, -1, -1):
                        if board_state[r_idx][col] == 0:
                            target_row_for_animation = r_idx
                            break
                else:
                    print("Invalid move chosen by player or outside board.")
                    # Temporary message for invalid move (could be prettier)
                    # (display_message will be called in main drawing section)

        # AI's turn logic (outside event loop, controlled by `turn` and `not is_animating`)
        if turn == AI_PIECE and not game_over and not is_animating:
            human_can_move = False # AI is thinking/moving
            temp_display_message_surface = pygame.Surface((width, SQUARESIZE))
            temp_display_message_surface.fill(MESSAGE_AREA_COLOR)
            font = pygame.font.SysFont("Arial", 35, bold=True)
            text_surface = font.render("AI (Yellow) Thinking...", True, MESSAGE_TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(width / 2, SQUARESIZE / 2))
            temp_display_message_surface.blit(text_surface, text_rect)
            screen.blit(temp_display_message_surface, (0,0))
            pygame.display.flip()

            canonical_board_ai = game.get_canonical_form(board_state, AI_PIECE)
            ai_policy = ai_mcts.getActionProb(canonical_board_ai, temp=0)
            ai_action = np.argmax(ai_policy)

            if game.get_valid_moves(board_state)[ai_action]:
                is_animating = True
                animating_piece_player = AI_PIECE
                animating_piece_col = ai_action
                animating_piece_y = SQUARESIZE / 2
                for r_idx in range(board_rows - 1, -1, -1):
                    if board_state[r_idx][ai_action] == 0:
                        target_row_for_animation = r_idx
                        break
                print(f"AI chose column: {ai_action}")
            else:
                # AI Error Handling (Simplified)
                print(f"Error: AI chose an invalid move: {ai_action}. Policy: {ai_policy}")
                # Let human play again or pick random for AI
                human_can_move = True 
                turn = PLAYER_PIECE # Or handle AI error more gracefully
                # (message update will happen in draw section)

        # Animation Logic
        if is_animating:
            target_y_pixel = (target_row_for_animation * SQUARESIZE) + SQUARESIZE + (SQUARESIZE / 2)
            animating_piece_y += animation_speed
            if animating_piece_y >= target_y_pixel:
                animating_piece_y = target_y_pixel # Snap to final position
                is_animating = False
                board_state[target_row_for_animation][animating_piece_col] = animating_piece_player
                
                # Check game end state after piece lands
                result = game.get_game_ended(board_state, animating_piece_player)
                if result == 1:
                    game_over = True
                elif result == -1: # Opponent won - this should not happen from current player perspective
                    game_over = True # Should be caught by a win for the other player
                elif result == 1e-4:
                    game_over = True 
                
                if not game_over:
                    turn = -animating_piece_player # Switch turn
                    if turn == PLAYER_PIECE:
                        human_can_move = True
                animating_piece_col = -1 # Reset animation col
        
        # --- Drawing Section (Order is important for layering) ---
        # 1. Fill the entire screen with a base color (e.g. message area color or other neutral)
        screen.fill(MESSAGE_AREA_COLOR) 

        # 2. Draw board slots background (black circles on board-colored rect)
        draw_board_slots_background(screen, board_rows, board_cols)

        # 3. Draw placed pieces
        draw_pieces(screen, board_state, board_rows, board_cols)
        
        # 4. Draw hover piece (if human turn, not animating, not game over)
        if turn == PLAYER_PIECE and not is_animating and not game_over and human_can_move:
            hover_color = list(RED_PLAYER) + [HOVER_ALPHA] # Add alpha
            temp_surface = pygame.Surface((RADIUS*2, RADIUS*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, hover_color, (RADIUS, RADIUS), RADIUS)
            screen.blit(temp_surface, (current_mouse_pos_x - RADIUS, int(SQUARESIZE/2) - RADIUS))
        
        # 5. Display current message
        if game_over:
            winner_player = animating_piece_player # The player who made the last move
            if game.get_game_ended(board_state, winner_player) == 1:
                msg = "You Win! (Red)" if winner_player == PLAYER_PIECE else "AI Wins! (Yellow)"
            elif game.get_game_ended(board_state, winner_player) == 1e-4:
                msg = "It's a Draw!"
            else: # Should mean opponent won, check with -winner_player
                 if game.get_game_ended(board_state, -winner_player) == 1:
                    msg = "AI Wins! (Yellow)" if -winner_player == AI_PIECE else "You Win! (Red)"
                 else:
                    msg = "Game Over! Press R to Restart"
            display_message(screen, msg, board_cols, font_size=38)
        elif is_animating:
            msg = "Piece Falling..."
            display_message(screen, msg, board_cols)
        elif turn == PLAYER_PIECE:
            display_message(screen, "Your (Red) Turn - Click a Column", board_cols)
        elif turn == AI_PIECE:
            display_message(screen, "AI (Yellow) Thinking...", board_cols)
        
        # 6. Draw the board grid
        draw_board_grid_overlay(screen, board_rows, board_cols)

        # 7. Draw the currently animating piece (if any) on top
        if is_animating and animating_piece_col != -1:
            draw_animated_piece(screen, animating_piece_player, animating_piece_col, animating_piece_y)

        # 8. Draw column indicator piece (if human turn, not animating, not game over)
        if turn == PLAYER_PIECE and not is_animating and not game_over and human_can_move:
            draw_column_indicator(screen, PLAYER_PIECE, current_mouse_pos_x)

        pygame.display.flip() # Update the full screen
        clock.tick(60) # Increased FPS for smoother animation

if __name__ == '__main__':
    play_game_with_ui() 