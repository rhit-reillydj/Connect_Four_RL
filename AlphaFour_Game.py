import streamlit as st
import numpy as np
import os
import time
from st_bridge import bridge, html # Import streamlit-bridge and html

# --- Page Configuration (Main App Page) ---
st.set_page_config(page_title="Dominic Reilly's AlphaFour Model!", page_icon="favicon/alphafour.png", layout="wide", initial_sidebar_state="collapsed")

# Assuming these files are in the same directory or accessible in PYTHONPATH
from src.connect_four import ConnectFourGame
from src.tflite_model import ConnectFourNNetTFLite # Import TFLite NNet
from src.mcts import MCTS
from src.utils import dotdict

# Initialize session state variables for timestamp tracking
if 'last_processed_timestamp' not in st.session_state:
    st.session_state.last_processed_timestamp = None
if 'last_restart_timestamp' not in st.session_state:
    st.session_state.last_restart_timestamp = None
if 'last_view_board_timestamp' not in st.session_state:
    st.session_state.last_view_board_timestamp = None
if 'last_back_to_results_timestamp' not in st.session_state:
    st.session_state.last_back_to_results_timestamp = None
# Initialize session state variables for move history navigation
if 'last_prev_move_timestamp' not in st.session_state:
    st.session_state.last_prev_move_timestamp = None
if 'last_next_move_timestamp' not in st.session_state:
    st.session_state.last_next_move_timestamp = None

# --- Constants ---
# SQUARESIZE_HTML and RADIUS_HTML are removed, will be controlled by CSS variables
BOARD_COLOR_HTML = "#0077cc" # A nice, modern blue
EMPTY_SLOT_COLOR_HTML = "#005fa3" # Darker blue for empty slot background
HOLE_COLOR_HTML = "#222222" # Color of the "hole" before a piece is dropped
RED_PLAYER_HTML = "#ff4136" # Brighter Red
YELLOW_AI_HTML = "#ffd700"   # Gold/Yellow

PLAYER_PIECE = 1
AI_PIECE = -1

MODEL_FOLDER = '.'
# MODEL_FILENAME = 'best.weights.h5' # Old constant, replaced by preferred/fallback
TFLITE_MODEL_DIR = "./src/temp_connect_four/"
TFLITE_MODEL_FILENAME = "model.tflite"

# Early Game MCTS Settings
EARLY_GAME_MOVE_THRESHOLD = 6  # Total moves (player + AI) for deeper search
EARLY_GAME_MCTS_SIMS = 400     # Number of MCTS simulations for early game

# AI_ARGS for training (can be kept for reference or other uses if any)
TRAINING_AI_ARGS = dotdict({
    'cpuct': 1.0,
    'num_mcts_sims': 300, # Original MCTS sims for training
    'tempThreshold': 0, # This might be temp_threshold from main.py
    'temp': 0.0,
    'lr': 0.001, 'epochs': 1, 'batch_size': 32 # NNet args, not directly used by MCTS class init
})

# AI_ARGS for inference in Streamlit (MUCH faster)
INFERENCE_AI_ARGS = dotdict({
    'cpuct': 1.0, # Standard exploration constant
    'num_mcts_sims': 300,  # CRITICAL: Reduced for faster inference
    # Noise parameters usually off for deterministic best move during inference
    'add_dirichlet_noise': False,
    'dirichlet_alpha': 0.0, 
    'epsilon_noise': 0.0
})

# --- Helper Functions ---

def get_player_color_html(piece):
    if piece == PLAYER_PIECE: return RED_PLAYER_HTML
    if piece == AI_PIECE: return YELLOW_AI_HTML
    return "transparent"

def draw_board_html(board_array, game_cols, valid_moves_array, game_over_flag, current_turn_player, last_move_coords=None):
    rows, cols = board_array.shape # game_cols is cols from the board

    # action_row width is now fit-content, grid-template-columns defines its structure
    action_row_html = f"<div class='action-row' style='grid-template-columns: repeat(var(--current-board-cols), var(--square-size));'>"
    can_player_act = (not game_over_flag and current_turn_player == PLAYER_PIECE)
    
    # Get the human player's color for hover effect
    human_color = st.session_state.get('player_color', PLAYER_PIECE)
    
    # Check if it's the human's turn (regardless of color)
    is_human_turn = False
    if human_color == PLAYER_PIECE and current_turn_player == PLAYER_PIECE:
        is_human_turn = True
    elif human_color == AI_PIECE and current_turn_player == AI_PIECE:
        is_human_turn = True
    
    can_player_act = (not game_over_flag and is_human_turn)

    for c in range(cols):
        is_valid_move = valid_moves_array[c]
        action_class = "action-slot-valid" if is_valid_move and can_player_act else "action-slot-disabled"
        # .action-slot CSS now defines width/height using var(--square-size)
        # .action-piece-visual CSS defines its own size using var(--radius)
        piece_visual_html = f"<div class='action-piece-visual'></div>"
        if is_valid_move and can_player_act:
            action_row_html += f"<div class='action-slot {action_class}' onclick=\"window.top.stBridges.send('board_action_bridge', {{ 'action_col': {c}, 'timestamp': new Date().getTime() }})\">{piece_visual_html}</div>"
        else:
            action_row_html += f"<div class='action-slot {action_class}'>{piece_visual_html}</div>"
    action_row_html += "</div>"

    # board-pieces-container width is fit-content, grid-template-columns defines its structure
    html_board_pieces = f"<div class='board-pieces-container' style='grid-template-columns: repeat(var(--current-board-cols), var(--square-size));'>"
    for r in range(rows):
        for c_idx in range(cols):
            piece_color_on_board = board_array[r][c_idx]
            display_piece_color = get_player_color_html(piece_color_on_board)
            highlight_class = ""
            if last_move_coords == (r, c_idx):
                if piece_color_on_board == PLAYER_PIECE: # Player's red piece
                    highlight_class = "last-move-player"
                elif piece_color_on_board == AI_PIECE: # AI's yellow piece
                    highlight_class = "last-move-ai"
            
            cell_html = (
                f"<div class='board-cell' style='width: var(--square-size); height: var(--square-size); background-color: {BOARD_COLOR_HTML}; display: flex; justify-content: center; align-items: center;'>"
                f"<div class='board-hole' style='width: calc(var(--radius) * 2); height: calc(var(--radius) * 2); background-color: {HOLE_COLOR_HTML}; border-radius: 50%; display: flex; justify-content: center; align-items: center; position: relative;'>"
                f"<div class='piece {highlight_class}' style='width: 100%; height: 100%; background-color: {display_piece_color}; border-radius: 50%; transition: background-color 0.3s ease; box-shadow: inset 0 -3px 5px rgba(0,0,0,0.3);'></div>"
                f"</div></div>"
            )
            html_board_pieces += cell_html
    html_board_pieces += "</div>"
    
    # .board-container width is now fit-content from CSS
    board_wrapper_html = f"<div class='board-container'>"
    board_wrapper_html += html_board_pieces
    board_wrapper_html += "</div>"

    final_html_content = action_row_html + board_wrapper_html
    
    # Wrap the final content in a div that sets the --current-board-cols CSS variable
    # This wrapper is styled by .responsive-board-wrapper in the main CSS block
    final_wrapper = f"<div class='responsive-board-wrapper' style='--current-board-cols: {cols};'>\n{final_html_content}\n</div>"
    return final_wrapper

@st.cache_resource
def load_model_and_game():
    game = ConnectFourGame()
    nnet = None
    mcts = None

    try:
        # Initialize with the TFLite model directly
        # Assumes model.tflite is now in TFLITE_MODEL_DIR relative to project root
        nnet = ConnectFourNNetTFLite(game, model_filename=TFLITE_MODEL_FILENAME, model_dir=TFLITE_MODEL_DIR)
        # Use INFERENCE_AI_ARGS for MCTS in Streamlit app
        mcts = MCTS(game, nnet, INFERENCE_AI_ARGS)
    except FileNotFoundError as e:
        st.error(f"TFLite model '{os.path.join(TFLITE_MODEL_DIR, TFLITE_MODEL_FILENAME)}' not found. Please run src/converter_script.py first. Details: {e}")
        # game will be returned, but nnet and mcts will be None
    except Exception as e:
        st.error(f"Error loading TFLite model or initializing MCTS: {e}")
        # game will be returned, but nnet and mcts will be None
        
    return game, nnet, mcts

def initialize_game_state():
    print("DEBUG: Initializing or restarting game state.")
    game, nnet, mcts = load_model_and_game()
    st.session_state.game = game
    st.session_state.nnet = nnet
    st.session_state.mcts = mcts
    if game is None or nnet is None or mcts is None:
        st.session_state.error_message = "Failed to initialize game components."
        st.session_state.game_ready = False
    else:
        st.session_state.board = game.get_initial_board()
        
        # Initialize color selection states
        if 'last_color_selection_timestamp' not in st.session_state:
            st.session_state.last_color_selection_timestamp = None
            
        # Always reset color selection on game restart
        st.session_state.color_selected = False
        st.session_state.player_color = None
        
        # Set default turn and message (will be updated when color is selected)
        st.session_state.turn = PLAYER_PIECE
        st.session_state.message = ""
        st.session_state.ai_thinking = False
        
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.error_message = None
        st.session_state.game_ready = True
        st.session_state.total_moves_count = 0 # Initialize total moves count
        # Initialize move history tracking
        st.session_state.move_history = [st.session_state.board.copy()]  # Start with initial board
        st.session_state.current_history_index = 0  # Index in move_history for current view
    st.session_state.game_restarted = False
    st.session_state.last_move_coords = None # Initialize last_move_coords
    st.session_state.viewing_board = False # Initialize viewing board state

# --- New Helper Functions for Game Over Displays ---
def display_win_celebration():
    st.balloons()
    win_html = f"""
    <div id="win-overlay">
        <div class="overlay-content">
            <h1 class="win-title">VICTORY!</h1>
            <p class="win-subtitle">You conquered AlphaFour!</p>
            <p class="win-message">🎉 Congratulations, Master Strategist! 🎉</p>
            <div class="button-container">
                <div class="view-board-button-overlay" onclick=\"window.top.stBridges.send('view_board_signal', {{ 'timestamp': new Date().getTime() }})\">View Board</div>
                <div class="restart-button-overlay" onclick=\"window.top.stBridges.send('restart_game_signal', {{ 'timestamp': new Date().getTime() }})\">Play Again?</div>
            </div>
        </div>
    </div>
    """
    html(win_html)

def display_loss_devastation():
    loss_html = f"""
    <div id="loss-overlay">
        <div class="overlay-content">
            <h1 class="loss-title">DEFEATED</h1>
            <p class="loss-subtitle">AlphaFour reigns supreme...</p>
            <p class="loss-message">💔 Better luck next time! 💔</p>
            <div class="button-container">
                <div class="view-board-button-overlay" onclick=\"window.top.stBridges.send('view_board_signal', {{ 'timestamp': new Date().getTime() }})\">View Board</div>
                <div class="restart-button-overlay" onclick=\"window.top.stBridges.send('restart_game_signal', {{ 'timestamp': new Date().getTime() }})\">Try Again?</div>
            </div>
        </div>
    </div>
    """
    html(loss_html)

def display_draw_message():
    draw_html = f"""
    <div id="draw-overlay">
        <div class="overlay-content">
            <h1 class="draw-title">IT'S A DRAW!</h1>
            <p class="draw-subtitle">A hard-fought battle!</p>
            <p class="draw-message">🤝 Well played by both sides! 🤝</p>
            <div class="button-container">
                <div class="view-board-button-overlay" onclick=\"window.top.stBridges.send('view_board_signal', {{ 'timestamp': new Date().getTime() }})\">View Board</div>
                <div class="restart-button-overlay" onclick=\"window.top.stBridges.send('restart_game_signal', {{ 'timestamp': new Date().getTime() }})\">Rematch?</div>
            </div>
        </div>
    </div>
    """
    html(draw_html)

def display_board_view_controls():
    """Display the controls when viewing the board after game over"""
    # Check if we have move history to navigate
    history_length = len(st.session_state.get('move_history', []))
    current_index = st.session_state.get('current_history_index', 0)
    
    # Calculate move number for display
    move_number = current_index
    total_moves = history_length - 1  # Subtract 1 because we include initial board
    
    controls_html = f"""
    <div class="board-view-controls">
        <div class="move-counter">
            Move {move_number} / {total_moves}
        </div>
        <div class="board-view-button-container">
            <div class="back-to-results-button" onclick=\"window.top.stBridges.send('back_to_results_signal', {{ 'timestamp': new Date().getTime() }})\">Back to Results</div>
            <div class="restart-button-overlay" onclick=\"window.top.stBridges.send('restart_game_signal', {{ 'timestamp': new Date().getTime() }})\">Try Again?</div>
        </div>
    </div>
    """
    html(controls_html)

def display_color_selection():
    """Display the color selection popup for the first game"""
    color_html = f"""
    <div id="color-selection-overlay">
        <div class="overlay-content">
            <h1 class="color-title">Choose Your Color!</h1>
            <p class="color-subtitle">Who goes first?</p>
            <div class="color-button-container">
                <div class="red-button-overlay" onclick=\"window.top.stBridges.send('color_selection_bridge', {{ 'color': 'red', 'timestamp': new Date().getTime() }})\">
                    <span class="color-icon">🔴</span>
                    <span>Play First (Red)</span>
                </div>
                <div class="yellow-button-overlay" onclick=\"window.top.stBridges.send('color_selection_bridge', {{ 'color': 'yellow', 'timestamp': new Date().getTime() }})\">
                    <span class="color-icon">🟡</span>
                    <span>Play Second (Yellow)</span>
                </div>
            </div>
            <p class="color-hint">Red always goes first!</p>
        </div>
    </div>
    """
    html(color_html)

def display_board_with_navigation(board_html_content):
    """Display the board with navigation arrows on the sides when viewing board history"""
    # Check if we have move history to navigate
    history_length = len(st.session_state.get('move_history', []))
    current_index = st.session_state.get('current_history_index', 0)
    
    # Determine if navigation buttons should be enabled
    can_go_prev = current_index > 0
    can_go_next = current_index < history_length - 1
    
    # Pre-define the JavaScript code to avoid backslashes in f-string expressions
    prev_onclick = "window.top.stBridges.send('prev_move_signal', { 'timestamp': new Date().getTime() })" if can_go_prev else ""
    next_onclick = "window.top.stBridges.send('next_move_signal', { 'timestamp': new Date().getTime() })" if can_go_next else ""
    
    board_with_nav_html = f"""
    <div class="board-with-navigation">
        <div class="nav-arrow-left move-nav-button {'disabled' if not can_go_prev else ''}" 
             onclick="{prev_onclick}">
            <span class="nav-arrow">←</span>
        </div>
        <div class="board-content">
            {board_html_content}
        </div>
        <div class="nav-arrow-right move-nav-button {'disabled' if not can_go_next else ''}" 
             onclick="{next_onclick}">
            <span class="nav-arrow">→</span>
        </div>
    </div>
    """
    html(board_with_nav_html)

# --- Initialize or process restart ---
if 'game_ready' not in st.session_state or st.session_state.get('game_restarted', False):
    initialize_game_state()

# --- Debug: Show bridge data at the top of the script execution ---
clicked_action_data = bridge("board_action_bridge", default=None, key="player_action_bridge_key")
print(f"DEBUG bridge (top of script): clicked_action_data = {clicked_action_data}, last_processed_timestamp = {st.session_state.get('last_processed_timestamp')}")

if clicked_action_data is not None:
    event_action_col = clicked_action_data.get("action_col")
    event_timestamp = clicked_action_data.get("timestamp")

    if event_timestamp is not None and event_timestamp != st.session_state.get('last_processed_timestamp'):
        print(f"DEBUG bridge: New event received. Timestamp: {event_timestamp}, Col: {event_action_col}")
        
        # Get the human player's color
        human_color = st.session_state.get('player_color', PLAYER_PIECE)
        
        # Check if it's the human's turn
        is_human_turn = (not st.session_state.game_over and st.session_state.turn == human_color)
        
        if is_human_turn:
            action_col = int(event_action_col)
            print(f"DEBUG bridge: Processing action for column {action_col}")
            current_board = st.session_state.board
            game_instance = st.session_state.game

            if game_instance.get_valid_moves(current_board)[action_col]:
                print(f"DEBUG bridge: Move in col {action_col} is valid. Getting next state.")
                new_board, _, move_row = game_instance.get_next_state(current_board, human_color, action_col)
                st.session_state.board = new_board
                st.session_state.last_move_coords = (move_row, action_col) # Store player's last move
                st.session_state.total_moves_count += 1 # Increment total moves count
                # Add the new board state to move history
                st.session_state.move_history.append(new_board.copy())
                st.session_state.current_history_index = len(st.session_state.move_history) - 1
                game_end_result = game_instance.get_game_ended(st.session_state.board, human_color, last_move_col=action_col, last_move_row=move_row)
                
                if game_end_result != 0:
                    st.session_state.game_over = True
                    
                    # game_end_result is from perspective of human_color
                    # 1 means human won, -1 means AI won
                    if game_end_result == 1:  # Human won
                        st.session_state.winner = human_color
                        color_name = "Red" if human_color == PLAYER_PIECE else "Yellow"
                        st.session_state.message = f"You ({color_name}) win! 🎉"
                    elif game_end_result == -1:  # AI won
                        st.session_state.winner = -human_color  # AI color
                        ai_color_name = "Yellow" if human_color == PLAYER_PIECE else "Red"
                        st.session_state.message = f"AI ({ai_color_name}) wins! 😞"
                    else:  # Draw
                        st.session_state.winner = 0  # Use 0 for draw
                        st.session_state.message = "It's a Draw! 🤝"
                else:
                    # Switch turn to AI
                    st.session_state.turn = -human_color
                    ai_color_name = "Yellow" if human_color == PLAYER_PIECE else "Red"
                    st.session_state.message = f"AI ({ai_color_name}) is thinking... 🤔"
                    st.session_state.ai_thinking = True
            else:
                print(f"DEBUG bridge: Move in col {action_col} is invalid.")
                st.session_state.message = "Invalid move attempted. Please click a valid column."
            st.session_state.last_processed_timestamp = event_timestamp
            print(f"DEBUG bridge: last_processed_timestamp updated to {event_timestamp}")
        else:
            print(f"DEBUG bridge: New event {event_timestamp} received, but not player's turn or game over. Storing timestamp to prevent re-processing.")
            st.session_state.last_processed_timestamp = event_timestamp
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_processed_timestamp'):
        print(f"DEBUG bridge: Stale event (timestamp matches last processed: {event_timestamp}). Col: {event_action_col}. Ignoring.")
    elif event_timestamp is None:
        print(f"DEBUG bridge: Event received without timestamp. Data: {clicked_action_data}. Ignoring.")

# --- Bridge for restart game signal (must be called on every run to listen) ---
restart_signal_data = bridge("restart_game_signal", default=None, key="restart_game_bridge_key")
if restart_signal_data is not None:
    event_timestamp = restart_signal_data.get("timestamp")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_restart_timestamp'):
        print(f"DEBUG bridge: Restart signal received. Timestamp: {event_timestamp}")
        st.session_state.last_restart_timestamp = event_timestamp
        st.session_state.game_restarted = True
        st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_restart_timestamp'):
        print(f"DEBUG bridge: Stale restart signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- Bridge for view board signal (must be called on every run to listen) ---
view_board_signal_data = bridge("view_board_signal", default=None, key="view_board_bridge_key")
if view_board_signal_data is not None:
    event_timestamp = view_board_signal_data.get("timestamp")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_view_board_timestamp'):
        print(f"DEBUG bridge: View board signal received. Timestamp: {event_timestamp}")
        st.session_state.last_view_board_timestamp = event_timestamp
        st.session_state.viewing_board = True
        # Set to show the final board state when first viewing
        move_history = st.session_state.get('move_history', [])
        if len(move_history) > 0:
            st.session_state.current_history_index = len(move_history) - 1
        st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_view_board_timestamp'):
        print(f"DEBUG bridge: Stale view board signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- Bridge for back to results signal (must be called on every run to listen) ---
back_to_results_signal_data = bridge("back_to_results_signal", default=None, key="back_to_results_bridge_key")
if back_to_results_signal_data is not None:
    event_timestamp = back_to_results_signal_data.get("timestamp")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_back_to_results_timestamp'):
        print(f"DEBUG bridge: Back to results signal received. Timestamp: {event_timestamp}")
        st.session_state.last_back_to_results_timestamp = event_timestamp
        st.session_state.viewing_board = False
        st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_back_to_results_timestamp'):
        print(f"DEBUG bridge: Stale back to results signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- Bridge for color selection signal (must be called on every run to listen) ---
color_selection_data = bridge("color_selection_bridge", default=None, key="color_selection_bridge_key")
if color_selection_data is not None:
    event_timestamp = color_selection_data.get("timestamp")
    selected_color = color_selection_data.get("color")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_color_selection_timestamp'):
        print(f"DEBUG bridge: Color selection signal received. Timestamp: {event_timestamp}, Color: {selected_color}")
        st.session_state.last_color_selection_timestamp = event_timestamp
        
        # Set player color based on selection
        if selected_color == "red":
            st.session_state.player_color = PLAYER_PIECE  # Human plays as red (1)
            st.session_state.turn = PLAYER_PIECE
            st.session_state.message = "You are Red! Your turn, select a column."
            st.session_state.ai_thinking = False
        else:  # yellow
            st.session_state.player_color = AI_PIECE  # Human plays as yellow (-1)
            st.session_state.turn = PLAYER_PIECE  # Red always goes first, which is AI
            st.session_state.message = "You are Yellow! AI (Red) is making the first move... 🤔"
            st.session_state.ai_thinking = True
        
        st.session_state.color_selected = True
        st.session_state.first_game = False
        st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_color_selection_timestamp'):
        print(f"DEBUG bridge: Stale color selection signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- Bridge for previous move navigation (must be called on every run to listen) ---
prev_move_signal_data = bridge("prev_move_signal", default=None, key="prev_move_bridge_key")
if prev_move_signal_data is not None:
    event_timestamp = prev_move_signal_data.get("timestamp")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_prev_move_timestamp'):
        print(f"DEBUG bridge: Previous move signal received. Timestamp: {event_timestamp}")
        st.session_state.last_prev_move_timestamp = event_timestamp
        # Navigate to previous move in history
        current_index = st.session_state.get('current_history_index', 0)
        if current_index > 0:
            st.session_state.current_history_index = current_index - 1
            st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_prev_move_timestamp'):
        print(f"DEBUG bridge: Stale previous move signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- Bridge for next move navigation (must be called on every run to listen) ---
next_move_signal_data = bridge("next_move_signal", default=None, key="next_move_bridge_key")
if next_move_signal_data is not None:
    event_timestamp = next_move_signal_data.get("timestamp")
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_next_move_timestamp'):
        print(f"DEBUG bridge: Next move signal received. Timestamp: {event_timestamp}")
        st.session_state.last_next_move_timestamp = event_timestamp
        # Navigate to next move in history
        current_index = st.session_state.get('current_history_index', 0)
        history_length = len(st.session_state.get('move_history', []))
        if current_index < history_length - 1:
            st.session_state.current_history_index = current_index + 1
            st.rerun()
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_next_move_timestamp'):
        print(f"DEBUG bridge: Stale next move signal (timestamp matches last processed: {event_timestamp}). Ignoring.")

# --- CSS Styling --- 
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    /* Hide Streamlit's default header anchor link */
    [data-testid="stHeaderActionElements"] {{
        display: none !important;
    }}

    :root {{
        --square-size: clamp(40px, 8vw, 65px); /* Adjusted for a slightly larger board */
        --radius: calc(var(--square-size) * 0.42);
        --action-row-gap: clamp(2px, 0.8vw, 5px);
        --board-pieces-gap: clamp(1px, 0.5vw, 2px);
        --board-padding: clamp(5px, 1.5vw, 10px);
        --player-highlight-color: {YELLOW_AI_HTML}; /* For player's red piece */
        --ai-highlight-color: {RED_PLAYER_HTML};     /* For AI's yellow piece */
        --human-hover-color: {RED_PLAYER_HTML if st.session_state.get('player_color', PLAYER_PIECE) == PLAYER_PIECE else YELLOW_AI_HTML};
    }}

    body, .stApp {{ 
        font-family: 'Poppins', sans-serif; 
        background: linear-gradient(to right, #232526, #414345); 
        color: #f0f2f6; 
        overflow-x: hidden;
    }}
    /* Responsive Main Title */
    h1 {{
        font-weight: 700; 
        text-align: center; 
        color: #ffffff; 
        padding-top: clamp(10px, 3vh, 20px); 
        padding-bottom: clamp(5px, 2vh, 10px); 
        letter-spacing: 1px; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: clamp(1.6em, 6vw, 3.2em); /* Adjusted responsive font size */
    }}
    /* Responsive Game Message Bar */
    .game-message {{
        text-align: center; 
        font-size: clamp(1em, 4vw, 1.5em); /* Responsive font size */
        font-weight: 600; 
        padding: clamp(10px, 2vw, 15px); /* Responsive padding */
        border-radius: 10px; 
        background-color: rgba(255, 255, 255, 0.1); 
        color: #ffffff; 
        margin: clamp(10px, 3vh, 20px) auto; /* Responsive margin */
        width: fit-content; 
        max-width: 90%; /* Ensure it doesn't get too wide on large screens */
        box-shadow: 0 4px 10px rgba(0,0,0,0.2); 
    }}
    
    .responsive-board-wrapper {{
        width: 90%; /* Take up 90% of parent, up to max-width */
        max-width: 600px; /* Max width for larger screens */
        min-width: 280px; /* Ensure it doesn't get too crunched */
        margin: 20px auto; /* Centering and some vertical margin */
        display: flex;
        flex-direction: column;
        align-items: center; /* Center .action-row and .board-container */
        /* --current-board-cols will be set inline here by Python */
    }}
    
    .action-row {{
        display: grid; 
        /* grid-template-columns will be set inline by draw_board_html */
        gap: var(--action-row-gap);
        width: fit-content; /* Width determined by its content */
        margin-bottom: 15px;
        position: relative; 
        z-index: 10; 
    }}
    .action-slot {{
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: transparent;
        width: var(--square-size); /* Cell width */
        height: var(--square-size); /* Cell height */
    }}

    .action-piece-visual {{
        width: calc(var(--radius) * 2); 
        height: calc(var(--radius) * 2); 
        border-radius: 50%;
        box-sizing: border-box; 
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease, border-color 0.2s ease, opacity 0.2s ease;
        background-color: {"rgba(255, 65, 54, 0.3)" if st.session_state.get('player_color', PLAYER_PIECE) == PLAYER_PIECE else "rgba(255, 215, 0, 0.3)"};
        border: 2px solid {"rgba(255, 65, 54, 0.6)" if st.session_state.get('player_color', PLAYER_PIECE) == PLAYER_PIECE else "rgba(255, 215, 0, 0.6)"};
    }}
    .action-slot-valid {{ cursor: pointer !important; }}
    .action-slot-valid:hover .action-piece-visual {{
        background-color: var(--human-hover-color);
        background-image: none; 
        border-color: var(--human-hover-color);
        box-shadow: inset 0 -3px 5px rgba(0,0,0,0.3);
        transform: scale(1.05);
        opacity: 1.0;
    }}
    .action-slot-disabled {{ cursor: not-allowed !important; }}
    .action-slot-disabled .action-piece-visual {{
        background-color: #707070; 
        background-image: none; 
        border-color: #505050;
        opacity: 0.4;
    }}

    .board-container {{
        width: fit-content; /* Width determined by its content */
        background-color: {BOARD_COLOR_HTML};
        border: var(--board-padding) solid {BOARD_COLOR_HTML};
        border-radius: 15px; 
        box-shadow: 0 10px 20px rgba(0,0,0,0.3); 
        padding: var(--board-padding); /* Use CSS variable for padding */
        position: relative;
        z-index: 1;
    }}
    .board-pieces-container {{ 
        display: grid;
        /* grid-template-columns will be set inline by draw_board_html */
        gap: var(--board-pieces-gap);
    }}

    /* board-cell, board-hole, piece styles will use CSS vars in draw_board_html */
    
    .stButton[data-testid*=\"restart_game_main_btn\"]>button {{ 
        /* These styles are for the old Streamlit button, not the new overlay button */
    }}
    .stButton[data-testid*=\"restart_game_main_btn\"]>button:hover:not(:disabled) {{ 
        /* These styles are for the old Streamlit button, not the new overlay button */
    }}
    /* Responsive Footer */
    .footer {{
        text-align: center; 
        padding: clamp(10px, 3vh, 20px) clamp(5px, 2vw, 10px);
        color: #aaa; 
        font-size: clamp(0.7em, 2.5vw, 0.9em); /* Responsive font size */
    }}
    
    div.st-key-player_action_bridge_key,
    div.st-key-restart_game_bridge_key,
    div.st-key-view_board_bridge_key,
    div.st-key-back_to_results_bridge_key,
    div.st-key-color_selection_bridge_key,
    div.st-key-prev_move_bridge_key,
    div.st-key-next_move_bridge_key {{
        position: absolute !important; top: -9999px !important; left: -9999px !important;
        width: 0px !important; height: 0px !important; overflow: hidden !important;
        padding: 0px !important; margin: 0px !important; border: none !important;
        visibility: hidden !important; line-height: 0px !important; font-size: 0px !important;
    }}

    /* Fullscreen Overlay Styles - Make them responsive */
    #win-overlay, #loss-overlay, #draw-overlay, #color-selection-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        z-index: 1000;
        padding: clamp(10px, 3vw, 20px); /* Responsive padding for the overlay itself */
        box-sizing: border-box;
    }}
    
    /* Color Selection Overlay Styles */
    #color-selection-overlay {{
        background: linear-gradient(45deg, rgba(33, 37, 41, 0.95), rgba(52, 58, 64, 0.95));
    }}
    
    .color-title {{
        font-family: 'Press Start 2P', cursive;
        font-size: clamp(1.8em, 9vw, 4em);
        color: #fff;
        text-shadow: 2px 2px 0px #333;
        margin-bottom: 0.3em;
    }}
    
    .color-subtitle {{
        font-size: clamp(1em, 5vw, 1.8em);
        color: #ddd;
        margin-bottom: 1em;
    }}
    
    .color-hint {{
        font-size: clamp(0.9em, 4vw, 1.4em);
        color: #aaa;
        margin-top: 1.5em;
        font-style: italic;
    }}
    
    .color-button-container {{
        display: flex;
        flex-direction: row;
        gap: clamp(20px, 5vw, 40px);
        align-items: center;
        flex-wrap: wrap;
        justify-content: center;
        margin: clamp(20px, 4vh, 30px) 0;
    }}
    
    .red-button-overlay, .yellow-button-overlay {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: clamp(10px, 2vh, 15px);
        padding: clamp(20px, 4vw, 30px) clamp(30px, 6vw, 50px);
        font-size: clamp(1em, 4vw, 1.3em);
        border-radius: 15px;
        border: 3px solid transparent;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        color: white;
    }}
    
    .red-button-overlay {{
        background-color: {RED_PLAYER_HTML};
        border-color: {RED_PLAYER_HTML};
    }}
    
    .red-button-overlay:hover {{
        background-color: #d13026;
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 30px rgba(255, 65, 54, 0.4);
    }}
    
    .yellow-button-overlay {{
        background-color: {YELLOW_AI_HTML};
        border-color: {YELLOW_AI_HTML};
        color: #333; /* Dark text on yellow background */
    }}
    
    .yellow-button-overlay:hover {{
        background-color: #e6c200;
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
    }}
    
    .color-icon {{
        font-size: clamp(2em, 8vw, 3em);
    }}
    
    /* Responsive adjustments for color selection */
    @media (max-width: 600px) {{
        .color-button-container {{
            flex-direction: column;
            gap: clamp(15px, 3vh, 25px);
        }}
    }}
    
    .overlay-content {{
        background-color: rgba(0,0,0,0.75); /* Slightly darker for better contrast with text */
        padding: clamp(20px, 5vw, 40px); /* Responsive padding */
        border-radius: clamp(15px, 3vw, 20px); /* Responsive border-radius */
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 95%; /* Ensure content box doesn't touch screen edges */
    }}

    /* Win Celebration Styles - Responsive Text */
    #win-overlay {{
        background: linear-gradient(45deg, rgba(0,128,0,0.85), rgba(60,179,113,0.85));
    }}
    .win-title {{
        font-family: 'Press Start 2P', cursive;
        font-size: clamp(2em, 10vw, 4.5em); /* Responsive font size */
        color: #fff;
        text-shadow: 3px 3px 0px #006400, 6px 6px 0px #2E8B57;
        margin-bottom: 0.2em;
        animation: pulse-light 1.5s infinite ease-in-out;
    }}
    .win-subtitle {{
        font-size: clamp(1em, 5vw, 2em); /* Responsive font size */
        color: #fff;
        font-weight: 600;
        margin-bottom: 0.5em;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }}
    .win-message {{
        font-size: clamp(1.2em, 6vw, 2.5em); /* Responsive font size */
        color: #fff;
        margin-top: 1em;
        animation: bounce-emoji 2s infinite;
    }}

    /* Loss Devastation Styles - Responsive Text */
    #loss-overlay {{
        background: linear-gradient(45deg, rgba(50,0,0,0.7), rgba(100,0,0,0.8));
    }}
    #loss-overlay .overlay-content {{
        background-color: rgba(0,0,0,0.6); 
    }}
    .loss-title {{
        font-family: 'Press Start 2P', cursive;
        font-size: clamp(2em, 10vw, 4.5em); /* Responsive font size */
        color: #a00;
        text-shadow: 2px 2px 0px #400, -2px -2px 0px #fcc;
        margin-bottom: 0.2em;
        animation: shake-heavy 0.8s cubic-bezier(.36,.07,.19,.97) infinite;
    }}
    .loss-subtitle {{
        font-size: clamp(0.9em, 4.5vw, 1.8em); /* Responsive font size */
        color: #ccc;
        font-style: italic;
        margin-bottom: 0.5em;
    }}
    .loss-message {{
        font-size: clamp(1.2em, 6vw, 2.5em); /* Responsive font size */
        color: #bbb;
        margin-top: 1em;
    }}

    /* Draw Message Styles - Responsive Text */
    #draw-overlay {{
        background: linear-gradient(45deg, rgba(100,100,100,0.9), rgba(150,150,150,0.95));
    }}
    .draw-title {{
        font-family: 'Press Start 2P', cursive;
        font-size: clamp(1.8em, 9vw, 4em); /* Responsive font size */
        color: #eee;
        text-shadow: 2px 2px 0px #555;
        margin-bottom: 0.2em;
    }}
    .draw-subtitle, .draw-message {{
        font-size: clamp(1em, 4.5vw, 1.8em); /* Responsive font size */
        color: #ddd;
        margin-bottom: 0.5em;
    }}
    
    /* Responsive Restart Button in Overlay */
    .restart-button-overlay {{
        background-color: {RED_PLAYER_HTML};
        color: white;
        padding: clamp(10px, 3vw, 15px) clamp(20px, 5vw, 30px); /* Responsive padding */
        font-size: clamp(1em, 4vw, 1.3em); /* Responsive font size */
        border-radius: 10px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        margin-top: clamp(20px, 4vh, 30px); /* Responsive margin */
        transition: background-color 0.2s ease, transform 0.1s ease;
    }}
    .restart-button-overlay:hover {{
        background-color: #d13026;
        transform: translateY(-2px);
    }}
    
    /* Button Container for Game Over Screens */
    .button-container {{
        display: flex;
        flex-direction: row;
        gap: clamp(10px, 3vw, 20px);
        align-items: center;
        flex-wrap: wrap;
        justify-content: center;
        margin-top: clamp(20px, 4vh, 30px);
    }}
    
    /* View Board Button */
    .view-board-button-overlay {{
        background-color: #4CAF50; /* Green for "view" action */
        color: white;
        padding: clamp(10px, 3vw, 15px) clamp(20px, 5vw, 30px); /* Responsive padding */
        font-size: clamp(1em, 4vw, 1.3em); /* Responsive font size */
        border-radius: 10px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s ease, transform 0.1s ease;
    }}
    .view-board-button-overlay:hover {{
        background-color: #45a049;
        transform: translateY(-2px);
    }}
    
    /* Board View Controls - Positioned below the board */
    .board-view-controls {{
        width: 100%;
        max-width: 600px;
        margin: clamp(20px, 4vh, 30px) auto 0 auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: clamp(15px, 3vh, 20px);
        padding: 0 clamp(10px, 2vw, 20px);
    }}
    
    /* Move Counter */
    .move-counter {{
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        padding: clamp(8px, 2vw, 12px) clamp(12px, 3vw, 16px);
        border-radius: 20px;
        font-size: clamp(0.9em, 3vw, 1.1em);
        font-weight: 600;
        white-space: nowrap;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }}
    
    /* Board with Navigation Layout */
    .board-with-navigation {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(15px, 3vw, 25px);
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        padding: 0 clamp(10px, 2vw, 20px);
    }}
    
    .board-content {{
        flex: 0 0 auto;
    }}
    
    /* Navigation Arrow Buttons */
    .move-nav-button {{
        width: clamp(50px, 10vw, 70px);
        height: clamp(50px, 10vw, 70px);
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: clamp(24px, 5vw, 32px);
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        user-select: none;
        flex: 0 0 auto;
    }}
    
    .move-nav-button:hover:not(.disabled) {{
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }}
    
    .move-nav-button.disabled {{
        background-color: #cccccc;
        color: #888888;
        cursor: not-allowed;
        opacity: 0.6;
    }}
    
    .nav-arrow {{
        line-height: 1;
        font-family: 'Arial', sans-serif;
    }}
    
    /* Responsive adjustments for navigation */
    @media (max-width: 600px) {{
        .board-with-navigation {{
            gap: clamp(10px, 2vw, 15px);
        }}
        .move-nav-button {{
            width: clamp(45px, 8vw, 55px);
            height: clamp(45px, 8vw, 55px);
            font-size: clamp(20px, 4vw, 26px);
        }}
    }}
    
    .board-view-button-container {{
        display: flex;
        flex-direction: row;
        gap: clamp(15px, 4vw, 25px);
        align-items: center;
        flex-wrap: wrap;
        justify-content: center;
    }}
    
    /* Back to Results Button */
    .back-to-results-button {{
        background-color: #2196F3; /* Blue for navigation action */
        color: white;
        padding: clamp(10px, 3vw, 15px) clamp(20px, 5vw, 30px);
        font-size: clamp(1em, 4vw, 1.3em);
        border-radius: 10px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s ease, transform 0.1s ease;
        white-space: nowrap;
    }}
    .back-to-results-button:hover {{
        background-color: #1976D2;
        transform: translateY(-2px);
    }}
    
    /* Responsive adjustments for small screens */
    @media (max-width: 480px) {{
        .button-container {{
            flex-direction: column;
            gap: clamp(8px, 2vh, 15px);
        }}
        .board-view-button-container {{
            flex-direction: column;
            gap: clamp(10px, 2vh, 15px);
        }}
    }}

    /* Animations */
    @keyframes pulse-light {{
        0% {{ transform: scale(1); opacity: 0.9; }}
        50% {{ transform: scale(1.05); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 0.9; }}
    }}
    @keyframes bounce-emoji {{
        0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
        40% {{ transform: translateY(-20px); }}
        60% {{ transform: translateY(-10px); }}
    }}
    @keyframes shake-heavy {{
      10%, 90% {{ transform: translate3d(-1px, -1px, 0); }}
      20%, 80% {{ transform: translate3d(2px, 2px, 0); }}
      30%, 50%, 70% {{ transform: translate3d(-3px, -3px, 0); }}
      40%, 60% {{ transform: translate3d(3px, 3px, 0); }}
    }}

    .piece.last-move-player, .piece.last-move-ai {{
        /* outline: 2px solid #fff; */ /* White outline for base contrast - REMOVED */
        box-sizing: border-box;
        transform: scale(1.10);
        z-index: 20;
        transition: border 0.1s, /* outline 0.1s, */ transform 0.1s; /* outline transition removed */
    }}
    .piece.last-move-player {{
        border: 4px solid var(--player-highlight-color);
    }}
    .piece.last-move-ai {{
        border: 4px solid var(--ai-highlight-color);
    }}

    /* Reset restart button margin when in button containers */
    .button-container .restart-button-overlay,
    .board-view-button-container .restart-button-overlay {{
        margin-top: 0;
    }}

    /* Hide all bridge component iframes and containers */
    iframe[src*="st_bridge.bridge.bridge"] {{
        display: none !important;
        height: 0px !important;
        width: 0px !important;
    }}
    
    /* Hide bridge component containers by targeting specific class patterns */
    /* COMMENTED OUT - :has() selector not universally supported
    .st-emotion-cache-8atqhb:has(iframe[src*="st_bridge"]) {{
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }}
    */
    
    /* Fallback: hide custom components with bridge iframes */
    div[data-testid="stCustomComponentV1"] {{
        position: relative;
    }}
    
    /* COMMENTED OUT - was hiding the board
    div[data-testid="stCustomComponentV1"] iframe[height="0"] {{
        display: none !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
    }}
    */

</style>

<!--
<script>
// Hide bridge component containers that take up space - but preserve board components
document.addEventListener('DOMContentLoaded', function() {{
    function hideBridgeElements() {{
        // Only hide iframes with specific bridge URLs that have height="0"
        const bridgeIframes = document.querySelectorAll('iframe[src*="st_bridge.bridge.bridge"][height="0"]');
        bridgeIframes.forEach(iframe => {{
            // Hide the iframe itself
            iframe.style.display = 'none';
            iframe.style.height = '0px';
            iframe.style.width = '0px';
            
            // Hide parent container only if it contains ONLY bridge content
            const parent = iframe.closest('[data-testid="stCustomComponentV1"]');
            if (parent && parent.children.length === 1) {{
                parent.style.display = 'none';
                parent.style.height = '0px';
                parent.style.margin = '0px';
                parent.style.padding = '0px';
            }}
        }});
    }}
    
    // Run immediately and on mutations
    hideBridgeElements();
    
    // Watch for new elements being added
    const observer = new MutationObserver(hideBridgeElements);
    observer.observe(document.body, {{ childList: true, subtree: true }});
}});
</script>
-->
""", unsafe_allow_html=True)

# --- Main App Title ---
st.title("Dominic Reilly's AlphaFour Model!")

# --- Error handling for game initialization ---
if st.session_state.get('error_message'):
    st.error(st.session_state.error_message)
    if st.button("Try Re-initializing Game", key="reinit_err_btn"):
        st.session_state.game_restarted = True 
        st.rerun()
    st.stop()

if not st.session_state.get("game_ready", False):
    st.error("Game could not be initialized. Please check model files or console.")
    if st.button("Retry Initialization", key="retry_init_btn"):
        st.session_state.game_restarted = True
        st.rerun()
    st.stop()

# --- Retrieve current game state variables ---
game = st.session_state.game
board_rows, board_cols = game.get_board_size()
board = st.session_state.board
turn = st.session_state.turn
game_over = st.session_state.game_over
ai_thinking = st.session_state.ai_thinking
nnet = st.session_state.nnet
mcts = st.session_state.mcts

# --- Check if color selection is needed ---
if st.session_state.get("game_ready", False) and not st.session_state.get('color_selected', False):
    display_color_selection()
    st.stop()  # Stop execution until color is selected

# --- Main UI Display (conditionally rendered if game not over, or to show final board state) ---
if not game_over:
    st.markdown(f"<div class='game-message'>{st.session_state.message}</div>", unsafe_allow_html=True)
elif game_over and st.session_state.get('viewing_board', False):
    # Show the final game result when viewing board
    st.markdown(f"<div class='game-message'>{st.session_state.message}</div>", unsafe_allow_html=True)

# Always display the board and action row if game is ready, so player can see final state
if st.session_state.get("game_ready", False) and st.session_state.get('color_selected', False):
    # When viewing board and navigating history, show the historical board state
    if game_over and st.session_state.get('viewing_board', False):
        # Get the board state from move history based on current index
        current_index = st.session_state.get('current_history_index', 0)
        move_history = st.session_state.get('move_history', [])
        if current_index < len(move_history):
            display_board = move_history[current_index]
        else:
            display_board = board  # Fallback to current board
        
        # For historical moves, disable interaction (all moves invalid)
        valid_moves = [False] * board_cols
        
        # Calculate last move coordinates for highlighting
        display_last_move_coords = None
        if current_index > 0 and current_index < len(move_history):
            # Find the difference between current and previous board to highlight the move
            prev_board = move_history[current_index - 1]
            curr_board = move_history[current_index]
            
            # Find where a piece was added
            for r in range(board_rows):
                for c in range(board_cols):
                    if prev_board[r][c] == 0 and curr_board[r][c] != 0:
                        display_last_move_coords = (r, c)
                        break
                if display_last_move_coords:
                    break
        
        board_html_content = draw_board_html(display_board, board_cols, valid_moves, True, turn, display_last_move_coords)
        display_board_with_navigation(board_html_content)
    else:
        # Normal gameplay - show current board state
        valid_moves = game.get_valid_moves(board) 
        board_html_content = draw_board_html(board, board_cols, valid_moves, game_over, turn, st.session_state.get('last_move_coords'))
        html(board_html_content)

# --- AI's Turn Logic ---
# Get the human and AI colors
human_color = st.session_state.get('player_color', PLAYER_PIECE)
ai_color = -human_color

# Check if it's AI's turn
if not game_over and turn == ai_color and ai_thinking:
    if nnet is None or mcts is None:
        st.error("AI components not loaded. Cannot make a move.")
        st.session_state.message = "AI Error. Your turn."
        st.session_state.turn = human_color
        st.session_state.ai_thinking = False
    else:
        time.sleep(0.75) # Keep a small delay for UX
        canonical_board_ai = game.get_canonical_form(board, ai_color)
        st.session_state.mcts.reset_search_state() # Reset MCTS state before getting action

        # Determine MCTS simulations based on game phase
        default_inference_sims = INFERENCE_AI_ARGS.num_mcts_sims
        current_mcts_sims_to_use = default_inference_sims

        if st.session_state.total_moves_count < EARLY_GAME_MOVE_THRESHOLD:
            current_mcts_sims_to_use = EARLY_GAME_MCTS_SIMS
            print(f"DEBUG: Early game (move {st.session_state.total_moves_count + 1}), using {current_mcts_sims_to_use} MCTS sims.")
        else:
            print(f"DEBUG: Mid/Late game (move {st.session_state.total_moves_count + 1}), using {current_mcts_sims_to_use} MCTS sims.")

        # Temporarily set the desired number of sims for this specific call
        original_sims_in_mcts_args = st.session_state.mcts.args.num_mcts_sims
        st.session_state.mcts.args.num_mcts_sims = current_mcts_sims_to_use
        
        ai_action_probs = mcts.getActionProb(canonical_board_ai, temp=0)
        
        # Restore the MCTS args to its configured default for inference.
        st.session_state.mcts.args.num_mcts_sims = default_inference_sims # Restore to default inference sims

        ai_action = np.argmax(ai_action_probs)

        if not game.get_valid_moves(board)[ai_action]:
            st.warning(f"AI suggested invalid move {ai_action}. Picking first valid move.")
            valid_indices = [i for i, valid in enumerate(game.get_valid_moves(board)) if valid]
            if valid_indices: ai_action = valid_indices[0]
            else:
                st.session_state.game_over = True
                st.session_state.winner = 0 # Indicate draw or no moves
                st.session_state.message = "Game ended: No valid AI moves."
                st.rerun() 
                st.stop() 

        new_board, _, ai_move_row = game.get_next_state(board, ai_color, ai_action)
        st.session_state.board = new_board
        st.session_state.last_move_coords = (ai_move_row, ai_action) # Store AI's last move
        st.session_state.total_moves_count += 1 # Increment total moves count
        # Add the new board state to move history
        st.session_state.move_history.append(new_board.copy())
        st.session_state.current_history_index = len(st.session_state.move_history) - 1
        
        # Process game end after AI's move
        current_game_end_result = game.get_game_ended(st.session_state.board, ai_color, last_move_col=ai_action, last_move_row=ai_move_row)
        
        if current_game_end_result != 0: # Game has ended
            st.session_state.game_over = True
            
            # current_game_end_result is from perspective of ai_color
            # 1 means AI won, -1 means human won
            if current_game_end_result == 1:  # AI won
                st.session_state.winner = ai_color
                ai_color_name = "Red" if ai_color == PLAYER_PIECE else "Yellow"
                st.session_state.message = f"AI ({ai_color_name}) wins! 😞"
            elif current_game_end_result == -1:  # Human won
                st.session_state.winner = human_color
                human_color_name = "Red" if human_color == PLAYER_PIECE else "Yellow"
                st.session_state.message = f"You ({human_color_name}) win! 🎉"
            else:  # Draw
                st.session_state.winner = 0  # Use 0 for draw
                st.session_state.message = "It's a Draw! 🤝"
        else: # Game continues
            st.session_state.turn = human_color
            human_color_name = "Red" if human_color == PLAYER_PIECE else "Yellow"
            st.session_state.message = f"Your turn! Select a column."
            
    st.session_state.ai_thinking = False
    st.rerun()

# --- Game Over Display and Restart ---
if game_over:
    winner_val = st.session_state.get('winner')
    human_color = st.session_state.get('player_color', PLAYER_PIECE)

    # Show overlay only when not viewing board
    if not st.session_state.get('viewing_board', False):
        if winner_val == human_color:
            display_win_celebration()
        elif winner_val == -human_color:
            display_loss_devastation()
        else:
            display_draw_message()
    else:
        # When viewing board, show the back to results controls
        display_board_view_controls()
    
    # The restart button is now part of the HTML in the display_ functions
    # and handled by the 'restart_game_signal' bridge above.
    # Remove old st.button logic for restart here.

# --- Footer ---
if not game_over:
    st.markdown("<div class='footer'>Dominic Reilly's AlphaFour - Connect Four Streamlit Edition</div>", unsafe_allow_html=True)

# Debugging: Display session state (optional)
# with st.expander("Session State (Debug)"):
# st.write(st.session_state) 