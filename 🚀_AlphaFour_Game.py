import streamlit as st
import numpy as np
import os
import time
from st_bridge import bridge, html # Import streamlit-bridge and html

# Assuming these files are in the same directory or accessible in PYTHONPATH
from connect_four import ConnectFourGame
from model import ConnectFourNNet
from mcts import MCTS
from utils import dotdict

# --- Constants ---
SQUARESIZE_HTML = 70  # Slightly smaller for a tighter look
RADIUS_HTML = int(SQUARESIZE_HTML * 0.42) # Slightly larger radius within the square
BOARD_COLOR_HTML = "#0077cc" # A nice, modern blue
EMPTY_SLOT_COLOR_HTML = "#005fa3" # Darker blue for empty slot background
HOLE_COLOR_HTML = "#222222" # Color of the "hole" before a piece is dropped
RED_PLAYER_HTML = "#ff4136" # Brighter Red
YELLOW_AI_HTML = "#ffd700"   # Gold/Yellow

PLAYER_PIECE = 1
AI_PIECE = -1

MODEL_FOLDER = './temp_connect_four/'
MODEL_FILENAME = 'best.weights.h5'

AI_ARGS = dotdict({
    'cpuct': 1.0,
    'num_mcts_sims': 50,
    'tempThreshold': 15,
    'temp': 0.0,
    'lr': 0.001, 'epochs': 1, 'batch_size': 32
})

# --- Helper Functions ---

def get_player_color_html(piece):
    if piece == PLAYER_PIECE: return RED_PLAYER_HTML
    if piece == AI_PIECE: return YELLOW_AI_HTML
    return "transparent"

def draw_board_html(board_array, game_cols, valid_moves_array, game_over_flag, current_turn_player):
    rows, cols = board_array.shape
    action_row_html = "<div class='action-row' style='display: grid; grid-template-columns: repeat(" + str(game_cols) + ", 1fr); gap: 5px; margin-bottom: 15px; width: " + str(cols * SQUARESIZE_HTML + (cols -1) * 5) + "px; margin-left: auto; margin-right: auto;'>"
    can_player_act = (not game_over_flag and current_turn_player == PLAYER_PIECE)

    for c in range(game_cols):
        is_valid_move = valid_moves_array[c]
        action_class = "action-slot-valid" if is_valid_move and can_player_act else "action-slot-disabled"
        # All pieces will now use the same default style, defined in CSS for .action-piece-visual
        piece_visual_html = f"<div class='action-piece-visual'></div>"
        if is_valid_move and can_player_act:
            action_row_html += f"<div class='action-slot {action_class}' onclick=\"window.top.stBridges.send('board_action_bridge', {{ 'action_col': {c}, 'timestamp': new Date().getTime() }})\">{piece_visual_html}</div>"
        else:
            action_row_html += f"<div class='action-slot {action_class}'>{piece_visual_html}</div>"
    action_row_html += "</div>"

    html_board_pieces = f"<div class='board-pieces-container' style='display: grid; grid-template-columns: repeat({cols}, {SQUARESIZE_HTML}px); grid-gap: 2px;'>"
    for r in range(rows):
        for c in range(cols):
            piece_color = get_player_color_html(board_array[r][c])
            cell_html = (
                f"<div class='board-cell' style='width: {SQUARESIZE_HTML}px; height: {SQUARESIZE_HTML}px; background-color: {BOARD_COLOR_HTML}; display: flex; justify-content: center; align-items: center;'>"
                f"<div class='board-hole' style='width: {RADIUS_HTML*2}px; height: {RADIUS_HTML*2}px; background-color: {HOLE_COLOR_HTML}; border-radius: 50%; display: flex; justify-content: center; align-items: center; position: relative;'>"
                f"<div class='piece' style='width: 100%; height: 100%; background-color: {piece_color}; border-radius: 50%; transition: background-color 0.3s ease; box-shadow: inset 0 -3px 5px rgba(0,0,0,0.3);'></div>"
                f"</div></div>"
            )
            html_board_pieces += cell_html
    html_board_pieces += "</div>"
    
    board_wrapper_html = f"<div class='board-container' style='background-color: {BOARD_COLOR_HTML}; border: 10px solid {BOARD_COLOR_HTML}; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); width: fit-content; margin: 0 auto; padding: 10px;'>"
    board_wrapper_html += html_board_pieces
    board_wrapper_html += "</div>"

    # Concatenate action row first, then the board container
    final_html = action_row_html + board_wrapper_html
    return final_html

@st.cache_resource
def load_model_and_game():
    game = ConnectFourGame()
    nnet = ConnectFourNNet(game, AI_ARGS)
    model_path = os.path.join(MODEL_FOLDER, MODEL_FILENAME)
    if os.path.exists(model_path):
        try:
            nnet.load_checkpoint(folder=MODEL_FOLDER, filename=MODEL_FILENAME)
        except Exception as e:
            st.error(f"Error loading model {MODEL_FILENAME}: {e}. AI may not function.")
            return game, None, None
    else:
        st.error(f"Model not found: {model_path}. AI will not function.")
        return game, None, None
    mcts = MCTS(game, nnet, AI_ARGS)
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
        st.session_state.turn = PLAYER_PIECE
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.message = "Your turn, Red Player! Select a column."
        st.session_state.ai_thinking = False
        st.session_state.error_message = None
        st.session_state.game_ready = True
    st.session_state.game_restarted = False
    st.session_state.last_processed_timestamp = None

# --- Page Configuration (Main App Page) ---
st.set_page_config(page_title="Dominic Reilly's AlphaFour", layout="wide", initial_sidebar_state="collapsed")

# --- Initialize or process restart ---
if 'game_ready' not in st.session_state or st.session_state.get('game_restarted', False):
    initialize_game_state()

# --- Debug: Show bridge data at the top of the script execution ---
clicked_action_data = bridge("board_action_bridge", default=None, key="player_action_bridge_key")
print(f"DEBUG bridge (top of script): clicked_action_data = {clicked_action_data}, last_processed_timestamp = {st.session_state.get('last_processed_timestamp')}")

if clicked_action_data is not None:
    event_action_col = clicked_action_data.get("action_col")
    event_timestamp = clicked_action_data.get("timestamp")

    # Check if this is a new event based on the timestamp
    if event_timestamp is not None and event_timestamp != st.session_state.get('last_processed_timestamp'):
        print(f"DEBUG bridge: New event received. Timestamp: {event_timestamp}, Col: {event_action_col}")

        # Proceed only if it's the player's turn and game not over
        if not st.session_state.game_over and st.session_state.turn == PLAYER_PIECE:
            action_col = int(event_action_col) # Ensure it's an int for game logic
            print(f"DEBUG bridge: Processing action for column {action_col}")
            
            current_board = st.session_state.board
            game_instance = st.session_state.game
            player_piece_val = PLAYER_PIECE

            if game_instance.get_valid_moves(current_board)[action_col]:
                print(f"DEBUG bridge: Move in col {action_col} is valid. Getting next state.")
                new_board, _, move_row = game_instance.get_next_state(current_board, player_piece_val, action_col)
                st.session_state.board = new_board
                game_end_result = game_instance.get_game_ended(st.session_state.board, player_piece_val, last_move_col=action_col, last_move_row=move_row)
                if game_end_result != 0:
                    st.session_state.game_over = True
                    if game_end_result == 1: st.session_state.message = "You (Red) win! 🎉"
                    elif game_end_result == -1: st.session_state.message = "AI (Yellow) wins! 😞"
                    else: st.session_state.message = "It's a Draw! 🤝"
                else:
                    st.session_state.turn = AI_PIECE
                    st.session_state.message = "AI (Yellow) is thinking... 🤔"
                    st.session_state.ai_thinking = True
            else:
                print(f"DEBUG bridge: Move in col {action_col} is invalid.")
                st.session_state.message = "Invalid move attempted. Please click a valid column."
            
            st.session_state.last_processed_timestamp = event_timestamp # Update last processed timestamp
            print(f"DEBUG bridge: last_processed_timestamp updated to {event_timestamp}")
        else:
            print(f"DEBUG bridge: New event {event_timestamp} received, but not player's turn or game over. Storing timestamp to prevent re-processing.")
            st.session_state.last_processed_timestamp = event_timestamp
    elif event_timestamp is not None and event_timestamp == st.session_state.get('last_processed_timestamp'):
        print(f"DEBUG bridge: Stale event (timestamp matches last processed: {event_timestamp}). Col: {event_action_col}. Ignoring.")
    elif event_timestamp is None:
        print(f"DEBUG bridge: Event received without timestamp. Data: {clicked_action_data}. Ignoring.")

# --- CSS Styling --- 
# Using f-string to inject Python variables for piece sizes and colors

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    body, .stApp {{ font-family: 'Poppins', sans-serif; background: linear-gradient(to right, #232526, #414345); color: #f0f2f6; }}
    h1 {{ font-weight: 700; text-align: center; color: #ffffff; padding-top: 20px; padding-bottom: 10px; letter-spacing: 1px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
    .game-message {{ text-align: center; font-size: 1.5em; font-weight: 600; padding: 15px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.1); color: #ffffff; margin: 20px auto; width: fit-content; max-width: 80%; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }}
    
    .action-row {{
        display: grid; 
        position: relative; 
        z-index: 10; 
    }}
    .action-slot {{
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: transparent;
        height: {SQUARESIZE_HTML}px;
        /* cursor: default; /* REMOVE default cursor from base slot */
    }}

    /* Default style for all action pieces */
    .action-piece-visual {{
        width: {RADIUS_HTML * 2}px;
        height: {RADIUS_HTML * 2}px;
        border-radius: 50%;
        box-sizing: border-box; 
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease, border-color 0.2s ease, opacity 0.2s ease;
        background-color: rgba(255, 65, 54, 0.3);
        border: 2px solid rgba(255, 65, 54, 0.6);
        /* No cursor style here, parent slot should handle it */
    }}

    .action-slot-valid {{
        cursor: pointer !important; /* Ensure this applies */
    }}
    .action-slot-valid:hover .action-piece-visual {{
        background-color: {RED_PLAYER_HTML};
        background-image: none; 
        border-color: {RED_PLAYER_HTML};
        box-shadow: inset 0 -3px 5px rgba(0,0,0,0.3);
        transform: scale(1.05);
        opacity: 1.0;
    }}

    .action-slot-disabled {{
        cursor: not-allowed !important; /* Ensure this applies */
    }}
    .action-slot-disabled .action-piece-visual {{
        background-color: #707070; 
        background-image: none; 
        border-color: #505050;
        opacity: 0.4;
        /* No cursor style here, parent slot should handle it */
    }}

    .board-container {{
        position: relative;
        z-index: 1;
    }}

    .stButton[data-testid*="restart_game_main_btn"]>button {{ background-color: {RED_PLAYER_HTML} !important; }}
    .stButton[data-testid*="restart_game_main_btn"]>button:hover:not(:disabled) {{ background-color: #d13026 !important; }}
    .footer {{ text-align: center; padding: 20px; color: #aaa; font-size: 0.9em; }}

    div.st-key-player_action_bridge_key {{
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 0px !important;
        height: 0px !important;
        overflow: hidden !important;
        padding: 0px !important;
        margin: 0px !important;
        border: none !important;
        visibility: hidden !important;
        line-height: 0px !important; 
        font-size: 0px !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Main App Title ---
st.title("Dominic Reilly's AlphaFour Model!")

# --- Error handling for game initialization ---
if st.session_state.get('error_message'):
    st.error(st.session_state.error_message)
    if st.button("Try Re-initializing Game", key="reinit_err_btn"):
        st.session_state.game_restarted = True # Mark for reinitialization
        st.rerun()
    st.stop()

if not st.session_state.get("game_ready", False):
    st.error("Game could not be initialized. Please check model files or console.")
    if st.button("Retry Initialization", key="retry_init_btn"):
        st.session_state.game_restarted = True # Mark for reinitialization
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

# --- Main UI Display ---
st.markdown(f"<div class='game-message'>{st.session_state.message}</div>", unsafe_allow_html=True)

# Board display (NOW INCLUDES ACTION ROW)
# We need to pass valid_moves and game_over status to draw_board_html
valid_moves = game.get_valid_moves(board) # Get current valid moves
board_html_content = draw_board_html(board, board_cols, valid_moves, game_over, turn)
html(board_html_content) # New way using st_bridge.html

# --- AI's Turn Logic ---
if not game_over and turn == AI_PIECE and ai_thinking:
    if nnet is None or mcts is None:
        st.error("AI components not loaded. Cannot make a move.")
        st.session_state.message = "AI Error. Your turn."
        st.session_state.turn = PLAYER_PIECE
        st.session_state.ai_thinking = False
    else:
        time.sleep(0.75)
        canonical_board_ai = game.get_canonical_form(board, AI_PIECE)
        ai_action_probs = mcts.getActionProb(canonical_board_ai, temp=0) 
        ai_action = np.argmax(ai_action_probs)

        if not game.get_valid_moves(board)[ai_action]:
            st.warning(f"AI suggested invalid move {ai_action}. Picking first valid move.")
            valid_indices = [i for i, valid in enumerate(game.get_valid_moves(board)) if valid]
            if valid_indices: ai_action = valid_indices[0]
            else:
                st.session_state.game_over = True
                st.session_state.message = "Game ended: No valid AI moves."
                st.rerun() # Rerun to show game over
                st.stop() 

        new_board, _, ai_move_row = game.get_next_state(board, AI_PIECE, ai_action)
        st.session_state.board = new_board
        game_end_result = game.get_game_ended(st.session_state.board, AI_PIECE, last_move_col=ai_action, last_move_row=ai_move_row)
        
        if game_end_result != 0:
            st.session_state.game_over = True
            if game_end_result == 1: st.session_state.message = "AI (Yellow) wins! 😞"
            elif game_end_result == -1: st.session_state.message = "You (Red) win! 🎉"
            else: st.session_state.message = "It's a Draw! 🤝"
        else:
            st.session_state.turn = PLAYER_PIECE
            st.session_state.message = "Your turn, Red Player! Select a column."
    st.session_state.ai_thinking = False
    print("DEBUG AI: AI turn finished, about to call st.rerun()")
    st.rerun()

# --- Game Over Display and Restart ---
if game_over:
    if "You (Red) win!" in st.session_state.message: st.success(st.session_state.message)
    elif "AI (Yellow) wins!" in st.session_state.message: st.error(st.session_state.message) 
    else: st.info(st.session_state.message)
    
    if st.button("Restart Game", key="restart_game_main_btn"): # Key used for specific CSS if needed
        st.session_state.game_restarted = True
        print("DEBUG Restart: Restart button clicked, game_restarted set, about to call st.rerun()") # Console Debug
        st.rerun()

# --- Footer ---
st.markdown("<div class='footer'>Dominic Reilly's AlphaFour - Connect Four Streamlit Edition</div>", unsafe_allow_html=True)

# Debugging: Display session state (optional)
# with st.expander("Session State (Debug)"):
# st.write(st.session_state) 