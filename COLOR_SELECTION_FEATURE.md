# Connect Four Color Selection Feature

## Overview
This feature allows players to choose their color (red or yellow) in the Connect Four game against the AI. The implementation follows these rules:
- Every game: Player chooses their color via a popup
- Red pieces always make the first move
- Yellow pieces always go second

## Changes Made

### 1. Session State Variables Added
- `player_color`: Stores the human player's color (PLAYER_PIECE=1 for red, AI_PIECE=-1 for yellow)
- `color_selected`: Boolean flag indicating if color has been chosen (reset to False on restart)
- `last_color_selection_timestamp`: Timestamp for color selection events

### 2. New Functions

#### `display_color_selection()`
- Displays a fullscreen overlay popup with two buttons
- Red button: "Play First (Red)" 
- Yellow button: "Play Second (Yellow)"
- Uses st_bridge to communicate selection back to Streamlit

### 3. Color Selection Logic

#### Every Game
- In `initialize_game_state()`, `color_selected` is reset to False
- This causes the color selection popup to appear
- Player clicks either red or yellow button
- Selection is processed via bridge listener
- Game initializes with appropriate turns

### 4. UI Updates

#### Dynamic Hover Colors
- Hover effect shows the human player's color
- CSS variable `--human-hover-color` updates based on `player_color`
- Action piece visual colors update dynamically

#### Turn Logic
- `draw_board_html()` checks if it's human's turn regardless of color
- Player move handling works for both red and yellow human players
- AI turn logic adapted to handle playing as either color

#### Messages
- All game messages now show actual color names (Red/Yellow)
- Win/loss messages correctly identify winner by color

#### Winner Detection
- Fixed to properly interpret `get_game_ended` result
- Result is from perspective of the player who just moved
- 1 means that player won, -1 means opponent won
- Properly sets `st.session_state.winner` to actual color value

### 5. CSS Styling

#### Color Selection Overlay
- Fullscreen overlay with gradient background
- Responsive button design with hover effects
- Red and yellow themed buttons with emoji indicators

#### Bridge Component Hiding
- Added `color_selection_bridge_key` to hidden elements list

## Technical Implementation Details

### Turn Management
- Red (PLAYER_PIECE = 1) always goes first
- Yellow (AI_PIECE = -1) always goes second
- Turn switching uses `-human_color` to get opponent

### AI Adaptation
- AI can play as either red or yellow
- Uses `canonical_form` to always see board from its perspective
- MCTS works correctly regardless of AI color

### Edge Cases Handled
- First move when human is yellow (AI goes first)
- Color selection appears every game
- Draw scenarios properly detected

## Testing Checklist
- [x] Color selection popup appears every game
- [x] Selecting red allows human to go first
- [x] Selecting yellow makes AI go first
- [x] Hover colors match human's chosen color
- [x] Win detection works correctly for both colors
- [x] All messages show correct color names
- [x] Restart button triggers new color selection