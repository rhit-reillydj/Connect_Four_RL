import streamlit as st

# Moved st.set_page_config to the global scope
st.set_page_config(page_title="How AlphaFour Works - Dominic Reilly", layout="wide", initial_sidebar_state="collapsed")

def show_how_it_works_page():
    # --- Custom CSS for this page ---
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Base styling */
        body, .stApp { 
            font-family: 'Poppins', sans-serif; 
            background: linear-gradient(to right, #232526, #414345); 
            color: #f0f2f6; 
        }

        .main-title {
            font-size: clamp(2em, 6vw, 3em); /* Responsive */
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5em;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .sub-title {
            font-size: clamp(1em, 4vw, 1.5em); /* Responsive */
            font-weight: 300;
            text-align: center;
            margin-bottom: 2em;
            color: #cccccc;
        }

        .section {
            background-color: rgba(255, 255, 255, 0.05);
            padding: clamp(1em, 3vw, 2em); /* Responsive */
            border-radius: 15px;
            margin-bottom: clamp(1em, 3vw, 2em); /* Responsive */
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .section h2 {
            font-size: clamp(1.5em, 5vw, 2.2em); /* Responsive */
            font-weight: 600;
            color: #00a2ff; /* A vibrant accent blue */
            margin-bottom: 0.7em;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #00a2ff;
            display: flex; /* Changed from inline-block to flex */
            align-items: center; /* Align icon and text */
        }
        
        .section h3 {
            font-size: clamp(1.2em, 4vw, 1.6em); /* Responsive */
            font-weight: 600;
            color: #f0f2f6;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }

        .section p, .section li {
            font-size: clamp(0.9em, 2.5vw, 1.1em); /* Responsive */
            line-height: 1.8;
            color: #d0d0d0;
            margin-bottom: 1em;
        }
        
        .section strong {
            color: #ffffff;
            font-weight: 600;
        }

        .icon { /* Placeholder for icons */
            font-size: clamp(1.8em, 4vw, 2.5em); /* Responsive */
            margin-right: 0.5em;
            vertical-align: middle;
            color: #00a2ff;
        }
        
        /* Responsive columns */
        .st-emotion-cache-1r6slb0 { /* This targets st.columns generated divs, might need inspection if class changes */
            gap: 2em;
        }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>Understanding AlphaFour</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>The Intelligence Behind Dominic Reilly\'s Connect Four Champion</div>", unsafe_allow_html=True)

    # --- Introduction ---
    with st.container():
        st.markdown("""
        <div class='section'>
            <h2><span class='icon'>🚀</span>The AlphaFour Odyssey</h2>
            <p>Welcome to a deep dive into <strong>Dominic Reilly\'s AlphaFour</strong>! This project takes inspiration from groundbreaking AI like DeepMind\'s AlphaGo and AlphaZero, aiming to create a formidable Connect Four opponent that learns and evolves through sophisticated AI techniques. AlphaFour isn\'t just about playing a game; it\'s an exploration into the fascinating world of artificial intelligence, neural networks, and strategic decision-making.</p>
            <p>Our journey involves a blend of cutting-edge concepts, each playing a crucial role in enabling AlphaFour to master the aparentemente simple, yet deeply strategic, game of Connect Four.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Core Components ---
    st.markdown("<hr style='border-top: 2px solid #00a2ff; margin: 3em 0;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-weight: 600; font-size: clamp(1.8em, 5.5vw, 2.5em);'>Core Components of AlphaFour</h1>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    with cols[0]:
        # --- CNN Section ---
        with st.container():
            st.markdown("""
            <div class='section'>
                <h2><span class='icon'>👁️</span>The Neural Eye: CNN</h2>
                <p>At the heart of AlphaFour\'s ability to evaluate board positions is a <strong>Convolutional Neural Network (CNN)</strong>. Traditionally used for image recognition, a CNN is adept at identifying patterns. For AlphaFour, the game board is like an image, and the CNN learns to:</p>
                <ul>
                    <li>Recognize crucial patterns, threats, and opportunities.</li>
                    <li>Estimate the likelihood of winning from the current position (the <strong>value</strong> of the state).</li>
                    <li>Suggest promising moves to explore (the <strong>policy</strong>).</li>
                </ul>
                <p>This \'vision\' allows AlphaFour to make informed judgments that go beyond simple rule-based heuristics.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.image("path/to/cnn_diagram.png", caption="Simplified CNN Architecture") # Placeholder

        # --- MCTS Section ---
        with st.container():
            st.markdown("""
            <div class='section'>
                <h2><span class='icon'>🌳</span>Strategic Foresight: MCTS</h2>
                <p>While the CNN provides an \'intuition\', <strong>Monte Carlo Tree Search (MCTS)</strong> provides the \'reasoning\'. MCTS is a powerful search algorithm that AlphaFour uses to:</p>
                <ul>
                    <li>Simulate thousands of possible game continuations from the current state.</li>
                    <li>Build a game tree, exploring promising branches guided by the CNN\'s policy and value estimates.</li>
                    <li>Balance exploring new, uncertain moves (<strong>exploration</strong>) with focusing on moves that have historically led to good outcomes (<strong>exploitation</strong>).</li>
                </ul>
                <p>By the end of its \'thinking\' time, MCTS selects the move that has proven most robust across its simulations, effectively allowing AlphaFour to \'look ahead\' many steps.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.image("path/to/mcts_diagram.png", caption="Simplified MCTS Process") # Placeholder
            
    with cols[1]:
        # --- DRL Section ---
        with st.container():
            st.markdown("""
            <div class='section'>
                <h2><span class='icon'>🧠</span>The Learning Engine: DRL</h2>
                <p><strong>Deep Reinforcement Learning (DRL)</strong> is the philosophy that drives AlphaFour\'s learning. It combines Deep Learning (the \'Deep\' from our CNN) with Reinforcement Learning. The core idea is:</p>
                <ul>
                    <li>The AI (or \'agent\') learns by interacting with its environment (the Connect Four game).</li>
                    <li>It receives <strong>rewards</strong> (for winning) or <strong>penalties</strong> (for losing or drawing).</li>
                    <li>Over time, the agent adjusts its internal neural network (the CNN) to maximize these rewards, effectively learning to play better.</li>
                </ul>
                <p>This is a powerful paradigm, as the AI discovers strategies on its own, rather than being explicitly programmed with them.</p>
            </div>
            """, unsafe_allow_html=True)

        # --- Self-Play Section ---
        with st.container():
            st.markdown("""
            <div class='section'>
                <h2><span class='icon'>🔄</span>The Training Loop: Self-Play</h2>
                <p>Inspired by AlphaGo Zero, AlphaFour primarily learns through <strong>self-play</strong>. This is a continuous improvement cycle:</p>
                <ol>
                    <li><strong>Play:</strong> The current best version of the AlphaFour model plays thousands of games against itself. The MCTS algorithm guides its move choices during these games.</li>
                    <li><strong>Generate Data:</strong> Each move made and the eventual outcome of these games (win/loss/draw) are recorded as training data. This data teaches the model which board states are valuable and which moves are strong.</li>
                    <li><strong>Train:</strong> The neural network (CNN) is then trained on this newly generated data. It learns to better predict move probabilities (policy) and game outcomes (value) from any given board state.</li>
                    <li><strong>Evaluate & Iterate:</strong> The newly trained model is pitted against the previous best version. If it performs significantly better, it becomes the new champion, and the cycle repeats.</li>
                </ol>
                <p>This relentless process allows AlphaFour to bootstrap its intelligence, starting from random play (or a very basic understanding) and progressively becoming a highly skilled player.</p>
            </div>
            """, unsafe_allow_html=True)
            # st.image("path/to/self_play_loop.png", caption="The AlphaGo-inspired Training Loop") # Placeholder

    # --- Conclusion ---
    st.markdown("<hr style='border-top: 2px solid #00a2ff; margin: 3em 0;'>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
        <div class='section' style='text-align:center;'>
            <h2><span class='icon'>💡</span>Synergy in AlphaFour</h2>
            <p>These components don\'t work in isolation. The CNN guides the MCTS search, MCTS provides strong move choices for self-play, self-play generates rich data for DRL, and DRL refines the CNN. It\'s this beautiful synergy, pioneered by systems like AlphaGo, that Dominic Reilly\'s AlphaFour aims to capture for mastering Connect Four.</p>
            <p>Thank you for exploring the mind of AlphaFour!</p>
        </div>
        """, unsafe_allow_html=True)

# Call the function to render the page content
show_how_it_works_page()

# The if __name__ == "__main__": block is not strictly necessary for Streamlit pages
# but calling the main function like this is a common pattern.
# if __name__ == "__main__":
# show_how_it_works_page() 