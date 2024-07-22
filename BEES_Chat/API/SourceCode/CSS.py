css_selector = """
        <style>
        .stButton button {
            text-align: left;
            padding:8px 80px 5px;
            margin-bottom: 0px;
        }
        button {
            border: none;
            background: none;
            text-align: left;
            padding:10px 2px 3px;
            margin-bottom: 10px;
            width: 100%;
        }
        button[data-testid="baseButton-secondary"] p {
    text-align: left; /* Align text inside <p> to the left */
    margin: 0; /* Ensure no extra margin */
    padding: 0; /* Ensure no extra padding */
}
.copy-button {
        margin-left: 10px;
        padding: 8px 16px;
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 4px;
        cursor: pointer;
    }
.st-emotion-cache-1rci6ej {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: flex-start;
    -webkit-box-pack: start;
    justify-content: start;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: 100%;
    user-select: none;
    }
    .highlighted-button {
    background: #FFD700; /* Highlight color (example: gold) */
    color: #000000; /* Text color */
    border-color: #FFD700; /* Border color (optional) */
    /* Add any other styles for highlighted state */
}
.sidebar .sidebar-content {
    position: fixed;
    bottom: 20px; /* Adjust as needed */
    width: 100%; /* Ensure full width */
    z-index: 999; /* Ensure it's above other content */
}
:host, html {
    -webkit-text-size-adjust: 100%;
    font-feature-settings: normal;
    -webkit-tap-highlight-color: transparent;
    font-family: ui-sans-serif, -apple-system, system-ui, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif, Helvetica, Apple Color Emoji, Arial, Segoe UI Emoji, Segoe UI Symbol;
    font-variation-settings: normal;
    line-height: 1.5;
    tab-size: 4;
    font-size: 18px;
}
        .question-answer-container {
            display: flex;
            flex-direction: column-reverse;
            align-items: flex-start;
        }
        .question {
              background-color: #E0E0E0;
              font-size: 18px;
       padding-bottom: .625rem;
        padding-top: .625rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        border-radius: 1.5rem;
        margin: 5px 5px 5px auto;
        max-width: FIT-CONTENT;
        text-align: right;
        }
        .answer {
            max-width:  FIT-CONTENT;
            margin-bottom: 1.25em;
        }
        .answer-container {
        display: flex;
        max-width: FIT-CONTENT;
        font-size: 18px;
    }
    .loading {
        display: flex;
        align-items: center;
        font-size: 18px;
    }
    .loading-spinner {
        margin-right: 10px;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .st-bz {
            height: 50px !important;
        }
    .button-highlight {{
        background-color: lightblue; /* Highlight color */
        color: black; /* Text color */
    }}
    p, ol, ul, dl {
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 18px;
    font-weight: 400;
}

        </style>
    """

custom_js_1 = """
<script>
// Function to handle button clicks
function handleButtonClick(event) {
    // Remove 'highlighted-button' class from all buttons
    var buttons = document.querySelectorAll('.stButton button');
    buttons.forEach(function(button) {
        button.classList.remove('highlighted-button');
    });
    // Add 'highlighted-button' class to the clicked button
    event.target.classList.add('highlighted-button');
}

// Attach the handleButtonClick function to all Streamlit buttons
document.addEventListener('DOMContentLoaded', function() {
    var buttons = document.querySelectorAll('.stButton button');
    buttons.forEach(function(button) {
        button.addEventListener('click', handleButtonClick);
    });
});
function highlightButton(buttonId) {{
        // Remove the highlight class from all buttons
        document.querySelectorAll('.button').forEach(button => {
            button.classList.remove('button-highlight');
        });
        // Add the highlight class to the clicked button
        document.getElementById(buttonId).classList.add('button-highlight');
        // Optionally set a cookie or local storage to persist the highlight across page reloads
        localStorage.setItem('highlightedButton', buttonId);
    }}
// On page load, highlight the button from localStorage if present
window.onload = function() {{
const highlightedButtonId = localStorage.getItem('highlightedButton');
if (highlightedButtonId) {{
    document.getElementById(highlightedButtonId)?.classList.add('button-highlight');
        }}
}};
</script>
"""

custom_js = """
<script>
function highlightButton(buttonId) {{
        alert("66") 
        // Remove the highlight class from all buttons
        document.querySelectorAll('.button').forEach(button => {
            button.classList.remove('button-highlight');
        });
        // Add the highlight class to the clicked button
        document.getElementById(buttonId).classList.add('button-highlight');
        // Optionally set a cookie or local storage to persist the highlight across page reloads
        localStorage.setItem('highlightedButton', buttonId);
    }}
}};
</script>"""