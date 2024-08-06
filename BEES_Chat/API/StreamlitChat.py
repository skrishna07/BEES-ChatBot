import time

import streamlit as st
import requests
from SourceCode.GetSessionDetails import get_ip_related_details
from SourceCode.GetSessionDetails import get_session_details
import warnings
from datetime import datetime, timedelta
from SourceCode.CSS import css_selector
from SourceCode.CSS import custom_js
# Current date
from streamlit_javascript import st_javascript
import streamlit.components.v1 as components
current_date = datetime.now().strftime('%Y-%m-%d')

# Previous date (yesterday) in yyyy-mm-dd format
previous_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
warnings.filterwarnings("ignore", category=DeprecationWarning)




def call_api(user_message, ip, session_id):
    # Mockoon API endpoint (replace with your actual URL)
    api_url = "http://74.225.252.130:80/api/chat"

    # Send POST request with user message as data
    data = {"query": user_message, "ip_address": ip,
            "session_id": session_id}
    headers = {"Authorization": "token f264b428d4e998383a15e667fb4050a49e9b2dc7"}
    response = requests.post(api_url, json=data, headers=headers)
    # Check for successful response
    if response.status_code == 200:
        return response.json()  # Extract response message
    else:
        return "API Error: " + str(response.status_code)


def get_ip():
    url = 'https://api.ipify.org?format=json'
    script = (f'await fetch("{url}"). then('
              f'function(response) {{'
              f'return response.json();'
              f'}})')
    try:
        result = st_javascript(script)
        if isinstance(result, dict) and 'ip' in result:
            return result['ip']
    except:
        pass
    return None


def main():
    st.set_page_config(page_title="BEES Chat", page_icon=":books:")
    ip = get_ip()
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.new_chat = True
    # if 'new_chat' not in st.session_state:
    #     st.session_state.new_chat = False
    if not st.session_state.new_chat:
        st.session_state.chat_history = []
    if "link_clicked" not in st.session_state:
        st.session_state.link_clicked = False
    try:
        chat_history_json_response = get_ip_related_details(ip)
    except:
        chat_history_json_response = []
    st.markdown(css_selector,unsafe_allow_html=True)
    st_javascript(custom_js)
    base_url = st_javascript('window.location.origin')
    if st.sidebar.button('New Chat',use_container_width=True):
        st.query_params.clear()
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.new_chat = True
    today_buttons = []
    yesterday_buttons = []
    other_buttons = []
    st.sidebar.title("Chat Sessions")
    if chat_history_json_response is not None:
        if len(chat_history_json_response) != 0:
            for session in chat_history_json_response:
                converted_date = datetime.strptime(session["datetime"], '%Y-%m-%d %H:%M:%S.%f').strftime('%Y-%m-%d')
                if converted_date == current_date:
                    today_buttons.append(session)
                elif converted_date == previous_date:
                    yesterday_buttons.append(session)
                else:
                    other_buttons.append(session)
            if len(today_buttons) != 0:
                st.sidebar.markdown("<h3 style='text-align: left;'>Today</h3>", unsafe_allow_html=True)
                for today_button in today_buttons:
                    url = f'{base_url}/?session_id={today_button["session_id"]}'
                    st.sidebar.markdown(f'''
                           <a href="{url}" target = "_self" onclick="highlightButton('{today_button["session_id"]}');">
                               <button id="{today_button["session_id"]}" class="button">{today_button["responses"][0]["query"]}</button>
                           </a>
        
                       ''', unsafe_allow_html=True)
            if len(yesterday_buttons) != 0:
                st.sidebar.markdown("<h3 style='text-align: left;'>Yesterday</h3>", unsafe_allow_html=True)
                for yesterday_button in yesterday_buttons:
                    url = f'{base_url}/?session_id={yesterday_button["session_id"]}'
                    st.sidebar.markdown(f'''
                                              <a href="{url}" target = "_self" onclick="highlightButton('{yesterday_button["session_id"]}');return false;">
                                                  <button id="{yesterday_button["session_id"]}" class="button">{yesterday_button["responses"][0]["query"]}</button>
                                              </a>
                                          ''', unsafe_allow_html=True)
            if len(other_buttons) != 0:
                st.sidebar.markdown("<h3 style='text-align: left;'>Older</h3>", unsafe_allow_html=True)
                for other_button in other_buttons:
                    url = f'{base_url}/?session_id={other_button["session_id"]}'
                    st.sidebar.markdown(f'''
                                                                  <a href="{url}" target = "_self" onclick="highlightButton('{other_button["session_id"]}');return false;">
                                                                      <button id="{other_button["session_id"]}" class="button">{other_button["responses"][0]["query"]}</button>
                                                                  </a>
                                                              ''', unsafe_allow_html=True)
            query_params = st.query_params
            print(query_params)
            selected_session_id = query_params.get("session_id", None)
            if selected_session_id is not None:
                st.session_state.new_chat = False
                st.session_state.session_id = selected_session_id
                # css_style = f"""$('#' {st.session_state.session_id}).style('background-color:red')"""
                # st_javascript(css_style)
                st_javascript(f"""
                    alert("Ok")
                        document.getElementById("{st.session_state.session_id}").style.backgroundColor = "Grey";
                """)
                session_response_json = get_session_details(selected_session_id)
                print(session_response_json)
                if session_response_json is not None:
                    for output_response in session_response_json['responses']:
                        query = output_response['query']
                        source = output_response['source']
                        if "https" not in source:
                            url = str(source).replace(' ', '%20%')
                            output_answer = str(output_response['response']) + "\n\nSource - " + f"https://biologicale.blob.core.windows.net/beesfiles{url}"
                        else:
                            output_answer = str(output_response['response']) + "\n\nSource - " + str(source)
                        st.session_state.chat_history.append(query)
                        st.session_state.chat_history.append(output_answer)
    if human := st.chat_input():
        st.session_state.chat_history.append(human)
        api_response = call_api(human, ip, st.session_state.session_id)
        if api_response.get("session_id"):
            st.session_state.session_id = api_response.get("session_id")
        if api_response.get("source"):
            if "https" not in api_response.get("source"):
                url = str(api_response.get("source")).replace(' ', '%20%')
                chatbot_response = str(api_response.get(
                    "response")) + "\n\nSource - " + f"https://biologicale.blob.core.windows.net/beesfiles{url}"
            else:
                chatbot_response = str(api_response.get("response")) + "\n\nSource - " + str(api_response.get("source"))
        else:
            chatbot_response = api_response.get("response")

        st.session_state.chat_history.append(chatbot_response)
        st.rerun()
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # st.chat_message('user').write(msg)
            st.markdown(f'<div class="question">{msg}</div>', unsafe_allow_html=True)
            st.write('\n')
        else:
            # st.chat_message('assistant').write(msg)
            # st.markdown(f'<div class="answer">{msg}</div>', unsafe_allow_html=True)
            st.markdown(f"""
                        <div class="answer-container">
                            <img src="https://img.freepik.com/free-vector/chatbot-chat-message-vectorart_78370-4104.jpg?t=st=1721201561~exp=1721205161~hmac=5e8766c7d7380af200994280265c68031c3715fbe120a7432d6c498caff0cb58&w=826" alt="Logo" height="30">
                            <div class="answer">{msg}</div>
                        </div>
                    """, unsafe_allow_html=True)
            # button_html = f'<button id="{i}">Copy to Clipboard</button>'
            # st.markdown(button_html, unsafe_allow_html=True)
            # script = f"""
            # <script>
            # document.getElementById('{i}').addEventListener('click', function() {{
            #     var textToCopy = '{msg}';
            #
            #     // Create a temporary textarea element
            #     var textarea = document.createElement('textarea');
            #     textarea.value = textToCopy;
            #     textarea.setAttribute('readonly', '');
            #     textarea.style.position = 'absolute';
            #     textarea.style.left = '-9999px'; // Move the textarea off-screen
            #     document.body.appendChild(textarea);
            #
            #     // Select and copy the text inside the textarea
            #     textarea.select();
            #     textarea.setSelectionRange(0, textarea.value.length);
            #     document.execCommand('copy');
            #
            #     // Clean up
            #     document.body.removeChild(textarea);
            #
            #     // Alert the user
            #     alert('Text copied to clipboard: ' + textToCopy);
            # }});
            # </script>
            # """
            #
            # # Render the JavaScript
            # st.markdown(script, unsafe_allow_html=True)
            st.write('\n')
        st.write('\n')

if __name__ == "__main__":
    main()
