import time

import streamlit as st
import requests
import streamlit_javascript

chat_history = []


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

def get_client_ip():
    url = 'https://api.ipify.org?format=json'
    script = (f'await fetch("{url}"). then('
              f'function(response) {{'
              f'return response.json();'
              f'}})')
    try:
        result = streamlit_javascript.st_javascript(script)
        if isinstance(result, dict) and 'ip' in result:
            return result['ip']
    except:
        pass
    return None


def main():
    st.set_page_config(page_title="BEES Chat", page_icon=":books:")
    st.title("BEES CHATBOT")
    ip = get_client_ip()
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if human := st.chat_input():
        st.session_state.chat_history.append(human)
        api_response = call_api(human, ip, st.session_state.session_id)
        if api_response.get("session_id"):
            st.session_state.session_id = api_response.get("session_id")
        if api_response.get("source"):
            if "https" not in api_response.get("source"):
                url = str(api_response.get("source")).replace(' ', '%20%')
                chatbot_response = str(api_response.get(
                    "response")) + "\n\nsource - " + f"https://biologicale.blob.core.windows.net/beesfiles{url}"
            else:
                chatbot_response = str(api_response.get("response")) + "\n\nsource - " + str(api_response.get("source"))
        else:
            chatbot_response = str(api_response.get("response"))
        st.session_state.chat_history.append(chatbot_response)

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message('user').write(msg)
        else:
            if "<table>" in msg:
                st.markdown(msg, unsafe_allow_html=True)
            else:
                st.chat_message('assistant').markdown(f'{msg}')


if __name__ == "__main__":
    main()
