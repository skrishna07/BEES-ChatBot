import time

import streamlit as st
import requests

chat_history = []


def call_api(user_message, ip, session_id):
    # Mockoon API endpoint (replace with your actual URL)
    api_url = "http://74.225.252.130:80/api/chat"

    # Send POST request with user message as data
    data = {"query": user_message, "ip_address": ip,
            "session_id": session_id}
    headers = {"Authorization": "token 616a306d17080c0fdbe9e7fefccf5641c838754b"}
    response = requests.post(api_url, json=data, headers=headers)
    # Check for successful response
    if response.status_code == 200:
        return response.json()  # Extract response message
    else:
        return "API Error: " + str(response.status_code)


def get_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except requests.RequestException as e:
        st.error(f"Error fetching IP address: {e}")
        return None


def main():
    st.set_page_config(page_title="BEES Chat", page_icon=":books:")
    ip = get_ip()
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if human := st.chat_input():
        st.session_state.chat_history.append(human)
        api_response = call_api(human, ip, st.session_state.session_id)
        if api_response.get("session_id"):
            st.session_state.session_id = api_response.get("session_id")

        chatbot_response = api_response.get("response")

        st.session_state.chat_history.append(chatbot_response)

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message('user').write(msg)
        else:
            st.chat_message('assistant').write(msg)


if __name__ == "__main__":
    main()
