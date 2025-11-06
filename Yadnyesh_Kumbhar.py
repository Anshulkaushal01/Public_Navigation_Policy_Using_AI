import streamlit as st
import json
import os
import hashlib
from datetime import datetime
import uuid
import subprocess
import tempfile
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import ollama

# ===================== CONFIG =====================
CHAT_FILE = "chats.json"
USER_FILE = "users.json"
MODEL_NAME = "llama3.2"  # Change if you want another Ollama model

# ===================== SESSION STATE =====================
def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = None

def load_chats():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHAT_FILE, "w") as f:
        json.dump(chats, f, indent=2)

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# ===================== AUTHENTICATION =====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username, password, email):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.now().isoformat()
    }
    save_users(users)
    return True, "User created successfully"

def log_in(username, password):
    users = load_users()
    if username not in users:
        return False, "Username not found"
    if users[username]["password"] != hash_password(password):
        return False, "Incorrect password"
    st.session_state.authenticated = True
    st.session_state.current_user = username
    return True, "Login successful"

# ===================== CHAT MANAGEMENT =====================
def generate_chat_id():
    return str(uuid.uuid4())

def create_new_chat(title="New Chat"):
    user = st.session_state.current_user
    chat_id = generate_chat_id()
    if user not in st.session_state.chats:
        st.session_state.chats[user] = {}
    st.session_state.chats[user][chat_id] = {
        "title": title,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    save_chats(st.session_state.chats)
    return chat_id

def delete_chat(chat_id):
    user = st.session_state.current_user
    if user in st.session_state.chats and chat_id in st.session_state.chats[user]:
        del st.session_state.chats[user][chat_id]
        save_chats(st.session_state.chats)
        return True
    return False

def add_message_to_chat(chat_id, role, content):
    user = st.session_state.current_user
    chat = st.session_state.chats[user][chat_id]
    chat["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    chat["updated_at"] = datetime.now().isoformat()
    save_chats(st.session_state.chats)

# ===================== OCR FUNCTION =====================
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

# ===================== OLLAMA AI =====================
def generate_ai_response(prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error contacting Ollama: {e}"

# ===================== STREAMLIT UI =====================
init_session_state()

st.set_page_config(page_title="Chatbot with OCR & Ollama", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("🤖 Chatbot with OCR + Ollama")

    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                ok, msg = log_in(username, password)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with tab2:
            with tab2:
                new_user = st.text_input("New Username", key="signup_username")
                new_email = st.text_input("Email", key="signup_email")
                new_pass = st.text_input("Password", type="password", key="signup_password")
                confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")

            if st.button("Create Account"):
                if new_pass != confirm:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    ok, msg = sign_up(new_user, new_pass, new_email)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    else:
        st.success(f"Logged in as {st.session_state.current_user}")
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.current_chat = create_new_chat()
            st.rerun()

        # Show chats
        user_chats = st.session_state.chats.get(st.session_state.current_user, {})
        for cid, chat in sorted(user_chats.items(), key=lambda x: x[1]["updated_at"], reverse=True):
            if st.button(chat["title"], key=cid):
                st.session_state.current_chat = cid
                st.rerun()

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()

if st.session_state.authenticated:
    if st.session_state.current_chat:
        chat = st.session_state.chats[st.session_state.current_user][st.session_state.current_chat]
        st.header(chat["title"])

        for msg in chat["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        st.divider()

        uploaded_file = st.file_uploader("Upload Image (for OCR)", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            text = extract_text_from_image(uploaded_file)
            st.info(f"Extracted Text:\n\n{text}")

        user_input = st.chat_input("Type your message here or use OCR text above...")

        if user_input:
            add_message_to_chat(st.session_state.current_chat, "user", user_input)
            with st.spinner("Thinking..."):
                reply = generate_ai_response(user_input)
            add_message_to_chat(st.session_state.current_chat, "assistant", reply)
            st.rerun()

    else:
        st.header("Welcome to Chatbot with OCR & Ollama")
        st.info("Start a new chat or select one from the sidebar.")
else:
    st.header("Welcome 👋")
    st.markdown("""
    Please log in or sign up using the sidebar to start chatting.
    You can also upload images to extract text using OCR!
    """)