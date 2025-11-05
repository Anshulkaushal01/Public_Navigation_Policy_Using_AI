import streamlit as st
from datetime import datetime
import ollama
import easyocr
from PIL import Image
import io
import numpy as np

# =========================
# 📌 OCR SETUP (PaddleOCR)
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes

ocr = PaddleOCR(use_angle_cls=True, lang='en')


# =========================
# 📌 PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# 📌 CUSTOM CSS (Theme & Styling)
# =========================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4B0082;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        color: #0d47a1;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #FFF3E0;
        border: 1px solid #FFD54F;
        color: #BF360C;
        margin-right: auto;
    }
    .sidebar-chat {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        cursor: pointer;
        border: 1px solid #B39DDB;
        color: #4A148C;
    }
    .sidebar-chat:hover {
        background-color: #D1C4E9;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #B39DDB;
        padding: 10px 20px;
        color: #4A148C;
    }
    div[style*='text-align: center; color: #666;'] {
        color: #4A148C;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 📌 AI RESPONSE FUNCTION
# =========================
def get_ai_response(prompt, model="tinydolphin"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        st.error(f"⚠️ Ollama se connect nahi ho pa raha: {e}")
        return None

# =========================
# 📌 SESSION STATE INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = 0
if "last_extracted_text" not in st.session_state:
    st.session_state.last_extracted_text = None  # ✅ store OCR text

# =========================
# 📌 SIDEBAR (Chat History)
# =========================
st.sidebar.title("💬 Chat History")

# ➕ New Chat
if st.sidebar.button("➕ New Chat", use_container_width=True):
    if st.session_state.messages:
        first_text = next((msg['content'] for msg in st.session_state.messages if msg.get('type') != 'image'), "Chat")
        chat_title = first_text[:30] + "..."
        st.session_state.chat_history.append({
            "id": st.session_state.current_chat_id,
            "title": chat_title,
            "messages": st.session_state.messages.copy(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
    st.session_state.messages = []
    st.session_state.last_extracted_text = None
    st.session_state.current_chat_id += 1
    st.rerun()

# Previous Chats
if st.session_state.chat_history:
    st.sidebar.subheader("Previous Chats")
    for chat in reversed(st.session_state.chat_history[-10:]):
        if st.sidebar.button(f"💬 {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
            st.session_state.messages = chat["messages"].copy()
            st.session_state.last_extracted_text = None
            st.rerun()

# Clear All Chats
if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.last_extracted_text = None
    st.rerun()

# =========================
# 📌 MAIN HEADER
# =========================
st.markdown('<h1 class="main-header">🤖 AI Chatbot</h1>', unsafe_allow_html=True)

# =========================
# 📌 DISPLAY CHAT MESSAGES
# =========================
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image":
                # ✅ Show smaller image (fixed width)
                st.image(message["data"], caption=message.get("caption", "Uploaded Image"), width=300)
            if "content" in message and message["content"]:
                st.markdown(message["content"])

# =========================
# 📌 INPUT & FILE UPLOADER
# =========================
with st.container():
    uploaded_file = st.file_uploader(
        "File upload karein (optional)",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        label_visibility="collapsed"
    )
    prompt = st.chat_input("Type your message here...")

# =========================
# 📌 FILE HANDLING (Image / PDF)
# =========================
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    # 🖼️ IMAGE OCR
    if uploaded_file.type.startswith('image/'):
        image = Image.open(io.BytesIO(file_bytes))
        result = ocr.ocr(np.array(image))
        extracted_text = "\n".join([line[1][0] for line in result[0]]).strip()

        if extracted_text:
            st.session_state.last_extracted_text = extracted_text
            st.session_state.messages.append({
                "role": "user",
                "content": f"🖼️ Extracted text from image:\n\n{extracted_text}"
            })
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": "⚠️ Image se koi text extract nahi ho paaya."
            })

        st.session_state.messages.append({
            "role": "user",
            "type": "image",
            "data": file_bytes,
            "caption": uploaded_file.name
        })

    # 📄 PDF Handling
    elif uploaded_file.type == "application/pdf":
        images = convert_from_bytes(file_bytes)
        extracted_text_list = []
        for img in images:
            result = ocr.ocr(np.array(img))
            extracted_text_list.extend([line[1][0] for line in result[0]])
        extracted_text = "\n".join(extracted_text_list).strip()

        if extracted_text:
            st.session_state.last_extracted_text = extracted_text
            st.session_state.messages.append({
                "role": "user",
                "content": f"📄 Extracted text from PDF:\n\n{extracted_text[:2000]}..."  # Limit preview
            })
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": "⚠️ PDF se text extract nahi ho paaya."
            })


# =========================
# 📌 HANDLE USER PROMPT
# =========================
if prompt:
    # Merge OCR + Prompt if available
    if st.session_state.last_extracted_text:
        final_prompt = f"Image se yeh text extract hua hai:\n\n{st.session_state.last_extracted_text}\n\nUser ka sawaal hai: {prompt}\n\nExtracted text aur question dono ko use karke jawab do."
    else:
        final_prompt = prompt

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # AI Response
    response = get_ai_response(final_prompt, model="tinydolphin")
    if response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    st.rerun()

# =========================
# 📌 FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    ChatGPT Clone built with Streamlit • Powered by Ollama 🚀
</div>
""", unsafe_allow_html=True)
