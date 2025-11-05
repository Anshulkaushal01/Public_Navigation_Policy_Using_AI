import streamlit as st
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from textblob import TextBlob
import requests
import json
import time
from datetime import datetime
import os

# -----------------------------------------------------------
# 🎨 Custom CSS for Modern UI
# -----------------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%);
    padding: 20px;
}

.stChatMessage {
    padding: 15px 20px;
    border-radius: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    max-width: 85%;
    animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.stChatMessage[data-testid="stChatMessage"] > div:first-child {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}

.stChatMessage[data-testid="stChatMessage"] > div:last-child {
    background-color: #f8f9fa;
    border-right: 4px solid #6c757d;
}

.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #4a6fa5, #2c3e50);
    color: white;
}

.stFileUploader {
    border: 2px dashed #4a6fa5;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #2196f3;
    background-color: rgba(74, 111, 165, 0.05);
}

.sentiment-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    margin: 8px 0;
    font-size: 14px;
}
.positive { background-color: #e8f5e9; color: #2e7d32; }
.negative { background-color: #ffebee; color: #c62828; }
.neutral { background-color: #e0e0e0; color: #616161; }

.typing-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #6c757d;
    margin: 0 3px;
    animation: typing 1.4s infinite;
}
.typing-indicator:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.full-text-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧭 Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Document Chatbot 🤖", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 AI Document Assistant with OCR & PDF Analysis")

# -----------------------------------------------------------
# ⚙️ Sidebar Settings
# -----------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712107.png", width=100)
    st.header("⚙️ Settings")
    model = st.selectbox(
        "Choose LLaMA 3 Model", 
        ["llama3", "llama3:8b", "llama3:70b"],
        help="Select the model size for better responses"
    )
    st.divider()
    st.info("💡 **How to use:**\n1. Upload an image/PDF\n2. Ask questions\n3. Download full report!")
    st.markdown("### 🌟 Pro Tips\n- Use clear images for better OCR\n- Ask specific questions\n- Try 'Summarize this' or 'Explain key points'")

# -----------------------------------------------------------
# 🧠 Configure Tesseract Path (Try common paths)
# -----------------------------------------------------------
def find_tesseract():
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    return None

tess_path = find_tesseract()
if tess_path:
    pytesseract.pytesseract.tesseract_cmd = tess_path
else:
    st.warning("⚠️ Tesseract not found. OCR may not work. Install Tesseract OCR.")

# -----------------------------------------------------------
# 🧩 Helper Functions
# -----------------------------------------------------------
def analyze_sentiment(text):
    if not text.strip():
        return "neutral", "😐"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1: 
        return "positive", "😊"
    elif polarity < -0.1: 
        return "negative", "😔"
    else: 
        return "neutral", "😐"

def extract_text_from_image(uploaded_image):
    try:
        img = Image.open(uploaded_image)
        text = pytesseract.image_to_string(img).strip()
        return text if text else "⚠️ No text detected in image"
    except Exception as e:
        return f"⚠️ OCR failed: {str(e)}"

def extract_text_from_pdf(uploaded_pdf):
    try:
        pdf_reader = PdfReader(uploaded_pdf)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}"
        return text.strip() if text else "⚠️ No text found in PDF"
    except Exception as e:
        return f"⚠️ PDF reading failed: {str(e)}"

def ask_llama3(prompt, model):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        full_reply = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded = chunk.decode("utf-8", errors="ignore")
                    data = json.loads(decoded)
                    if "response" in data:
                        full_reply += data["response"]
                        yield full_reply
                    if data.get("done"):
                        break
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    except requests.exceptions.RequestException as e:
        yield f"❌ Connection error: {str(e)}"
    except Exception as e:
        yield f"❌ Unexpected error: {str(e)}"

def generate_report():
    report = []
    
    # Document content
    if st.session_state.context_text and "⚠️" not in st.session_state.context_text:
        report.append("="*60)
        report.append("DOCUMENT CONTENT")
        report.append("="*60)
        report.append(st.session_state.context_text)
        report.append("\n" + "="*60 + "\n")
    
    # Sentiment
    if st.session_state.sentiment[0] != "neutral":
        report.append(f"Document Sentiment: {st.session_state.sentiment[0].title()}")
        report.append("")
    
    # Chat history
    if st.session_state.messages:
        report.append("CHAT HISTORY")
        report.append("="*60)
        for msg in st.session_state.messages:
            role = "USER" if msg["role"] == "user" else "AI ASSISTANT"
            report.append(f"[{role}]: {msg['content']}")
        report.append("\n" + "="*60)
    
    # Footer
    report.append(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("Powered by AI Document Assistant")
    
    return "\n".join(report)

# -----------------------------------------------------------
# 💬 Initialize Session State
# -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "sentiment" not in st.session_state:
    st.session_state.sentiment = ("neutral", "😐")
if "show_full_text" not in st.session_state:
    st.session_state.show_full_text = False

# -----------------------------------------------------------
# 📤 File Upload Section
# -----------------------------------------------------------
st.subheader("📂 Upload Your Document")
file_type = st.radio("Choose file type:", ["Image (OCR)", "PDF"], horizontal=True)

uploaded_file = st.file_uploader(
    f"Upload a {'image' if file_type == 'Image (OCR)' else 'PDF'} file",
    type=["jpg", "jpeg", "png"] if file_type == "Image (OCR)" else ["pdf"]
)

# -----------------------------------------------------------
# 🧾 Handle File Upload & Text Extraction
# -----------------------------------------------------------
if uploaded_file:
    with st.spinner(f"Processing {file_type.lower()}..."):
        if file_type == "Image (OCR)":
            extracted_text = extract_text_from_image(uploaded_file)
        else:
            extracted_text = extract_text_from_pdf(uploaded_file)
        
        if "⚠️" not in extracted_text:
            st.session_state.context_text = extracted_text
            sentiment, emoji = analyze_sentiment(extracted_text)
            st.session_state.sentiment = (sentiment, emoji)
            st.success(f"✅ Successfully processed {file_type}")
        else:
            st.error(extracted_text)
            st.session_state.context_text = ""

# -----------------------------------------------------------
# 📜 Display Document Controls
# -----------------------------------------------------------
if st.session_state.context_text and "⚠️" not in st.session_state.context_text:
    preview = st.session_state.context_text[:1500] + "..." if len(st.session_state.context_text) > 1500 else st.session_state.context_text
    st.text_area("📜 Document Preview", preview, height=200, key="preview_area")
    
    sentiment, emoji = st.session_state.sentiment
    st.markdown(f'<div class="sentiment-badge {sentiment}">{emoji} Sentiment: {sentiment.title()}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 Show Full Text"):
            st.session_state.show_full_text = True
    
    with col2:
        if st.button("💾 Download Full Report", type="primary"):
            report_content = generate_report()
            st.download_button(
                label="⬇️ Click to Download Report",
                data=report_content,
                file_name=f"document_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_report"
            )
    
    with col3:
        if st.session_state.show_full_text:
            if st.button("CloseOperation Full Text"):
                st.session_state.show_full_text = False
    
    if st.session_state.show_full_text:
        st.markdown('<div class="full-text-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Full Extracted Text")
        st.text_area(
            "", 
            st.session_state.context_text, 
            height=500,
            key="full_text_display"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# 💬 Display Chat History — FIXED SYNTAX
# -----------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # ✅ CORRECTED: Added missing parenthesis
        st.markdown(msg["content"])

# -----------------------------------------------------------
# 🧠 Process Chat Input
# -----------------------------------------------------------
if prompt := st.chat_input("💬 Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context-aware prompt
    if st.session_state.context_text and "⚠️" not in st.session_state.context_text:
        sentiment, _ = st.session_state.sentiment
        # Truncate to fit Llama3 context window
        context_snippet = st.session_state.context_text[:8000]
        full_prompt = (
            f"You are an expert document analyst. The following document has a {sentiment} sentiment.\n\n"
            f"Document:\n{context_snippet}\n\n"
            f"User Question: {prompt}\n\n"
            f"Answer concisely and accurately based ONLY on the document above. If the question cannot be answered from the document, say 'I cannot answer that based on the provided document.'"
        )
    else:
        full_prompt = f"Answer this general question: {prompt}"
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Show typing indicator
        message_placeholder.markdown(
            '<div class="typing-indicator"></div>'
            '<div class="typing-indicator"></div>'
            '<div class="typing-indicator"></div>', 
            unsafe_allow_html=True
        )
        
        full_response = ""
        for partial_response in ask_llama3(full_prompt, model):
            full_response = partial_response
            # Update in real-time
            message_placeholder.markdown(full_response + " 🤖")
            time.sleep(0.01)  # Smooth streaming
        
        # Final update
        message_placeholder.markdown(full_response + " 🤖")
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# -----------------------------------------------------------
# 📌 Footer
# -----------------------------------------------------------
st.divider()
st.caption("💡 Powered by LLaMA 3 via Ollama | OCR with Tesseract | PDF with PyPDF2 | Sentiment with TextBlob")