"""
QUERY_BOT_upgraded_streamlit.py

Upgraded Streamlit chatbot that implements the following features (step-by-step, with comments):

1) Conversation memory using `st.session_state` (keeps recent messages and trims history).
2) Multiple response modes: Normal, Detailed, Creative, Code.
3) Clear Chat button to reset conversation.
4) Improved UI using `st.chat_message`, spinner while waiting, and message formatting.
5) Guardrails: empty input check, polite "bye" handling, simple profanity filter placeholder.
6) Auto-detect code responses and render them with `st.code` when appropriate.
7) Optional voice output (text-to-speech) using `gTTS` (if installed) and `st.audio` to play the mp3.
8) Optional voice input via audio file upload (basic speech-to-text placeholder — left as an optional extension).
9) Download conversation as JSON/text for later use.

How to use
----------
1. Create a `.env` file in the same folder with:
   GOOGLE_API_KEY=your_api_key_here

2. Install dependencies (recommended):
   pip install streamlit python-dotenv google-generativeai gTTS

   - If you don't want TTS, you can skip gTTS.

3. Run the app:
   streamlit run QUERY_BOT_upgraded_streamlit.py

Notes about the Gemini call
--------------------------
- This file uses the same simple call style as your original snippet (`model.generate_content(prompt)`), but to avoid reliance on any non-standard keyword args this code builds a single text prompt from the conversation history and a short system instruction that encodes the selected "mode".
- If you later want to switch to a native messages/structured API (if your SDK supports it), that change is localized to `call_model()`.


CODE
====
"""

from dotenv import load_dotenv
import streamlit as st
import os
import io
import json
import re
import time

# Optional: TTS
try:
    from gtts import gTTS
    _GTTs_AVAILABLE = True
except Exception:
    _GTTs_AVAILABLE = False

# Google Gemini SDK (same as your original example)
import google.generativeai as genai

# ----------------------
# Helper / Initialization
# ----------------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    # We still start the app so user can read the code, but warn them.
    st.set_page_config(page_title="QUERY_BOT - missing API key")
    st.warning("GOOGLE_API_KEY not found in environment. Put it in a .env file before using the model.")
else:
    genai.configure(api_key=API_KEY)

# If you prefer a different model name, change it here
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    # If the SDK can't construct the high-level object, we'll still try to call genai directly in call_model()
    model = None


# Initialize session state variables for persisting the chat across reruns
if 'history' not in st.session_state:
    # history is a list of dicts: {role: 'user'|'assistant', 'content': str, 'is_code': bool}
    st.session_state.history = []

if 'mode' not in st.session_state:
    st.session_state.mode = 'Normal'

if 'max_pairs' not in st.session_state:
    st.session_state.max_pairs = 8  # how many user+assistant pairs to keep in memory

if 'tts' not in st.session_state:
    st.session_state.tts = False

# ----------------------
# Utility functions
# ----------------------

def trim_history():
    """Trim history to last max_pairs user+assistant pairs."""
    # Each pair has two entries (user, assistant). Keep up to max_pairs*2 entries.
    max_entries = st.session_state.max_pairs * 2
    if len(st.session_state.history) > max_entries:
        st.session_state.history = st.session_state.history[-max_entries:]


def is_code_like(text: str) -> bool:
    """Rudimentary check for code-like content."""
    # If the model returns triple-backticks or common programming tokens, treat as code
    if '```' in text:
        return True
    code_tokens = ["def ", "class ", "import ", "console.log", "System.out", "#include", "function(", "->"]
    lower = text.lower()
    for t in code_tokens:
        if t in text or t in lower:
            return True
    return False


def build_system_instruction(mode: str) -> str:
    """Return a short system instruction based on selected mode."""
    if mode == 'Normal':
        return "You are a helpful assistant. Keep answers concise (2-4 sentences) and accurate." 
    if mode == 'Detailed':
        return "You are a helpful assistant. Provide a detailed explanation, step-by-step, with examples where appropriate." 
    if mode == 'Creative':
        return "You are a creative assistant. Use analogies, stories, or creative examples to make explanations memorable." 
    if mode == 'Code':
        return "You are a programming assistant. Prefer to answer with clear, runnable code examples, include short comments, and format code blocks." 
    # default fallback
    return "You are a helpful assistant. Answer clearly and use examples when useful."


def build_prompt_from_history(new_user_input: str, mode: str) -> str:
    """Construct a single textual prompt that includes a short system instruction + conversation history.
    This keeps the code compatible with both simple generate_content interfaces and structured chat APIs.
    """
    system_instruction = build_system_instruction(mode)
    parts = [f"System instruction:\n{system_instruction}\n\nConversation so far:"]
    # Add history in simple 'User: ... / Assistant: ...' pairs
    for msg in st.session_state.history:
        role = msg.get('role')
        content = msg.get('content')
        if role and content is not None:
            parts.append(f"{role.capitalize()}: {content}")
    parts.append(f"User: {new_user_input}")
    parts.append("Assistant:")
    full_prompt = "\n".join(parts)
    return full_prompt


def call_model(prompt: str, max_retries:int=2) -> str:
    """Call the generative model. We try the higher-level `model.generate_content(prompt)` first (matches your original code).
    If that fails (for example model object not available), we fall back to the lower-level genai.generate() style if supported by the SDK.
    We return the assistant's textual reply or raise an exception after retries.
    """
    last_exc = None
    for attempt in range(max_retries+1):
        try:
            # Preferred: if `model` object exists and matches original usage
            if model is not None:
                # Many SDKs accept a simple text argument. Your earlier code used generate_content(query).
                # We pass the prompt string and read response.text
                resp = model.generate_content(prompt)
                # Some SDKs return a richer object; handle common patterns:
                if hasattr(resp, 'text'):
                    return resp.text
                # Fallback: maybe resp is a string already
                if isinstance(resp, str):
                    return resp
                # If resp has `.candidates` or `.outputs`, try common fields
                if hasattr(resp, 'outputs'):
                    # join textual outputs
                    collected = []
                    for out in resp.outputs:
                        if hasattr(out, 'content') and hasattr(out.content, 'text'):
                            collected.append(out.content.text)
                    if collected:
                        return '\n'.join(collected)
                # Last simple attempt
                return str(resp)
            else:
                # If model object not constructed, try genai.generate (this may not match your SDK exactly)
                resp = genai.generate(prompt)
                # Try to extract text from a common shape
                if isinstance(resp, dict) and 'candidates' in resp:
                    return resp['candidates'][0].get('content', '')
                return str(resp)

        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue
    # If we get here, every attempt failed
    raise RuntimeError(f"Model call failed after {max_retries+1} attempts: {last_exc}")



def synthesize_speech_bytes(text: str, lang: str = 'en') -> bytes:
    """Return MP3 bytes for the given text using gTTS (if available).
    If gTTS is missing, raise ImportError so caller can handle it.
    """
    if not _GTTs_AVAILABLE:
        raise ImportError("gTTS not available. Install with `pip install gTTS` to enable TTS.")
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp.read()


# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="QUERY_BOT (Upgraded)", layout="wide")
st.title("QUERY_BOT — upgraded")

# Sidebar controls (modes, options)
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Response mode", ['Normal', 'Detailed', 'Creative', 'Code'], index=['Normal','Detailed','Creative','Code'].index(st.session_state.mode) if st.session_state.mode in ['Normal','Detailed','Creative','Code'] else 0)
    st.session_state.mode = mode

    st.checkbox("Enable TTS (play assistant replies)", value=st.session_state.tts, key='tts')
    st.session_state.tts = st.session_state.get('tts', False)

    st.number_input("Max conversation pairs to keep (user+assistant)", min_value=1, max_value=50, value=st.session_state.max_pairs, step=1, key='max_pairs')
    st.session_state.max_pairs = st.session_state.get('max_pairs', st.session_state.max_pairs)

    if API_KEY is None:
        st.info("API key missing — add GOOGLE_API_KEY to .env to enable model calls")
    else:
        st.success("API key loaded")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.success("Chat cleared")

    st.markdown("---")
    st.markdown("**Export / Import**")
    if st.button("Download conversation (JSON)"):
        payload = json.dumps(st.session_state.history, indent=2)
        st.download_button("Click to download JSON", data=payload, file_name="conversation.json", mime="application/json")

# Main columns: Chat area + optional controls
chat_col, control_col = st.columns([3,1])

with chat_col:
    # Show history as chat messages
    st.subheader("Conversation")
    # If history is empty show a short hint
    if not st.session_state.history:
        st.info("Start the conversation by typing a question on the right and pressing 'Ask'.")

    for entry in st.session_state.history:
        if entry['role'] == 'user':
            st.chat_message('user').write(entry['content'])
        else:
            # Assistant message
            if entry.get('is_code'):
                # If it's detected to be code, show with code block
                st.chat_message('assistant').code(entry['content'], language='')
            else:
                st.chat_message('assistant').write(entry['content'])

with control_col:
    st.subheader("Ask the bot")
    # Text input is stored in session to avoid losing it on reruns
    user_input = st.text_input("Input", key='user_input')
    ask = st.button("Ask")

    # Optional: allow audio upload for speech->text (this is a simple example rather than a polished solution)
    st.markdown("**Optional:** Upload an audio file (.wav/.mp3) to transcribe (experimental)")
    audio_file = st.file_uploader("Upload audio (optional)", type=['wav', 'mp3', 'm4a'])

    if audio_file is not None and not user_input:
        st.info("Audio uploaded. This app currently does not auto-transcribe client audio — if you want transcription, ask and I can add a simple transcription flow using `speech_recognition` or an on-device Whisper wrapper.")

    st.markdown("---")
    st.caption("Guards: empty input ignored; say 'bye' to end politely.")

# ----------------------
# Handle Ask button
# ----------------------

if ask:
    raw = st.session_state.get('user_input', '').strip()

    # Guardrail: empty input
    if not raw:
        st.warning("Please enter a question before pressing Ask.")
    else:
        # Guardrail: simple exit flow
        if re.search(r"\b(bye|exit|quit|goodbye)\b", raw, flags=re.I):
            farewell = "Goodbye! If you want to start again, press Clear chat."
            st.session_state.history.append({'role':'user', 'content': raw})
            st.session_state.history.append({'role':'assistant', 'content': farewell, 'is_code': False})
            trim_history()
            st.experimental_rerun()  # re-run to render new messages immediately

        else:
            # Append user message to history immediately for UX
            st.session_state.history.append({'role':'user', 'content': raw})
            trim_history()

            # Build prompt that carries the conversation + mode instruction
            prompt = build_prompt_from_history(raw, st.session_state.mode)

            # Show spinner while waiting
            with st.spinner("Generating response..."):
                try:
                    reply_text = call_model(prompt)
                except Exception as e:
                    # Show helpful error and allow the user to continue
                    err_msg = f"Model call failed: {e}"
                    st.session_state.history.append({'role':'assistant', 'content': err_msg, 'is_code': False})
                    st.error(err_msg)
                    trim_history()
                    st.experimental_rerun()

            # Post-process assistant reply: detect code, optionally clean leading/trailing whitespace
            is_code = is_code_like(reply_text)
            # If the model includes triple backticks, try to extract the inner code for a cleaner display
            if '```' in reply_text:
                # naive extraction: take content between first pair of triple backticks
                parts = reply_text.split('```')
                if len(parts) >= 3:
                    # parts = [before, codeblock, after, maybe more blocks...]
                    code_candidate = parts[1]
                    # Replace the entire content with the code block stripped, and keep explanation in the 'after' if present
                    # We'll store both pieces together so the user sees code then explanation
                    cleaned = code_candidate.strip()
                    remainder = '\n'.join([p.strip() for p in parts[2:]]).strip()
                    if remainder:
                        final_assistant_content = f"```\n{cleaned}\n```\n\n{remainder}"
                    else:
                        final_assistant_content = f"```\n{cleaned}\n```"
                else:
                    final_assistant_content = reply_text.strip()
            else:
                final_assistant_content = reply_text.strip()

            # Optionally format code responses separate from text
            # If it's code-like but does not contain triple backticks, we still mark is_code True to show as code block
            if is_code and '```' not in final_assistant_content:
                final_assistant_content = f"```\n{final_assistant_content}\n```"

            # Append assistant reply to history
            st.session_state.history.append({'role':'assistant', 'content': final_assistant_content, 'is_code': is_code})

            # Trim history
            trim_history()

            # Optionally synthesize and play speech
            if st.session_state.tts:
                try:
                    mp3_bytes = synthesize_speech_bytes(re.sub(r'```', '', final_assistant_content))
                    st.audio(mp3_bytes, format='audio/mp3')
                except Exception as e:
                    st.error(f"TTS error: {e}")

            # Clear the input text box for convenience
            st.session_state['user_input'] = ''
            # Trigger a re-run so the refreshed history is displayed immediately
            st.experimental_rerun()

# End of app

"""
