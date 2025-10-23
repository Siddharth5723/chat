from dotenv import load_dotenv
import streamlit as st
import os
import io
import json
import re
import time

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except Exception:
    _GTTS_AVAILABLE = False

import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.set_page_config(page_title="QUERY_BOT - missing API key")
    st.warning("GOOGLE_API_KEY not found in environment. Put it in a .env file before using the model.")
else:
    genai.configure(api_key=API_KEY)

try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception:
    model = None

# Initializing session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'mode' not in st.session_state:
    st.session_state.mode = 'Normal'

if 'max_pairs' not in st.session_state:
    st.session_state.max_pairs = 8

if 'tts' not in st.session_state:
    st.session_state.tts = False

# Utility functions

def trim_history():
    max_entries = st.session_state.max_pairs * 2
    if len(st.session_state.history) > max_entries:
        st.session_state.history = st.session_state.history[-max_entries:]

def is_code_like(text: str) -> bool:
    if '```' in text:
        return True
    code_tokens = ["def ", "class ", "import ", "console.log", "System.out", "#include", "function(", "->"]
    lower = text.lower()
    for t in code_tokens:
        if t in text or t in lower:
            return True
    return False

def build_system_instruction(mode: str) -> str:
    if mode == 'Normal':
        return "You are a helpful assistant. Keep answers concise (2-4 sentences) and accurate."
    if mode == 'Detailed':
        return "You are a helpful assistant. Provide a detailed explanation, step-by-step, with examples where appropriate."
    if mode == 'Creative':
        return "You are a creative assistant. Use analogies, stories, or creative examples to make explanations memorable."
    if mode == 'Code':
        return "You are a programming assistant. Prefer to answer with clear, runnable code examples, include short comments, and format code blocks."
    return "You are a helpful assistant. Answer clearly and use examples when useful."

def build_prompt_from_history(new_user_input: str, mode: str) -> str:
    system_instruction = build_system_instruction(mode)
    parts = [f"System instruction:\n{system_instruction}\n\nConversation so far:"]
    for msg in st.session_state.history:
        role = msg.get('role')
        content = msg.get('content')
        if role and content is not None:
            parts.append(f"{role.capitalize()}: {content}")
    parts.append(f"User: {new_user_input}")
    parts.append("Assistant:")
    return "\n".join(parts)

def call_model(prompt: str, max_retries: int = 2) -> str:
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            if model is not None:
                resp = model.generate_content(prompt)
                if hasattr(resp, 'text'):
                    return resp.text
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, 'outputs'):
                    collected = []
                    for out in resp.outputs:
                        if hasattr(out, 'content') and hasattr(out.content, 'text'):
                            collected.append(out.content.text)
                    if collected:
                        return '\n'.join(collected)
                return str(resp)
            else:
                resp = genai.generate(prompt)
                if isinstance(resp, dict) and 'candidates' in resp:
                    return resp['candidates'][0].get('content', '')
                return str(resp)
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
            continue
    raise RuntimeError(f"Model call failed after {max_retries+1} attempts: {last_exc}")

def synthesize_speech_bytes(text: str, lang: str = 'en') -> bytes:
    if not _GTTS_AVAILABLE:
        raise ImportError("gTTS not available. Install with `pip install gTTS` to enable TTS.")
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp.read()

# Streamlit UI

st.set_page_config(page_title="QUERY_BOT (Upgraded)", layout="wide")
st.title("QUERY_BOT — upgraded")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Response mode",
        ['Normal', 'Detailed', 'Creative', 'Code'],
        index=['Normal','Detailed','Creative','Code'].index(st.session_state.mode) if st.session_state.mode in ['Normal','Detailed','Creative','Code'] else 0
    )
    st.session_state.mode = mode

    st.session_state.tts = st.checkbox("Enable TTS (play assistant replies)", value=st.session_state.tts)

    st.session_state.max_pairs = st.number_input(
        "Max conversation pairs to keep (user+assistant)",
        min_value=1, max_value=50,
        value=st.session_state.max_pairs, step=1
    )

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
    payload = json.dumps(st.session_state.history, indent=2)
    st.download_button("Download JSON", data=payload, file_name="conversation.json", mime="application/json")

chat_col, control_col = st.columns([3, 1])

with chat_col:
    st.subheader("Conversation")
    if not st.session_state.history:
        st.info("Start the conversation by typing a question on the right and pressing 'Ask'.")

    for entry in st.session_state.history:
        if entry['role'] == 'user':
            st.chat_message('user').write(entry['content'])
        else:
            if entry.get('is_code'):
                st.chat_message('assistant').code(entry['content'], language='')
            else:
                st.chat_message('assistant').write(entry['content'])

with control_col:
    st.subheader("Ask the bot")
    user_input = st.text_input("Input", key='user_input')
    ask = st.button("Ask")

    st.markdown("**Optional:** Upload an audio file (.wav/.mp3) to transcribe (experimental)")
    audio_file = st.file_uploader("Upload audio (optional)", type=['wav', 'mp3', 'm4a'])

    if audio_file is not None and not user_input:
        st.info("Audio uploaded. Transcription not yet implemented.")

    st.markdown("---")
    st.caption("Guards: empty input ignored; say 'bye' to end politely.")

# Handle Ask button

if ask:
    raw = st.session_state.get('user_input', '').strip()
    if not raw:
        st.warning("Please enter a question before pressing Ask.")
    else:
        if re.search(r"\b(bye|exit|quit|goodbye)\b", raw, flags=re.I):
            farewell = "Goodbye! If you want to start again, press Clear chat."
            st.session_state.history.append({'role':'user', 'content': raw})
            st.session_state.history.append({'role':'assistant', 'content': farewell, 'is_code': False})
            trim_history()
            st.rerun()
        else:
            st.session_state.history.append({'role':'user', 'content': raw})
            trim_history()
            prompt = build_prompt_from_history(raw, st.session_state.mode)

            with st.spinner("Generating response..."):
                try:
                    reply_text = call_model(prompt)
                except Exception as e:
                    err_msg = f"Model call failed: {e}"
                    st.session_state.history.append({'role':'assistant', 'content': err_msg, 'is_code': False})
                    st.error(err_msg)
                    trim_history()
                    st.rerun()

            is_code = is_code_like(reply_text)
            if '```' in reply_text:
                parts = reply_text.split('```')
                if len(parts) >= 3:
                    code_candidate = parts[1]
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

            if is_code and '```' not in final_assistant_content:
                final_assistant_content = f"```\n{final_assistant_content}\n```"

            st.session_state.history.append({'role':'assistant', 'content': final_assistant_content, 'is_code': is_code})
            trim_history()

            if st.session_state.tts:
                try:
                    mp3_bytes = synthesize_speech_bytes(re.sub(r'```', '', final_assistant_content))
                    st.audio(mp3_bytes, format='audio/mp3')
                except Exception as e:
                    st.error(f"TTS error: {e}")

            st.session_state['user_input'] = ''
            st.rerun()
