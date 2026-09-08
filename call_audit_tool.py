import streamlit as st
import google.generativeai as genai
import tempfile
import os
import whisper

st.set_page_config(
    page_title="Strategic Auditor 2026",
    layout="wide"
)

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

whisper_model = load_whisper()

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

st.title("🎙️ AI Strategic Call Auditor")

with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )
    st.info("Mode: Ultra-Light (Whisper Tiny + Gemini Flash)")


def analyze_with_gemini(transcript, key):
    genai.configure(api_key=key)

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Audit this call based on these PILLARS:

    1. Motivation
    2. Price
    3. Timeline
    4. Condition
    5. Rapport

    Transcript:
    {transcript}
    """

    res = model.generate_content(prompt)

    return res.text


uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file:

    if st.session_state.last_uploaded_file != uploaded_file.name:

        st.session_state.transcript = ""
        st.session_state.analysis = ""
        st.session_state.last_uploaded_file = uploaded_file.name

        st.rerun()

    if st.button("Step 1: Extract Transcript 📄"):

        with st.spinner("Transcribing... Using Whisper Tiny"):

            tmp_path = None

            try:
                suffix = f".{uploaded_file.name.split('.')[-1]}"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                result = whisper_model.transcribe(
                    tmp_path,
                    fp16=False
                )

                st.session_state.transcript = result["text"]

                st.success("✅ Transcription complete!")

            except Exception as e:

                st.error(f"Transcription Error: {e}")

            finally:

                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if st.session_state.transcript:

        st.text_area(
            "Transcript:",
            st.session_state.transcript,
            height=200
        )

        if st.button("Step 2: Run Strategic Analysis 🚀"):

            if user_api_key:

                with st.spinner("Analyzing..."):

                    try:

                        analysis = analyze_with_gemini(
                            st.session_state.transcript,
                            user_api_key
                        )

                        st.session_state.analysis = analysis

                        st.success("✅ Analysis Complete!")

                    except Exception as e:

                        st.error(f"Analysis Error: {e}")

            else:

                st.warning("Please enter API Key")

    if st.session_state.analysis:

        st.markdown(st.session_state.analysis)

        st.download_button(
            "Download Report",
            st.session_state.analysis,
            file_name="audit.md"
        )
