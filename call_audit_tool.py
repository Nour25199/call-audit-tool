import streamlit as st
import google.generativeai as genai
import tempfile
import os
import whisper
import imageio_ffmpeg

# --------------------------------------------------
# FFmpeg Setup
# --------------------------------------------------

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

os.environ["PATH"] = (
    os.path.dirname(ffmpeg_path)
    + os.pathsep
    + os.environ.get("PATH", "")
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Strategic Auditor 2026",
    layout="wide"
)

# --------------------------------------------------
# Load Whisper
# --------------------------------------------------

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")


whisper_model = load_whisper()

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎙️ AI Strategic Call Auditor")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    st.info(
        "Mode: Ultra-Light "
        "(Whisper Tiny + Gemini Flash)"
    )

# --------------------------------------------------
# Gemini Analysis
# --------------------------------------------------

def analyze_with_gemini(transcript, key):

    genai.configure(api_key=key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    prompt = f"""
Audit this Lead Manager call based on these
Five Pillars:

1. Motivation
2. Price
3. Timeline
4. Condition
5. Rapport

Provide detailed coaching feedback.

Transcript:
{transcript}
"""

    response = model.generate_content(prompt)

    return response.text


# --------------------------------------------------
# Audio Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)

# --------------------------------------------------
# Main Logic
# --------------------------------------------------

if uploaded_file:

    # Detect new file
    if (
        st.session_state.last_uploaded_file
        != uploaded_file.name
    ):

        st.session_state.transcript = ""
        st.session_state.analysis = ""
        st.session_state.last_uploaded_file = (
            uploaded_file.name
        )

        st.rerun()

    # --------------------------------------------------
    # Step 1 - Transcription
    # --------------------------------------------------

    if st.button(
        "Step 1: Extract Transcript 📄"
    ):

        with st.spinner(
            "Transcribing... Using Whisper Tiny"
        ):

            tmp_path = None

            try:

                # Save uploaded audio temporarily
                suffix = (
                    "."
                    + uploaded_file.name.split(".")[-1]
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(
                        uploaded_file.getvalue()
                    )

                    tmp_path = tmp.name

                # Make sure FFmpeg is available
                if not os.path.exists(ffmpeg_path):

                    raise FileNotFoundError(
                        "FFmpeg executable was not found."
                    )

                # Transcribe
                result = whisper_model.transcribe(
                    tmp_path,
                    fp16=False
                )

                st.session_state.transcript = (
                    result["text"]
                )

                st.success(
                    "✅ Transcription complete!"
                )

            except Exception as e:

                st.error(
                    f"Transcription Error: {e}"
                )

            finally:

                # Delete temporary audio file
                if (
                    tmp_path
                    and os.path.exists(tmp_path)
                ):

                    os.remove(tmp_path)

    # --------------------------------------------------
    # Display Transcript
    # --------------------------------------------------

    if st.session_state.transcript:

        st.text_area(
            "Transcript:",
            st.session_state.transcript,
            height=250
        )

        # --------------------------------------------------
        # Step 2 - Gemini Analysis
        # --------------------------------------------------

        if st.button(
            "Step 2: Run Strategic Analysis 🚀"
        ):

            if user_api_key:

                with st.spinner(
                    "Analyzing..."
                ):

                    try:

                        analysis = (
                            analyze_with_gemini(
                                st.session_state.transcript,
                                user_api_key
                            )
                        )

                        st.session_state.analysis = (
                            analysis
                        )

                        st.success(
                            "✅ Analysis Complete!"
                        )

                    except Exception as e:

                        st.error(
                            f"Analysis Error: {e}"
                        )

            else:

                st.warning(
                    "Please enter your Gemini API Key."
                )

    # --------------------------------------------------
    # Display Analysis
    # --------------------------------------------------

    if st.session_state.analysis:

        st.markdown(
            st.session_state.analysis
        )

        st.download_button(
            "Download Report",
            st.session_state.analysis,
            file_name="audit.md",
            mime="text/markdown"
        )
