import streamlit as st
import google.generativeai as genai
import tempfile
import os
import whisper
import imageio_ffmpeg
import shutil


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Strategic Auditor 2026",
    layout="wide"
)


# =========================================================
# FFMPEG SETUP
# =========================================================

try:
    original_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # Create a writable temporary directory
    ffmpeg_dir = os.path.join(
        tempfile.gettempdir(),
        "ffmpeg_bin"
    )

    os.makedirs(
        ffmpeg_dir,
        exist_ok=True
    )

    # Create a local executable named "ffmpeg"
    local_ffmpeg = os.path.join(
        ffmpeg_dir,
        "ffmpeg"
    )

    # Copy FFmpeg to the writable directory
    if not os.path.exists(local_ffmpeg):
        shutil.copy2(
            original_ffmpeg,
            local_ffmpeg
        )

    # Make our copied FFmpeg executable
    os.chmod(
        local_ffmpeg,
        0o755
    )

    # Put our FFmpeg directory first in PATH
    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    # Check FFmpeg
    ffmpeg_found = shutil.which("ffmpeg")

except Exception as e:
    original_ffmpeg = None
    local_ffmpeg = None
    ffmpeg_found = None

    st.error(
        f"FFmpeg setup error: {e}"
    )


# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")


whisper_model = load_whisper()


# =========================================================
# SESSION STATE
# =========================================================

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# =========================================================
# TITLE
# =========================================================

st.title("🎙️ AI Strategic Call Auditor")


# =========================================================
# SIDEBAR
# =========================================================

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


# =========================================================
# GEMINI ANALYSIS
# =========================================================

def analyze_with_gemini(
    transcript,
    key
):

    genai.configure(
        api_key=key
    )

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

    response = model.generate_content(
        prompt
    )

    return response.text


# =========================================================
# AUDIO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=[
        "wav",
        "mp3",
        "m4a"
    ]
)


# =========================================================
# PROCESS AUDIO
# =========================================================

if uploaded_file:

    # Reset results for a new file
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


    # =====================================================
    # STEP 1 - TRANSCRIPTION
    # =====================================================

    if st.button(
        "Step 1: Extract Transcript 📄"
    ):

        with st.spinner(
            "Transcribing... Using Whisper Tiny"
        ):

            tmp_path = None

            try:

                # -----------------------------------------
                # Check FFmpeg
                # -----------------------------------------

                if local_ffmpeg is None:

                    raise RuntimeError(
                        "FFmpeg could not be prepared."
                    )

                if not os.path.exists(
                    local_ffmpeg
                ):

                    raise RuntimeError(
                        "FFmpeg executable does not exist."
                    )

                if not os.access(
                    local_ffmpeg,
                    os.X_OK
                ):

                    raise RuntimeError(
                        "FFmpeg is not executable."
                    )

                # -----------------------------------------
                # Save uploaded audio
                # -----------------------------------------

                suffix = (
                    "."
                    + uploaded_file.name
                    .split(".")[-1]
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(
                        uploaded_file.getvalue()
                    )

                    tmp_path = tmp.name

                # -----------------------------------------
                # Confirm FFmpeg is available
                # -----------------------------------------

                ffmpeg_check = shutil.which(
                    "ffmpeg"
                )

                if ffmpeg_check is None:

                    raise RuntimeError(
                        "FFmpeg is not available in PATH."
                    )

                # -----------------------------------------
                # Whisper transcription
                # -----------------------------------------

                result = whisper_model.transcribe(
                    tmp_path,
                    fp16=False
                )

                # -----------------------------------------
                # Save transcript
                # -----------------------------------------

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

                st.write(
                    "Original FFmpeg:"
                )

                st.code(
                    str(original_ffmpeg)
                )

                st.write(
                    "Local FFmpeg:"
                )

                st.code(
                    str(local_ffmpeg)
                )

                st.write(
                    "FFmpeg found in PATH:"
                )

                st.code(
                    str(
                        shutil.which("ffmpeg")
                    )
                )

            finally:

                # Remove temporary audio
                if (
                    tmp_path
                    and os.path.exists(
                        tmp_path
                    )
                ):

                    os.remove(
                        tmp_path
                    )


    # =====================================================
    # SHOW TRANSCRIPT
    # =====================================================

    if st.session_state.transcript:

        st.text_area(
            "Transcript:",
            st.session_state.transcript,
            height=250
        )


        # =================================================
        # STEP 2 - GEMINI ANALYSIS
        # =================================================

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


    # =====================================================
    # SHOW ANALYSIS
    # =====================================================

    if st.session_state.analysis:

        st.markdown(
            st.session_state.analysis
        )


        # =================================================
        # DOWNLOAD REPORT
        # =================================================

        st.download_button(
            "Download Report",
            st.session_state.analysis,
            file_name="audit.md",
            mime="text/markdown"
        )
