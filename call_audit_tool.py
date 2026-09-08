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
    # Get the FFmpeg executable bundled with imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # Make sure FFmpeg is executable
    os.chmod(ffmpeg_path, 0o755)

    # Get the folder containing FFmpeg
    ffmpeg_dir = os.path.dirname(ffmpeg_path)

    # Add FFmpeg folder to PATH
    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    # Check if FFmpeg can now be found
    ffmpeg_found = shutil.which("ffmpeg")

except Exception as e:
    ffmpeg_path = None
    ffmpeg_found = None
    st.error(f"FFmpeg setup error: {e}")


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


# =========================================================
# AUDIO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)


# =========================================================
# AUDIO PROCESSING
# =========================================================

if uploaded_file:

    # Reset results when a new file is uploaded
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

                if not ffmpeg_path:
                    raise RuntimeError(
                        "FFmpeg executable could not be located."
                    )

                if not os.path.exists(ffmpeg_path):
                    raise RuntimeError(
                        f"FFmpeg file does not exist: "
                        f"{ffmpeg_path}"
                    )

                # -----------------------------------------
                # Check executable permission
                # -----------------------------------------

                if not os.access(
                    ffmpeg_path,
                    os.X_OK
                ):

                    os.chmod(
                        ffmpeg_path,
                        0o755
                    )

                # -----------------------------------------
                # Save uploaded audio temporarily
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
                # Verify FFmpeg through PATH
                # -----------------------------------------

                ffmpeg_check = shutil.which(
                    "ffmpeg"
                )

                if ffmpeg_check is None:

                    # Directly put the exact FFmpeg
                    # executable into PATH
                    os.environ["PATH"] = (
                        os.path.dirname(ffmpeg_path)
                        + os.pathsep
                        + os.environ.get("PATH", "")
                    )

                    ffmpeg_check = shutil.which(
                        "ffmpeg"
                    )

                # -----------------------------------------
                # If still not found, show exact problem
                # -----------------------------------------

                if ffmpeg_check is None:

                    raise FileNotFoundError(
                        "FFmpeg exists but could not "
                        "be found in PATH."
                    )

                # -----------------------------------------
                # Transcribe audio
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

                # Debug information
                st.write(
                    "FFmpeg expected at:"
                )

                st.code(
                    str(ffmpeg_path)
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

                # -----------------------------------------
                # Delete temporary audio
                # -----------------------------------------

                if (
                    tmp_path
                    and os.path.exists(tmp_path)
                ):

                    os.remove(tmp_path)


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
