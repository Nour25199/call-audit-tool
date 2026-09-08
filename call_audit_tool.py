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

    ffmpeg_dir = os.path.join(
        tempfile.gettempdir(),
        "ffmpeg_bin"
    )

    os.makedirs(
        ffmpeg_dir,
        exist_ok=True
    )

    local_ffmpeg = os.path.join(
        ffmpeg_dir,
        "ffmpeg"
    )

    if not os.path.exists(local_ffmpeg):
        shutil.copy2(
            original_ffmpeg,
            local_ffmpeg
        )

    os.chmod(
        local_ffmpeg,
        0o755
    )

    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ.get("PATH", "")
    )

except Exception as e:
    original_ffmpeg = None
    local_ffmpeg = None

    st.error(
        f"FFmpeg setup error: {e}"
    )


# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_whisper():

    # Tiny = fastest Whisper model
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
        "Mode: Fast Whisper Tiny + Gemini Flash"
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

    # Reset when a new audio file is uploaded
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
            "Transcribing... Please wait"
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

                # -----------------------------------------
                # Save audio temporarily
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
                # Verify FFmpeg
                # -----------------------------------------

                ffmpeg_check = shutil.which(
                    "ffmpeg"
                )

                if ffmpeg_check is None:

                    raise RuntimeError(
                        "FFmpeg is not available."
                    )


                # -----------------------------------------
                # FAST WHISPER TRANSCRIPTION
                # -----------------------------------------

                result = whisper_model.transcribe(

                    tmp_path,

                    # CPU optimization
                    fp16=False,

                    # Faster decoding
                    temperature=0,

                    # Don't waste time trying many
                    # alternative decoding attempts
                    best_of=1,

                    beam_size=1,

                    # Automatically detect language
                    # and transcribe
                    task="transcribe"
                )


                # -----------------------------------------
                # Save transcript
                # -----------------------------------------

                st.session_state.transcript = (
                    result["text"].strip()
                )

                st.success(
                    "✅ Transcription complete!"
                )


            except Exception as e:

                st.error(
                    f"Transcription Error: {e}"
                )

                st.write(
                    "FFmpeg:"
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
