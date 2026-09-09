import streamlit as st
import google.generativeai as genai
import tempfile
import os
import imageio_ffmpeg
import shutil

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Strategic Auditor 2026",
    layout="wide"
)


# --------------------------------------------------
# FFMPEG SETUP
# --------------------------------------------------

try:
    original_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_dir = os.path.join(
        tempfile.gettempdir(),
        "ffmpeg_bin"
    )

    os.makedirs(ffmpeg_dir, exist_ok=True)

    local_ffmpeg = os.path.join(
        ffmpeg_dir,
        "ffmpeg"
    )

    if not os.path.exists(local_ffmpeg):
        shutil.copy2(
            original_ffmpeg,
            local_ffmpeg
        )

    os.chmod(local_ffmpeg, 0o755)

    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ.get("PATH", "")
    )

except Exception as e:
    st.error(f"FFmpeg setup error: {e}")


# --------------------------------------------------
# FASTER WHISPER
# --------------------------------------------------

@st.cache_resource
def load_whisper():

    from faster_whisper import WhisperModel

    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1
    )


whisper_model = load_whisper()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎙️ AI Strategic Call Auditor")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    st.info(
        "Mode: Faster-Whisper Tiny + Gemini Flash"
    )


# --------------------------------------------------
# GEMINI
# --------------------------------------------------

def analyze_with_gemini(transcript, key):

    genai.configure(api_key=key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    prompt = f"""
Audit this Lead Manager call based on:

1. Motivation
2. Price
3. Timeline
4. Condition
5. Rapport

Identify:

- Call Summary
- Situation
- Motivation / Pain
- Timeline
- Condition
- Price Expectation + reason
- Decision Maker(s)
- Objections / Concerns
- Outcome / Next Step
- Important Notes
- Strength
- Areas to Improve
- Missed Opportunity

Transcript:

{transcript}
"""

    response = model.generate_content(prompt)

    return response.text


# --------------------------------------------------
# UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)


if uploaded_file:

    # Reset when a different file is uploaded
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
    # STEP 1
    # --------------------------------------------------

    if st.button(
        "Step 1: Extract Transcript 📄"
    ):

        tmp_path = None

        with st.spinner(
            "Transcribing..."
        ):

            try:

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


                # Faster-Whisper transcription
                segments, info = whisper_model.transcribe(
                    tmp_path,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=True
                )


                # Convert segments into text
                transcript_parts = []

                for segment in segments:
                    transcript_parts.append(
                        segment.text
                    )


                transcript = " ".join(
                    transcript_parts
                ).strip()


                st.session_state.transcript = transcript

                st.success("✅ Transcription Complete!")


            except Exception as e:

                st.error(
                    f"Transcription Error: {e}"
                )


            finally:

                if (
                    tmp_path
                    and os.path.exists(tmp_path)
                ):

                    try:
                        os.remove(tmp_path)
                    except:
                        pass


    # --------------------------------------------------
    # DISPLAY TRANSCRIPT
    # --------------------------------------------------

    if st.session_state.transcript:

        st.text_area(
            "Transcript:",
            st.session_state.transcript,
            height=250
        )


        # --------------------------------------------------
        # STEP 2
        # --------------------------------------------------

        if st.button(
            "Step 2: Run Strategic Analysis 🚀"
        ):

            if user_api_key:

                with st.spinner(
                    "Analyzing..."
                ):

                    try:

                        analysis = analyze_with_gemini(
                            st.session_state.transcript,
                            user_api_key
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
                    "Please enter API Key"
                )


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    if st.session_state.analysis:

        st.markdown(
            st.session_state.analysis
        )

        st.download_button(
            "Download Report",
            st.session_state.analysis,
            file_name="audit.md"
        )
