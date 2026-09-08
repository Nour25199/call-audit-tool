import streamlit as st
import google.generativeai as genai
import tempfile
import os
import whisper
import imageio_ffmpeg
import shutil

# --- 1. Page Config ---
st.set_page_config(
    page_title="Strategic Auditor 2026",
    layout="wide"
)

# --- 2. FFmpeg Setup ---
# imageio-ffmpeg provides the binary, but Streamlit Cloud may not
# allow chmod inside site-packages. So we copy it to /tmp first.

try:
    original_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_bin")
    os.makedirs(ffmpeg_dir, exist_ok=True)

    local_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg")

    if not os.path.exists(local_ffmpeg):
        shutil.copy2(original_ffmpeg, local_ffmpeg)

    os.chmod(local_ffmpeg, 0o755)

    # Put our writable FFmpeg folder first in PATH
    os.environ["PATH"] = (
        ffmpeg_dir
        + os.pathsep
        + os.environ.get("PATH", "")
    )

except Exception as e:
    st.error(f"FFmpeg setup error: {e}")


# --- 3. Load Whisper ---
@st.cache_resource
def load_whisper():
    # Same Tiny model as the old working version
    return whisper.load_model("tiny")


whisper_model = load_whisper()


# --- 4. State Management ---
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "analysis" not in st.session_state:
    st.session_state.analysis = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# --- 5. Title ---
st.title("🎙️ AI Strategic Call Auditor")


# --- 6. Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    user_api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    st.info("Mode: Ultra-Light (Whisper Tiny + Gemini Flash)")


# --- 7. Gemini Analysis ---
def analyze_with_gemini(transcript, key):

    genai.configure(api_key=key)

    model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    prompt = f"""
Audit this call based on these PILLARS:

1. Motivation
2. Price
3. Timeline
4. Condition
5. Rapport

Review the Lead Manager's performance and identify
strengths, areas to improve, and missed opportunities.

Transcript:

{transcript}
"""

    res = model.generate_content(prompt)

    return res.text


# --- 8. Upload Audio ---
uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "m4a"]
)


if uploaded_file:

    # Reset when a new file is uploaded
    if st.session_state.last_uploaded_file != uploaded_file.name:

        st.session_state.transcript = ""
        st.session_state.analysis = ""
        st.session_state.last_uploaded_file = uploaded_file.name

        st.rerun()


    # --- STEP 1: Transcription ---
    if st.button("Step 1: Extract Transcript 📄"):

        tmp_path = None

        with st.spinner("Transcribing... (Using Tiny Model for Speed)"):

            try:

                # Save uploaded audio to temp file
                suffix = f".{uploaded_file.name.split('.')[-1]}"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name


                # Transcribe using Whisper Tiny
                result = whisper_model.transcribe(
                    tmp_path,
                    fp16=False
                )

                st.session_state.transcript = result["text"]

                st.success("✅ Done!")


            except Exception as e:

                st.error(
                    f"Transcription Error: {e}"
                )


            finally:

                # Always delete temporary audio file
                if tmp_path and os.path.exists(tmp_path):

                    try:
                        os.remove(tmp_path)
                    except:
                        pass


    # --- Display Transcript ---
    if st.session_state.transcript:

        st.text_area(
            "Transcript:",
            st.session_state.transcript,
            height=200
        )


        # --- STEP 2: Analysis ---
        if st.button("Step 2: Run Strategic Analysis 🚀"):

            if user_api_key:

                with st.spinner("Analyzing..."):

                    try:

                        analysis = analyze_with_gemini(
                            st.session_state.transcript,
                            user_api_key
                        )

                        st.session_state.analysis = analysis

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


    # --- Display Results ---
    if st.session_state.analysis:

        st.markdown(
            st.session_state.analysis
        )

        st.download_button(
            "Download Report",
            st.session_state.analysis,
            file_name="audit.md"
        )
