import time
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Vision AI — Picture & Video Inspector",
    page_icon="🎥",
    layout="wide"
)

# Custom Styling for Dark/Sleek Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to query Gemini Flash with Media (Image or Video)
def analyze_media_with_gemini(media_file, mime_type, prompt_instruction, api_key):
    client = genai.Client(api_key=api_key)
    
    max_retries = 3
    backoff = 2
    
    for attempt in range(max_retries):
        try:
            if mime_type.startswith("image"):
                image = Image.open(media_file)
                contents = [image, prompt_instruction]
            else:
                # For video files, upload via Gemini Files API or send bytes/file object
                # Gemini SDK supports direct file uploads or passing uploaded file objects
                uploaded_file_ref = client.files.upload(file=media_file)
                
                # Wait briefly if video processing state requires it, then prompt
                contents = [uploaded_file_ref, prompt_instruction]

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if ("503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str) and attempt < max_retries - 1:
                time.sleep(backoff ** (attempt + 1))
                continue
            else:
                raise e

# UI Header
st.title("🎥 Vision AI — Picture & Video Inspector")
st.write("Upload a **photograph** or a **video file** to detect objects, analyze scene events, inspect flaws, or track activities using Gemini Flash.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Enter key from Google AI Studio")
gemini_api_key = st.secrets.get("GEMINI_API_KEY", api_key_input)

if not gemini_api_key:
    st.sidebar.warning("⚠️ Please enter your Gemini API Key or configure it in Streamlit Secrets.")

# Mode Selection
analysis_mode = st.selectbox(
    "Select Analysis Focus",
    [
        "Comprehensive Object & Scene Inventory",
        "Defect, Flaw or Anomaly Detection",
        "Action/Event Breakdown (Best for Videos)",
        "Custom Prompt Analysis"
    ]
)

custom_prompt = ""
if analysis_mode == "Custom Prompt Analysis":
    custom_prompt = st.text_input("Enter your custom query for the media:", placeholder="e.g., What happens at timestamp 00:15 in this video?")

# Universal Media Uploader (Images + Videos)
uploaded_file = st.file_uploader(
    "Upload Media File (Images: JPG, PNG | Videos: MP4, MOV, AVI)",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        if file_type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview", use_column_width=True)
        else:
            st.video(uploaded_file)
            st.caption("Uploaded Video Preview")
        
    with col2:
        if st.button("Run Multimodal Analysis", type="primary"):
            if not gemini_api_key:
                st.error("Missing API Key! Please enter your Gemini API Key in the sidebar.")
            else:
                # Map selections to precise prompts
                if analysis_mode == "Comprehensive Object & Scene Inventory":
                    prompt = "Analyze this media thoroughly. Identify and list all key objects, people, structures, vehicles, or items present, detailing their roles or movements."
                elif analysis_mode == "Defect, Flaw or Anomaly Detection":
                    prompt = "Examine this media meticulously for any visible defects, damages, anomalies, wear-and-tear, or irregularities. Report each finding with potential severity."
                elif analysis_mode == "Action/Event Breakdown (Best for Videos)":
                    prompt = "Provide a chronological sequence of events, actions, and key movements shown in this media asset."
                else:
                    prompt = custom_prompt if custom_prompt else "Describe everything visible in this media file in detail."

                with st.spinner("Processing media file with Gemini Flash... (Videos may take a moment to upload and analyze)"):
                    try:
                        # Reset file pointer to beginning just in case
                        uploaded_file.seek(0)
                        analysis_result = analyze_media_with_gemini(uploaded_file, file_type, prompt, gemini_api_key)
                        st.success("Analysis Complete!")
                        st.markdown("### 📊 Inspection & Detection Report")
                        st.markdown(analysis_result)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
