import streamlit as st
import torch
import librosa
import numpy as np
import io
import pandas as pd
import json
from transformers import pipeline
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt

# --- SCHEMA DEFINITION (MATCHING PDF TEMPLATE) ---
class ClinicalHistory(BaseModel):
    full_name: str = ""
    age: str = ""
    sex: str = ""
    dob: str = ""
    mrn_id: str = ""
    phone: str = ""
    address: str = ""
    occupation: str = ""
    marital_status: str = ""
    date_of_visit: str = ""
    
    chief_complaint: str = ""
    duration: str = ""
    
    hpi: str = ""
    pmh: str = ""
    medication_history: str = ""
    allergy_history: str = ""
    family_history: str = ""
    social_history: str = ""
    
    ros_general: str = ""
    ros_heent: str = ""
    ros_cardiovascular: str = ""
    ros_respiratory: str = ""
    ros_gastrointestinal: str = ""
    ros_genitourinary: str = ""
    ros_neurological: str = ""
    ros_musculoskeletal: str = ""
    ros_endocrine: str = ""
    ros_psychiatric: str = ""
    ros_skin: str = ""
    ros_hematologic: str = ""
    
    pe_general: str = ""
    pe_bp: str = ""
    pe_hr: str = ""
    pe_rr: str = ""
    pe_temp: str = ""
    pe_spo2: str = ""
    pe_weight: str = ""
    pe_height: str = ""
    pe_bmi: str = ""
    pe_sys_heent: str = ""
    pe_sys_cardio: str = ""
    pe_sys_resp: str = ""
    pe_sys_abd: str = ""
    pe_sys_gu: str = ""
    pe_sys_neuro: str = ""
    pe_sys_msk: str = ""
    pe_sys_skin: str = ""
    pe_sys_endo: str = ""
    pe_sys_psych: str = ""
    
    investigations: str = ""
    assessment: str = ""
    plan: str = ""
    follow_up: str = ""

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Amharic Medical Scribe", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Left-align buttons in the sidebar */
    [data-testid="stSidebar"] button div {
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] button p {
        text-align: left !important;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Patient Records")
    st.button("Abebe Kebede - 2026-03-24", use_container_width=True)
    st.button("Aster Mamo - 2026-03-23", use_container_width=True)
    st.button("Chala Demisse - 2026-03-20", use_container_width=True)
    st.button("Sara Tadesse - 2026-03-18", use_container_width=True)
    
    st.divider()
    api_key = st.text_input("Gemini API Key", type="password")

# --- MODEL INITIALIZATION ---
@st.cache_resource(show_spinner="Loading ASR Model...")
def load_asr_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline(
        task="automatic-speech-recognition",
        model="agkphysics/wav2vec2-large-xlsr-53-amharic",
        device=device
    )

asr_pipe = load_asr_model()

# --- AUDIO CHUNKING & TIME SYNC LOGIC ---
def transcribe_in_chunks(speech, sr, pipe, max_duration_sec=20):
    intervals = librosa.effects.split(speech, top_db=40)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_start_sample = 0
    
    for start, end in intervals:
        pad = int(0.2 * sr)
        s = max(0, start - pad)
        e = min(len(speech), end + pad)
        segment = speech[s:e]
        
        # Mark the start time of the new chunk
        if len(current_chunk) == 0:
            chunk_start_sample = s
            
        if current_length + len(segment) > max_duration_sec * sr and current_length > 0:
            chunks.append({
                "audio": np.concatenate(current_chunk),
                "start": chunk_start_sample,
                "end": s
            })
            current_chunk = [segment]
            current_length = len(segment)
            chunk_start_sample = s
        else:
            current_chunk.append(segment)
            current_length += len(segment)
            
    if current_chunk:
        chunks.append({
            "audio": np.concatenate(current_chunk),
            "start": chunk_start_sample,
            "end": len(speech)
        })
        
    full_text_blocks = []
    for c in chunks:
        res = pipe(c["audio"])
        if res and "text" in res:
            start_s = int(c["start"] / sr)
            end_s = int(c["end"] / sr)
            # Format as [MM:SS - MM:SS]
            time_str = f"[{start_s//60:02d}:{start_s%60:02d} - {end_s//60:02d}:{end_s%60:02d}]"
            full_text_blocks.append(f"{time_str} {res['text']}")
            
    return "\n\n".join(full_text_blocks)

# --- GEMINI API CALL ---
@retry(
    wait=wait_exponential(multiplier=2, min=3, max=10), 
    stop=stop_after_attempt(3),
    reraise=True
)
def process_with_gemini(transcription: str, key: str):
    client = genai.Client(api_key=key)
    
    prompt = f"""
    Role: Medical Scribe for Ethiopian Healthcare.
    Task: Convert the raw Amharic transcription into a structured General Practice Clinical History in English.
    
    Rules:
    1. Translate medical terms to precise clinical English.
    2. Extract and organize the information into the exact provided JSON schema fields.
    3. If information for a field is not mentioned, leave the string empty ("").
    4. Handle mixed Amharic/English naturally.
    5. CRITICAL: The transcription includes timestamps like [01:20 - 01:40]. Whenever you extract a symptom, complaint, or observation, APPEND the relevant timestamp to the end of your sentence in the output field so the doctor can reference the audio. 
       Example: "Patient reports severe abdominal pain for 3 days [00:15 - 00:35]."

    Transcription:
    {transcription}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClinicalHistory,
            temperature=0.1
        )
    )
    
    return json.loads(response.text)

# --- EXPORT HELPERS ---
def generate_html_report(data: dict) -> str:
    """Generates a styled HTML document. Empty values are marked in red."""
    
    def format_val(val):
        if not val or val.strip() == "":
            return '<span style="color: red; font-style: italic;">[Not Specified]</span>'
        return val

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; color: #333; }}
            h1 {{ text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            .section {{ margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; width: 30%; }}
        </style>
    </head>
    <body>
        <h1>GENERAL PRACTICE CLINICAL HISTORY</h1>
        
        <h2>1. PATIENT INFORMATION</h2>
        <table>
            <tr><th>Full Name</th><td>{format_val(data.get('full_name'))}</td></tr>
            <tr><th>Age</th><td>{format_val(data.get('age'))}</td></tr>
            <tr><th>Sex</th><td>{format_val(data.get('sex'))}</td></tr>
            <tr><th>DOB</th><td>{format_val(data.get('dob'))}</td></tr>
            <tr><th>MRN/ID</th><td>{format_val(data.get('mrn_id'))}</td></tr>
            <tr><th>Phone</th><td>{format_val(data.get('phone'))}</td></tr>
            <tr><th>Address</th><td>{format_val(data.get('address'))}</td></tr>
            <tr><th>Date of Visit</th><td>{format_val(data.get('date_of_visit'))}</td></tr>
        </table>

        <h2>2. CHIEF COMPLAINT (CC)</h2>
        <p><strong>Complaint:</strong> {format_val(data.get('chief_complaint'))}</p>
        <p><strong>Duration:</strong> {format_val(data.get('duration'))}</p>

        <h2>3. HISTORY OF PRESENT ILLNESS (HPI)</h2>
        <p>{format_val(data.get('hpi'))}</p>

        <h2>4. PAST MEDICAL HISTORY & MEDICATIONS</h2>
        <p><strong>PMH:</strong> {format_val(data.get('pmh'))}</p>
        <p><strong>Medications:</strong> {format_val(data.get('medication_history'))}</p>
        <p><strong>Allergies:</strong> {format_val(data.get('allergy_history'))}</p>

        <h2>5. FAMILY & SOCIAL HISTORY</h2>
        <p><strong>Family:</strong> {format_val(data.get('family_history'))}</p>
        <p><strong>Social:</strong> {format_val(data.get('social_history'))}</p>

        <h2>6. REVIEW OF SYSTEMS (ROS)</h2>
        <table>
            <tr><th>General</th><td>{format_val(data.get('ros_general'))}</td></tr>
            <tr><th>Cardiovascular</th><td>{format_val(data.get('ros_cardiovascular'))}</td></tr>
            <tr><th>Respiratory</th><td>{format_val(data.get('ros_respiratory'))}</td></tr>
            <tr><th>Gastrointestinal</th><td>{format_val(data.get('ros_gastrointestinal'))}</td></tr>
            <tr><th>Neurological</th><td>{format_val(data.get('ros_neurological'))}</td></tr>
        </table>

        <h2>7. PHYSICAL EXAMINATION</h2>
        <p><strong>General:</strong> {format_val(data.get('pe_general'))}</p>
        <p><strong>Vitals:</strong> BP: {format_val(data.get('pe_bp'))} | HR: {format_val(data.get('pe_hr'))} | RR: {format_val(data.get('pe_rr'))} | Temp: {format_val(data.get('pe_temp'))}</p>

        <h2>8. ASSESSMENT & PLAN</h2>
        <p><strong>Investigations:</strong> {format_val(data.get('investigations'))}</p>
        <p><strong>Assessment:</strong> {format_val(data.get('assessment'))}</p>
        <p><strong>Management Plan:</strong> {format_val(data.get('plan'))}</p>
        <p><strong>Follow-Up:</strong> {format_val(data.get('follow_up'))}</p>

    </body>
    </html>
    """
    return html

col_upload, col_record = st.columns(2)
with col_upload:
    audio_file = st.file_uploader("📁", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
with col_record:
    recorded_audio = st.audio_input("🎙️", label_visibility="collapsed")

audio_value = audio_file or recorded_audio

if audio_value is not None:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
        st.stop()

    with st.spinner("Transcribing and Structuring Note..."):
        audio_bytes = audio_value.read()
        speech, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # This will now include [MM:SS - MM:SS] tags
        transcription = transcribe_in_chunks(speech, sr, asr_pipe)
        
        try:
            # Storing in session state so it persists if the user clicks a download button
            if 'structured_data' not in st.session_state:
                st.session_state.structured_data = process_with_gemini(transcription, api_key)
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    structured_data = st.session_state.structured_data

    with st.expander("View Time-Synced Transcription"):
        # Display the formatted chunks
        st.text(transcription)

    st.divider()

    # --- EDITABLE CLINICAL HISTORY FORM ---
    st.header("GENERAL PRACTICE CLINICAL HISTORY FORM")
    st.caption("Review and edit the fields below. Changes will be reflected in your exports.")
    
    # We use a copy to capture edits
    edited_data = structured_data.copy()

    with st.container(border=True):
        st.subheader("1. PATIENT INFORMATION")
        col1, col2, col3 = st.columns(3)
        edited_data['full_name'] = col1.text_input("Full Name", value=structured_data.get('full_name', ''))
        edited_data['age'] = col2.text_input("Age", value=structured_data.get('age', ''))
        edited_data['sex'] = col3.text_input("Sex", value=structured_data.get('sex', ''))
        
        col4, col5, col6 = st.columns(3)
        edited_data['dob'] = col4.text_input("DOB", value=structured_data.get('dob', ''))
        edited_data['mrn_id'] = col5.text_input("MRN/ID", value=structured_data.get('mrn_id', ''))
        edited_data['phone'] = col6.text_input("Phone Number", value=structured_data.get('phone', ''))
        
        col7, col8, col9 = st.columns(3)
        edited_data['address'] = col7.text_input("Address", value=structured_data.get('address', ''))
        edited_data['occupation'] = col8.text_input("Occupation", value=structured_data.get('occupation', ''))
        edited_data['marital_status'] = col9.text_input("Marital Status", value=structured_data.get('marital_status', ''))
        
        edited_data['date_of_visit'] = st.text_input("Date of Visit", value=structured_data.get('date_of_visit', ''))

    with st.container(border=True):
        st.subheader("2. CHIEF COMPLAINT (CC)")
        edited_data['chief_complaint'] = st.text_input("Main complaint", value=structured_data.get('chief_complaint', ''))
        edited_data['duration'] = st.text_input("Duration", value=structured_data.get('duration', ''))

    with st.container(border=True):
        st.subheader("3. HISTORY OF PRESENT ILLNESS (HPI)")
        edited_data['hpi'] = st.text_area("Narrative description", value=structured_data.get('hpi', ''), height=100)

    # Note: Shortened remaining UI elements for brevity, but you get the pattern:
    with st.container(border=True):
        st.subheader("4. PAST MEDICAL HISTORY (PMH)")
        edited_data['pmh'] = st.text_area("Previous illnesses, hospitalizations, etc.", value=structured_data.get('pmh', ''))

    with st.container(border=True):
        st.subheader("12. ASSESSMENT / DIAGNOSIS")
        edited_data['assessment'] = st.text_area("Assessment", value=structured_data.get('assessment', ''))

    with st.container(border=True):
        st.subheader("13. MANAGEMENT PLAN")
        edited_data['plan'] = st.text_area("Plan", value=structured_data.get('plan', ''))

    st.divider()

    # --- EXPORT OPTIONS ---
    st.subheader("Export Document")
    
    dl_col1, dl_col2 = st.columns(2)
    
    # 1. HTML Download (Print to PDF)
    html_out = generate_html_report(edited_data)
    with dl_col1:
        st.download_button(
            label="Download HTML",
            data=html_out,
            file_name="clinical_history.html",
            mime="text/html",
            use_container_width=True
        )
        st.caption("Open the file in your browser and select 'Print > Save as PDF'. Empty fields will be highlighted in red.")

    # 2. CSV Download
    # Convert dict to a 2-column DataFrame
    df_csv = pd.DataFrame(list(edited_data.items()), columns=['Field', 'Value'])
    csv_out = df_csv.to_csv(index=False).encode('utf-8')
    
    with dl_col2:
        st.download_button(
            label="Download as CSV",
            data=csv_out,
            file_name="clinical_history.csv",
            mime="text/csv",
            use_container_width=True
        )