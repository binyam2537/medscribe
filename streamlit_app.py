import streamlit as st
import torch
import librosa
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

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("Patient Records")
    # Mock list of patient records
    st.button("Abebe Kebede - 2026-03-24", use_container_width=True)
    st.button("Aster Mamo - 2026-03-23", use_container_width=True)
    st.button("Chala Demisse - 2026-03-20", use_container_width=True)
    st.button("Sara Tadesse - 2026-03-18", use_container_width=True)
    
    st.divider()
    st.caption("Settings")
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

# --- MAIN UI WORKFLOW ---
st.title("Clinical History Scribe")

# Minimal Audio Input Section
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
        # Transcribe
        audio_bytes = audio_value.read()
        speech, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        transcription = asr_pipe(speech)["text"]
        
        # Process with Gemini
        try:
            structured_data = process_with_gemini(transcription, api_key)
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    with st.expander("View Raw Transcription"):
        st.write(transcription)

    st.divider()

    # --- EDITABLE CLINICAL HISTORY FORM ---
    st.header("GENERAL PRACTICE CLINICAL HISTORY FORM")
    
    with st.container(border=True):
        st.subheader("1. PATIENT INFORMATION")
        col1, col2, col3 = st.columns(3)
        structured_data['full_name'] = col1.text_input("Full Name", value=structured_data.get('full_name', ''))
        structured_data['age'] = col2.text_input("Age", value=structured_data.get('age', ''))
        structured_data['sex'] = col3.text_input("Sex", value=structured_data.get('sex', ''))
        
        col4, col5, col6 = st.columns(3)
        structured_data['dob'] = col4.text_input("DOB", value=structured_data.get('dob', ''))
        structured_data['mrn_id'] = col5.text_input("MRN/ID", value=structured_data.get('mrn_id', ''))
        structured_data['phone'] = col6.text_input("Phone Number", value=structured_data.get('phone', ''))
        
        col7, col8, col9 = st.columns(3)
        structured_data['address'] = col7.text_input("Address", value=structured_data.get('address', ''))
        structured_data['occupation'] = col8.text_input("Occupation", value=structured_data.get('occupation', ''))
        structured_data['marital_status'] = col9.text_input("Marital Status", value=structured_data.get('marital_status', ''))
        
        structured_data['date_of_visit'] = st.text_input("Date of Visit", value=structured_data.get('date_of_visit', ''))

    with st.container(border=True):
        st.subheader("2. CHIEF COMPLAINT (CC)")
        structured_data['chief_complaint'] = st.text_input("Main complaint", value=structured_data.get('chief_complaint', ''))
        structured_data['duration'] = st.text_input("Duration", value=structured_data.get('duration', ''))

    with st.container(border=True):
        st.subheader("3. HISTORY OF PRESENT ILLNESS (HPI)")
        structured_data['hpi'] = st.text_area("Narrative description", value=structured_data.get('hpi', ''), height=100)

    with st.container(border=True):
        st.subheader("4. PAST MEDICAL HISTORY (PMH)")
        structured_data['pmh'] = st.text_area("Previous illnesses, hospitalizations, etc.", value=structured_data.get('pmh', ''))

    with st.container(border=True):
        st.subheader("5. MEDICATION HISTORY")
        structured_data['medication_history'] = st.text_area("Current, past, OTC, herbal", value=structured_data.get('medication_history', ''))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("6. ALLERGIES")
        structured_data['allergy_history'] = st.text_area("Allergies/Reactions", value=structured_data.get('allergy_history', ''))
    with col_b:
        st.subheader("7. FAMILY HISTORY")
        structured_data['family_history'] = st.text_area("Familial/Hereditary", value=structured_data.get('family_history', ''))
    with col_c:
        st.subheader("8. SOCIAL HISTORY")
        structured_data['social_history'] = st.text_area("Lifestyle, habits, living", value=structured_data.get('social_history', ''))

    with st.container(border=True):
        st.subheader("9. REVIEW OF SYSTEMS (ROS)")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            structured_data['ros_general'] = st.text_input("General", value=structured_data.get('ros_general', ''))
            structured_data['ros_heent'] = st.text_input("HEENT", value=structured_data.get('ros_heent', ''))
            structured_data['ros_cardiovascular'] = st.text_input("Cardiovascular", value=structured_data.get('ros_cardiovascular', ''))
            structured_data['ros_respiratory'] = st.text_input("Respiratory", value=structured_data.get('ros_respiratory', ''))
            structured_data['ros_gastrointestinal'] = st.text_input("Gastrointestinal", value=structured_data.get('ros_gastrointestinal', ''))
            structured_data['ros_genitourinary'] = st.text_input("Genitourinary", value=structured_data.get('ros_genitourinary', ''))
        with r_col2:
            structured_data['ros_neurological'] = st.text_input("Neurological", value=structured_data.get('ros_neurological', ''))
            structured_data['ros_musculoskeletal'] = st.text_input("Musculoskeletal", value=structured_data.get('ros_musculoskeletal', ''))
            structured_data['ros_endocrine'] = st.text_input("Endocrine", value=structured_data.get('ros_endocrine', ''))
            structured_data['ros_psychiatric'] = st.text_input("Psychiatric", value=structured_data.get('ros_psychiatric', ''))
            structured_data['ros_skin'] = st.text_input("Skin", value=structured_data.get('ros_skin', ''))
            structured_data['ros_hematologic'] = st.text_input("Hematologic/Lymphatic", value=structured_data.get('ros_hematologic', ''))

    with st.container(border=True):
        st.subheader("10. PHYSICAL EXAMINATION")
        structured_data['pe_general'] = st.text_input("General Appearance", value=structured_data.get('pe_general', ''))
        
        st.write("**Vital Signs**")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        structured_data['pe_bp'] = v_col1.text_input("BP (mmHg)", value=structured_data.get('pe_bp', ''))
        structured_data['pe_hr'] = v_col2.text_input("HR (bpm)", value=structured_data.get('pe_hr', ''))
        structured_data['pe_rr'] = v_col3.text_input("RR (/min)", value=structured_data.get('pe_rr', ''))
        structured_data['pe_temp'] = v_col4.text_input("Temp (°C)", value=structured_data.get('pe_temp', ''))
        
        v_col5, v_col6, v_col7, v_col8 = st.columns(4)
        structured_data['pe_spo2'] = v_col5.text_input("SpO2 (%)", value=structured_data.get('pe_spo2', ''))
        structured_data['pe_weight'] = v_col6.text_input("Weight (kg)", value=structured_data.get('pe_weight', ''))
        structured_data['pe_height'] = v_col7.text_input("Height (cm)", value=structured_data.get('pe_height', ''))
        structured_data['pe_bmi'] = v_col8.text_input("BMI", value=structured_data.get('pe_bmi', ''))

        st.write("**Systemic Examination**")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            structured_data['pe_sys_heent'] = st.text_input("HEENT (Sys)", value=structured_data.get('pe_sys_heent', ''))
            structured_data['pe_sys_cardio'] = st.text_input("Cardiovascular (Sys)", value=structured_data.get('pe_sys_cardio', ''))
            structured_data['pe_sys_resp'] = st.text_input("Respiratory (Sys)", value=structured_data.get('pe_sys_resp', ''))
            structured_data['pe_sys_abd'] = st.text_input("Abdomen", value=structured_data.get('pe_sys_abd', ''))
            structured_data['pe_sys_gu'] = st.text_input("Genitourinary (Sys)", value=structured_data.get('pe_sys_gu', ''))
        with s_col2:
            structured_data['pe_sys_neuro'] = st.text_input("Neurological (Sys)", value=structured_data.get('pe_sys_neuro', ''))
            structured_data['pe_sys_msk'] = st.text_input("Musculoskeletal (Sys)", value=structured_data.get('pe_sys_msk', ''))
            structured_data['pe_sys_skin'] = st.text_input("Skin (Sys)", value=structured_data.get('pe_sys_skin', ''))
            structured_data['pe_sys_endo'] = st.text_input("Endocrine (Sys)", value=structured_data.get('pe_sys_endo', ''))
            structured_data['pe_sys_psych'] = st.text_input("Psychiatric (Sys)", value=structured_data.get('pe_sys_psych', ''))

    with st.container(border=True):
        st.subheader("11. INVESTIGATIONS")
        structured_data['investigations'] = st.text_area("Ordered tests and key findings", value=structured_data.get('investigations', ''))

    with st.container(border=True):
        st.subheader("12. ASSESSMENT / DIAGNOSIS")
        structured_data['assessment'] = st.text_area("Assessment", value=structured_data.get('assessment', ''))

    with st.container(border=True):
        st.subheader("13. MANAGEMENT PLAN")
        structured_data['plan'] = st.text_area("Plan", value=structured_data.get('plan', ''))

    with st.container(border=True):
        st.subheader("14. FOLLOW-UP")
        structured_data['follow_up'] = st.text_input("Follow-up details", value=structured_data.get('follow_up', ''))

    st.divider()

    # Save Configuration
    st.download_button(
        label="Download Clinical History (JSON)",
        data=json.dumps(structured_data, indent=2),
        file_name="clinical_history.json",
        mime="application/json"
    )