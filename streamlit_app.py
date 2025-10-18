import streamlit as st
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- GLOBAL CONFIGURATION ---
warnings.filterwarnings('ignore')
# This path MUST match the folder name you downloaded from Colab
MODEL_SAVE_PATH = "./final_t5_summarizer" 
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150

# --- 1. MODEL LOADING (Cached for Speed) ---

@st.cache_resource
def load_trained_model():
    """Loads the T5 model and tokenizer saved after training."""
    st.info(f"Loading trained T5 model from disk... (This may take a moment the first time)")
    
    try:
        # Load the fine-tuned T5 model and tokenizer from the saved directory
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SAVE_PATH)
        
        # Determine the best device (CPU or Mac's Metal/MPS GPU)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model! Please ensure the folder '{MODEL_SAVE_PATH}' is in the same directory as 'streamlit_app.py'.")
        st.caption("Detailed Error:")
        st.code(str(e))
        st.stop()
        
# --- 2. SUMMARIZATION FUNCTION ---

def generate_summary(tokenizer, model, device, article_text):
    """Generates a summary using the loaded T5 model."""
    
    # 1. Preprocess the input (add the T5 prefix)
    input_text = "summarize: " + article_text
    
    # 2. Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True
    ).to(device) # Move tensor to the correct device

    # 3. Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,              
        max_length=MAX_TARGET_LENGTH,
        min_length=30,
        length_penalty=2.0,       
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # 4. Decode and return
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return summary

# --- 3. STREAMLIT APP LAYOUT ---

st.set_page_config(
    page_title="T5 Few-Shot Summarizer Demo", 
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("📰 T5 Few-Shot News Summarizer")
st.markdown("A demonstration of the T5-small model fine-tuned on **$K=80$** CNN/DailyMail examples.")

# Load Model/Tokenizer (cached)
tokenizer, model, device = load_trained_model()

with st.container():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Source Article Input")
        
        # Example Article Text for easy demo
        default_article = (
            "The James Webb Space Telescope has detected water vapor in the atmosphere "
            "of an exoplanet orbiting a distant star, suggesting that the search for "
            "potentially habitable worlds is heating up. The planet, named WASP-96b, is "
            "a hot gas giant located about 1,150 light-years away. While WASP-96b itself is "
            "far too hot and large to support life, the detailed analysis of its atmosphere "
            "proves the telescope's capability to measure the subtle chemical signatures "
            "of water, methane, and carbon dioxide in exoplanet atmospheres. Scientists "
            "believe this breakthrough paves the way for future studies on smaller, cooler, "
            "and potentially Earth-like planets where liquid water could exist. The discovery was "
            "announced today at a press conference by NASA officials."
        )

        article_input = st.text_area(
            "Paste your news article here:",
            value=default_article,
            height=300
        )
        
        # Summarize Button
        if st.button("Generate Summary", type="primary", use_container_width=True):
            if article_input:
                with st.spinner(f"Model is running inference on {device.type}..."):
                    summary_output = generate_summary(tokenizer, model, device, article_input)
                
                with col2:
                    st.subheader("Generated Summary")
                    st.success(summary_output)
            else:
                st.warning("Please paste an article to summarize.")

    with col2:
        st.subheader("Summary Output Area")
        st.info("The summary will appear here after clicking 'Generate Summary'.")

st.markdown("---")
st.caption(f"Model Loaded to: **{device.type}**. T5-small fine-tuned for 15 epochs.")