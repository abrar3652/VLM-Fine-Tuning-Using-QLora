import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="VLM Document to Markdown",
    page_icon="📄",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMarkdown {
        font-family: 'Source Code Pro', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
# UPDATE THESE to match your specific model from the notebook
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"  # Change to your base model
CHECKPOINT_PATH = "final_checkpoint.pt" # Path to your .pt file

# --- Model Loading (Cached) ---
@st.cache_resource
def load_model():
    st.info("Loading model... This may take a minute.")
    
    # Use CPU if GPU is not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load Base Model (4-bit if using QLoRA)
    model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    
    # 3. Load your fine-tuned adapters
    try:
        # If your .pt is a full model state_dict
        # model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        
        # If it's a LoRA adapter (Most common for QLoRA)
        model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
        st.success("Fine-tuned weights loaded successfully!")
    except Exception as e:
        st.warning(f"Note: Could not load adapters via Peft ({e}). Ensure CHECKPOINT_PATH points to a valid LoRA folder or file.")
    
    model.eval()
    return processor, model, device

# --- Inference Function ---
def generate_markdown(image, processor, model, device):
    # Standard ChatML prompt format as per assignment requirements
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this document image into structured Markdown format. Include tables, headings, and lists where appropriate."}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Split to get only the assistant's response (adjust based on your template)
    if "assistant" in generated_text:
        return generated_text.split("assistant")[-1].strip()
    return generated_text

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.write("VLM Document Parser v1.0")
    st.divider()
    st.markdown("""
    **Model Info:**
    - Task: Image to Markdown
    - Method: QLoRA Fine-tuning
    - Precision: 4-bit/Float16
    """)
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.rerun()

# --- Main UI ---
st.title("📄 Document-to-Markdown Generator")
st.write("Upload a document image (invoice, form, report) to convert it into structured Markdown.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Document", use_column_width=True)
        
        generate_btn = st.button("🚀 Generate Markdown", type="primary", use_container_width=True)
    else:
        st.info("Please upload an image to begin.")

with col2:
    st.subheader("Markdown Output")
    if uploaded_file is not None and generate_btn:
        processor, model, device = load_model()
        
        with st.spinner("Analyzing document and generating markdown..."):
            start_time = time.time()
            try:
                markdown_result = generate_markdown(image, processor, model, device)
                duration = time.time() - start_time
                
                # Display success and time
                st.success(f"Generated in {duration:.2f}s")
                
                # Tabs for Raw and Preview
                tab1, tab2 = st.tabs(["👁️ Preview", "💻 Raw Code"])
                with tab1:
                    st.markdown(markdown_result)
                with tab2:
                    st.code(markdown_result, language="markdown")
                
                # Download Button
                st.download_button(
                    label="📥 Download Markdown",
                    data=markdown_result,
                    file_name="converted_document.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Output will appear here after generation.")

# --- Footer ---
st.divider()
st.caption("GenAI Assignment 05 - Vision Language Models")