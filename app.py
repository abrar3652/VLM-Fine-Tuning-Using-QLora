import streamlit as st
import torch
from PIL import Image
import time
import os
from transformers import AutoProcessor, AutoModel

# Check for PEFT (LoRA)
try:
    from peft import PeftModel
except ImportError:
    st.error("PEFT library not found. Please add 'peft' to your requirements.txt")

# --- Page Config ---
st.set_page_config(page_title="VLM Document to Markdown", page_icon="📄", layout="wide")

# --- Constants ---
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct" 
CHECKPOINT_PATH = "final_checkpoint.pt"

# --- Model Loading ---
@st.cache_resource
def load_vlm_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Processor
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    except Exception as e:
        st.error(f"Error loading processor: {e}")
        return None, None, device

    # 2. Load Base Model 
    try:
        # UPDATED: 'torch_dtype' is now simply 'dtype' in Transformers 5.x+
        model = AutoModel.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        st.error(f"Error loading base model: {e}")
        return None, None, device

    # 3. Load LoRA Adapters
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "base_model" not in str(checkpoint.keys()):
                 model.load_state_dict(checkpoint, strict=False)
            else:
                model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
            st.sidebar.success("✅ Fine-tuned weights loaded!")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Adapter warning: {e}")
    
    return processor, model, device

# --- UI Layout ---
st.title("📄 Document to Markdown AI")

with st.sidebar:
    st.header("Settings")
    # Tip: For HF_TOKEN error, set it in your local environment or Streamlit Secrets
    if "HF_TOKEN" not in os.environ:
        st.warning("HF_TOKEN not found. Downloads might be slow.")
    
    if st.button("🔄 Clear Model Cache", width='stretch'):
        st.cache_resource.clear()
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload document (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # UPDATED: 'use_container_width' replaced with width='stretch'
        st.image(image, width='stretch')
        generate_btn = st.button("✨ Convert to Markdown", type="primary", width='stretch')

with col2:
    st.subheader("Markdown Output")
    if uploaded_file and generate_btn:
        processor, model, device = load_vlm_model()
        
        if model:
            with st.spinner("Processing document..."):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Convert this document image into structured Markdown."}
                        ]
                    }
                ]
                
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
                
                start_time = time.time()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=500)
                
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                final_output = output_text.split("assistant")[-1].strip() if "assistant" in output_text else output_text
                
                st.caption(f"Done in {time.time() - start_time:.2f}s")
                
                tab1, tab2 = st.tabs(["Preview", "Code"])
                with tab1: st.markdown(final_output)
                with tab2: st.code(final_output, language="markdown")
                st.download_button("Download .md", final_output, file_name="output.md", width='stretch')