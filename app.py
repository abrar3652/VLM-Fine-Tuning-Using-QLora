import streamlit as st
import torch
from PIL import Image
import time
import os

# Robust imports to avoid the AutoModelForVision2Seq issue
import transformers
from transformers import AutoProcessor, AutoModel

# Check for PEFT (LoRA)
try:
    from peft import PeftModel
except ImportError:
    st.error("PEFT library not found. Please add 'peft' to your requirements.txt")

# --- Page Config ---
st.set_page_config(page_title="VLM Document to Markdown", page_icon="📄", layout="wide")

# --- Constants ---
# Using the SmolVLM from your notebook
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
    # Note: Using AutoModel.from_pretrained is safer across different transformer versions
    try:
        model = AutoModel.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        )
    except Exception as e:
        st.error(f"Error loading base model: {e}")
        return None, None, device

    # 3. Load LoRA Adapters from final_checkpoint.pt
    if os.path.exists(CHECKPOINT_PATH):
        try:
            # If the checkpoint is a full state dict
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            
            # If it was saved using peft.save_pretrained, it should be a folder.
            # If it's a single .pt file, we try to load the state dict:
            if isinstance(checkpoint, dict) and "base_model" not in str(checkpoint.keys()):
                 model.load_state_dict(checkpoint, strict=False)
            else:
                model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
            
            st.sidebar.success("✅ Fine-tuned weights loaded!")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load adapters: {e}")
    else:
        st.sidebar.info("💡 Running with base model (checkpoint not found).")

    return processor, model, device

# --- UI Layout ---
st.title("📄 Document to Markdown AI")
st.markdown("Convert your document images into clean, structured Markdown using a fine-tuned Vision Language Model.")

with st.sidebar:
    st.header("Model Info")
    st.write(f"**Base:** {BASE_MODEL_ID}")
    st.write(f"**Device:** {'GPU 🚀' if torch.cuda.is_available() else 'CPU 🐌'}")
    if st.button("🔄 Refresh Model"):
        st.cache_resource.clear()
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        generate_btn = st.button("✨ Convert to Markdown", type="primary", use_container_width=True)

with col2:
    st.subheader("Markdown Output")
    if uploaded_file and generate_btn:
        processor, model, device = load_vlm_model()
        
        if model:
            with st.spinner("Analyzing document structure..."):
                # Prepare Inputs (ChatML format as per assignment)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Extract the text from this image and return it in valid Markdown format. Include tables and headers."}
                        ]
                    }
                ]
                
                # Processing
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
                
                # Generation
                start_time = time.time()
                generated_ids = model.generate(**inputs, max_new_tokens=500)
                output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Cleanup output (remove prompt instructions)
                final_output = output_text.split("assistant")[-1].strip() if "assistant" in output_text else output_text
                
                duration = time.time() - start_time
                st.caption(f"Generated in {duration:.2f} seconds")
                
                # Display results
                tab1, tab2 = st.tabs(["Preview", "Raw Markdown"])
                with tab1:
                    st.markdown(final_output)
                with tab2:
                    st.code(final_output, language="markdown")
                
                st.download_button("Download .md", final_output, file_name="output.md")
    else:
        st.info("Upload an image and click 'Convert' to see the result.")