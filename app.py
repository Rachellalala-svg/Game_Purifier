# app.py
# Streamlit deployment for: Toxic Chat Detection (fine-tuned) + Humorous Rewriting (GPT2-XL)

import streamlit as st
from transformers import pipeline
import torch

# -------------------------------------------------------------------
# 1. Cached model loaders (improves performance on Streamlit Cloud)
# -------------------------------------------------------------------
@st.cache_resource
def load_toxicity_detector():
    """
    Pipeline 1: Fine-tuned toxic comment classifier.
    Replace 'RachelHu123/toxic-comment-finetuned' with your own HF model ID.
    """
    return pipeline(
        "text-classification",
        model="RachelHu123/toxic-comment-finetuned",   # Your fine-tuned model
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource
def load_rewriter():
    """
    Pipeline 2: GPT2-XL (1.5B) for humorous rewriting.
    Falls back to CPU if GPU not available – works on Streamlit Cloud.
    """
    return pipeline(
        "text-generation",
        model="gpt2-xl",
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------
def is_toxic(detector_output: dict) -> bool:
    """
    Convert pipeline output to boolean.
    The fine-tuned model returns labels 'toxic' or 'non-toxic'.
    """
    return detector_output['label'].lower() == 'toxic'

def rewrite_toxic_message(rewriter, toxic_text: str) -> str:
    """
    Use GPT2-XL to rewrite a toxic message into a humorous, polite compliment.
    Prompt engineering ensures an elegant British butler style.
    """
    prompt = (
        f"You are an elegant British butler. Rewrite the following toxic gaming chat "
        f"into a hilarious, polite compliment. Do NOT keep any offensive words.\n"
        f"Toxic chat: \"{toxic_text}\"\n"
        f"Polite & funny rewrite:"
    )
    # Generate with controlled length and creativity
    output = rewriter(
        prompt,
        max_new_tokens=40,
        temperature=0.85,
        do_sample=True,
        top_p=0.95,
        pad_token_id=rewriter.tokenizer.eos_token_id,
        truncation=True
    )
    # Extract only the newly generated part (remove the original prompt)
    generated = output[0]['generated_text'][len(prompt):].strip()
    # Fallback if generation is empty
    if not generated:
        generated = "I do believe a spot of kindness would brighten this conversation, old chap."
    return generated

# -------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Toxic Chat Purifier", page_icon="🧹")
st.title("🧹 Game Chat Purifier: Toxicity → Humor")
st.markdown("Detects toxic messages and rewrites them into hilarious, polite compliments.")

# Load models (spinner provides feedback during loading)
with st.spinner("Loading AI models (first time may take 30-60 seconds)..."):
    toxicity_model = load_toxicity_detector()
    rewriter_model = load_rewriter()
st.success("Models ready! ✅")

# User input
user_input = st.text_area("Enter a chat message:", height=100, 
                          placeholder="e.g., You're such a noob, uninstall the game!")

if st.button("Purify Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # ----- Pipeline 1: Toxicity Detection -----
        with st.spinner("Analyzing message..."):
            result = toxicity_model(user_input)[0]   # first (and only) prediction
            toxic_flag = is_toxic(result)

        # Display detection result
        col1, col2 = st.columns(2)
        col1.metric("Pipeline 1 Decision", "🚨 TOXIC" if toxic_flag else "✅ CLEAN")
        col2.metric("Confidence", f"{result['score']:.2%}")

        # ----- Pipeline 2: Rewriting (only if toxic) -----
        if toxic_flag:
            with st.spinner("Generating humorous rewrite (GPT2-XL)..."):
                rewritten = rewrite_toxic_message(rewriter_model, user_input)
            st.subheader("🧙‍♂️ Humorous Rewrite (sent to chat)")
            st.success(rewritten)
            st.caption("The original toxic message was **blocked** and replaced with the above.")
        else:
            st.subheader("✅ Message is clean")
            st.info(user_input)
            st.caption("No transformation applied – message passed through as is.")

# -------------------------------------------------------------------
# 4. Sidebar with instructions & model info
# -------------------------------------------------------------------
st.sidebar.header("How it works")
st.sidebar.markdown("""
1. **Pipeline 1 (Fine-tuned model)**  
   Detects toxic / offensive chat.  
   Model: `RachelHu123/toxic-comment-finetuned` (based on `martin-ha/toxic-comment-model`)

2. **Pipeline 2 (GPT2-XL 1.5B)**  
   Rewrites toxic messages into polite, humorous compliments (British butler style).

**Flow:**  
User input → Toxicity classifier → if toxic → GPT2-XL rewrite → display funny message.  
Normal messages are left untouched.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Deployment** – [Streamlit Cloud](https://streamlit.io/cloud) | [GitHub](https://github.com/)")
