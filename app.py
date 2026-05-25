import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────────────────────────────────
# B&G IT Consulting – The Chat Purifier
# Pipeline 1: Toxicity Detection (local fine-tuned model)
# Pipeline 2: Humorous Rewriting (HF Inference API – pre‑trained LLM)
# ─────────────────────────────────────────────────────────────────────────

# ------------- Session state for API key -------------
if "hf_token" not in st.session_state:
    st.session_state.hf_token = None

# ------------- Load classifier once -------------
@st.cache_resource(show_spinner=False)
def load_toxicity_classifier():
    model_name = "RachelHu123/toxic-comment-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,               # 0=non‑toxic, 1=toxic
        ignore_mismatched_sizes=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_toxicity_classifier()

# ------------- Toxicity check -------------
def is_toxic(text: str) -> tuple:
    """Returns (is_toxic_bool, confidence_float)."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    toxic_prob = probs[0, 1].item()   # label 1 = toxic
    return toxic_prob >= 0.5, toxic_prob

# ------------- Rewrite via Inference API -------------
def rewrite_message(original: str) -> str:
    """Use a pre‑trained LLM to transform toxic text into humorous polite version."""
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.2",   # Change to your chosen model
        token=st.session_state.hf_token,
    )
    prompt = (
        "You are a sophisticated British butler. "
        "Transform the following toxic chat message into an extremely polite, "
        "poetic, and humorous compliment. Do not keep any offensive words.\n\n"
        f"Original: {original}\n\n"
        "Rewritten:"
    )
    response = client.text_generation(prompt, max_new_tokens=80, temperature=0.7)
    return response.strip()

# ─────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat Purifier", layout="centered")
st.title("🎮 The Chat Purifier")
st.caption("Pipeline 1 → Toxicity Detection | Pipeline 2 → Humorous Rewriting")

# Sidebar for API token
with st.sidebar:
    st.header("⚙️ Hugging Face Token")
    token_input = st.text_input(
        "Enter your HF token (required for the rewriting model):",
        type="password",
    )
    if token_input:
        st.session_state.hf_token = token_input
        st.success("Token set ✅")
    else:
        st.warning("⚠️ Token required for Pipeline 2 (rewriting).")
    st.markdown("[Get your token here](https://huggingface.co/settings/tokens)")

# Main input
user_input = st.text_area("Enter a player chat message:", height=100)

if st.button("Purify"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # --- Pipeline 1: Toxicity check ---
        toxic_flag, confidence = is_toxic(user_input)

        if not toxic_flag:
            st.success("✅ Message is non‑toxic. Delivered unchanged.")
            st.write(f"> {user_input}")
        else:
            st.error(f"🚫 Toxicity detected (confidence: {confidence:.1%})")
            # --- Pipeline 2: Rewrite ---
            if not st.session_state.hf_token:
                st.error("Cannot rewrite: Hugging Face token missing. Please enter it in the sidebar.")
            else:
                with st.spinner("British butler is polishing the message…"):
                    try:
                        clean_version = rewrite_message(user_input)
                        st.success("✨ Toxic message neutralised and rewritten:")
                        st.write(f"> {clean_version}")
                    except Exception as e:
                        st.error(f"Rewriting failed: {e}")
