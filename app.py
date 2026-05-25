import streamlit as st
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────────────────────────────────
# B&G IT Consulting – The Chat Purifier
# Pipeline 1: Toxicity Detection (fine‑tuned model on HF)
# Pipeline 2: Humorous Rewriting (GPT-2 XL via HF Inference API)
# ─────────────────────────────────────────────────────────────────────────

# ---------- Token management (Streamlit secrets or sidebar) ----------
def get_hf_token() -> str:
    """Return HF token from secrets (cloud) or sidebar (local)."""
    # Try to load from Streamlit Cloud secrets first
    try:
        return st.secrets["HUGGINGFACE_TOKEN"]
    except (KeyError, FileNotFoundError):
        # Fallback to sidebar input
        return st.session_state.get("hf_token", None)

# ---------- Load toxicity classifier once ----------
@st.cache_resource(show_spinner=False)
def load_toxicity_classifier():
    model_name = "RachelHu123/toxic-comment-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,               # 0 = non‑toxic, 1 = toxic
        ignore_mismatched_sizes=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_toxicity_classifier()

# ---------- Toxicity check ----------
def is_toxic(text: str) -> tuple:
    """Return (is_toxic_bool, confidence_float)."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    toxic_prob = probs[0, 1].item()
    return toxic_prob >= 0.5, toxic_prob

# ---------- Rewrite with GPT-2 XL (few‑shot) ----------
def rewrite_message(original: str) -> str:
    """Use GPT-2 XL with few‑shot examples to produce polite/humorous rewrite."""
    token = get_hf_token()
    if not token:
        raise RuntimeError("Hugging Face token missing.")

    client = InferenceClient(model="gpt2-xl", token=token)

    # ⚠️ This few‑shot prompt replicates the pattern that produced your test outputs.
    # Keep exactly the same separators and format.
    prompt = (
        "Toxic: bot lane is feeding non stop, gg close the game\n"
        "Polite: botlane tactical pressure detected\n"
        "###\n"
        "Toxic: top gap is massive, our top is dog trash\n"
        "Polite: toplane efficiency variance noticed\n"
        "###\n"
        "Toxic: why no baron? report this idiot team\n"
        "Polite: baron control synchronization struggling\n"
        "###\n"
        f"Toxic: {original}\n"
        "Polite:"
    )

    response = client.text_generation(
        prompt,
        max_new_tokens=30,
        temperature=0.7,
        stop_sequences=["###", "\n"]
    )
    # Remove any trailing newlines / extra spaces
    return response.strip()

# ─────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chat Purifier", layout="centered")
st.title("🎮 The Chat Purifier")
st.caption("Pipeline 1 → Fine‑tuned Toxic‑BERT | Pipeline 2 → GPT‑2 XL Humorous Rewriter")

# --- Sidebar for token (only shown if not set in secrets) ---
with st.sidebar:
    st.header("⚙️ Hugging Face Token")
    # If token is already in secrets, don't show the input
    if "HUGGINGFACE_TOKEN" in st.secrets:
        st.success("Token loaded from secrets ✅")
    else:
        token_input = st.text_input(
            "Enter your HF token (required for GPT-2 XL):",
            type="password",
            key="hf_token"
        )
        if token_input:
            st.success("Token set ✅")
        else:
            st.warning("⚠️ Token required for Pipeline 2 (rewriting).")
        st.markdown("[Get your token here](https://huggingface.co/settings/tokens)")

# --- Main input area ---
user_input = st.text_area("Enter a player chat message:", height=100)

if st.button("Purify"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # ---------- Pipeline 1 ----------
        toxic_flag, confidence = is_toxic(user_input)

        if not toxic_flag:
            st.success("✅ Message is non‑toxic. Delivered unchanged.")
            st.write(f"> {user_input}")
        else:
            st.error(f"🚫 Toxicity detected (confidence: {confidence:.1%})")

            # ---------- Pipeline 2 ----------
            token = get_hf_token()
            if not token:
                st.error("Cannot rewrite: Hugging Face token missing. Please set it in the sidebar or secrets.")
            else:
                with st.spinner("GPT‑2 XL is polishing the message…"):
                    try:
                        clean_version = rewrite_message(user_input)
                        st.success("✨ Toxic message neutralised and rewritten:")
                        st.write(f"> {clean_version}")
                    except Exception as e:
                        st.error(f"Rewriting failed: {e}")
