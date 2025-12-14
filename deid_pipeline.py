"""
Automated Clinical Text De-Identification Demo
----------------------------------------------

This version is designed to JUST WORK for a demo:

- Uses a tiny in-code demo dataset (no external files needed)
- Trains BioClinicalBERT quickly on that tiny dataset
- Provides a Streamlit app for interactive de-identification
- Regex baseline is always used for patterns (dates, phone, email, IDs)
- RAG (SentenceTransformers + FAISS) is OPTIONAL:
    - If not installed, it is automatically disabled
- Includes:
    - Manual text input mode
    - Document upload mode (.txt) with de-identified download
    - Extension hooks for radiology / X-ray reports (text now, image later)
- EXTRA:
    - AGE and GENDER classification using regex/rules
    - Patient names normalized to [PATIENT]
    - Doctor signatures normalized to [DOCTOR]
    - User-selectable PHI categories to de-identify
"""

import os
import re
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from seqeval.metrics import classification_report, f1_score

import numpy as np
import streamlit as st

# =============================
# OPTIONAL RAG IMPORTS
# =============================

try:
    from sentence_transformers import SentenceTransformer
    import faiss

    RAG_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    faiss = None
    RAG_AVAILABLE = False


# =============================
# CONFIG
# =============================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 128  # shorter for speed in demo
BATCH_SIZE = 4
LR = 3e-5
NUM_EPOCHS = 1  # 1 epoch on tiny dataset for fast training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "phi_bioclincialbert_demo.pt"

LABEL_LIST = [
    "O",
    "B-PATIENT",
    "I-PATIENT",
    "B-HOSPITAL",
    "I-HOSPITAL",
    "B-DOCTOR",
    "I-DOCTOR",
    "B-LOCATION",
    "I-LOCATION",
    "B-DATE",
    "I-DATE",
    "B-ID",
    "I-ID",
    "B-CONTACT",
    "I-CONTACT",
]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

AMBIGUOUS_LABELS = {"B-PATIENT", "I-PATIENT", "B-HOSPITAL", "I-HOSPITAL", "B-LOCATION", "I-LOCATION"}

# =============================
# REGEX PATTERNS
# =============================

DATE_PATTERN = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
EMAIL_PATTERN = r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b"
ID_PATTERN = r"\b(?:MRN|ID|Record|Acc#)\s*[:#]?\s*\w+\b"
ZIP_PATTERN = r"\b\d{5}(?:-\d{4})?\b"

# Gender patterns (for global regex)
GENDER_PATTERN = r"\b(?:male|female|man|woman|boy|girl)\b"

# Age patterns (for global regex)
AGE_PATTERN = r"\b(?:\d{1,3}\s?(?:yo|y\/o|y\.o\.|years old|year-old|yrs old|yrs|years))\b"
AGE_PATTERN_2 = r"\bage\s?\d{1,3}\b"
AGE_PATTERN_3 = r"\b\d{1,3}-year-old\b"

# Token-level patterns for age/gender classification (per token)
GENDER_TOKEN_PATTERN = r"^(?:male|female|man|woman|boy|girl)$"
AGE_TOKEN_PATTERN = r"^\d{1,3}(?:yo|y/o|y\.o\.|yrs?|years?)?$"
AGE_TOKEN_PATTERN_2 = r"^\d{1,3}-year-old$"


# =============================
# HELPER FUNCTIONS FOR RULE-BASED MASKING
# =============================

def default_phi_options() -> Dict[str, bool]:
    """Default: everything ON."""
    return {
        "patient": True,
        "doctor": True,
        "dates": True,
        "contact": True,   # phone, email, IDs, ZIP
        "age": True,
        "gender": True,
        "location": True,  # hospitals + generic locations
    }


def mask_honorific_patient_names(text: str) -> str:
    """
    Replace patterns like 'Mr. Villegas' or 'Ms Smith' with 'Mr. [PATIENT]'.
    We keep the title, but mask the surname.
    """
    pattern = re.compile(r'\b(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+([A-Z][a-zA-Z\-]+)\b')

    def repl(match):
        title = match.group(1)
        return f"{title} [PATIENT]"

    return pattern.sub(repl, text)


def mask_doctor_signatures_full(text: str) -> str:
    """
    Replace FULL clinician signature lines with [DOCTOR].
    Targets lines like:
        - Xzavian G. Tavares, M.D.
        - John R. Adams MD
        - Jane Smith, M.D.
    """
    doctor_line_pattern = re.compile(
        r"^(?:\s*)([A-Z][a-zA-Z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z]+)?\s*,?\s*M\.?D\.?)\s*$",
        re.MULTILINE,
    )
    text = doctor_line_pattern.sub("[DOCTOR]", text)
    return text


def regex_deidentify(text: str, options: Optional[Dict[str, bool]] = None) -> str:
    """
    Baseline regex de-identification.
    Respects user-selected PHI options.
    """
    if options is None:
        options = default_phi_options()

    # Dates
    if options.get("dates", True):
        text = re.sub(DATE_PATTERN, "[DATE]", text)

    # Contact info (phone, email, generic IDs, ZIP)
    if options.get("contact", True):
        text = re.sub(PHONE_PATTERN, "[PHONE]", text)
        text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
        text = re.sub(ID_PATTERN, "[ID]", text)
        text = re.sub(ZIP_PATTERN, "[ZIP]", text)

    # Gender
    if options.get("gender", True):
        text = re.sub(GENDER_PATTERN, "[GENDER]", text, flags=re.IGNORECASE)

    # Age
    if options.get("age", True):
        text = re.sub(AGE_PATTERN, "[AGE]", text, flags=re.IGNORECASE)
        text = re.sub(AGE_PATTERN_2, "[AGE]", text, flags=re.IGNORECASE)
        text = re.sub(AGE_PATTERN_3, "[AGE]", text, flags=re.IGNORECASE)

    # Patient honorifics (Mr. X)
    if options.get("patient", True):
        text = mask_honorific_patient_names(text)

    # Doctor signatures
    if options.get("doctor", True):
        text = mask_doctor_signatures_full(text)

    return text


# =============================
# TINY DEMO DATASET (IN-CODE)
# =============================

def build_tiny_demo_dataset():
    """
    Tiny in-memory dataset with PHI-style annotations.
    Used for quick demo training so everything runs
    without any external files.
    """
    sentences = [
        ["Patient", "Mary", "Coleman", "visited", "Rockford", "Medical", "Center", "on", "03/14/2021", "."],
        ["Dr.", "Smith", "evaluated", "the", "patient", "in", "Boston", "."],
        ["Contact", "number", "is", "555-123-4567", "and", "email", "is", "mary.coleman@example.com", "."],
        ["Patient", "John", "Doe", "was", "seen", "at", "Mercy", "Hospital", "on", "2020-10-05", "."],
    ]
    labels = [
        ["O", "B-PATIENT", "I-PATIENT", "O", "B-HOSPITAL", "I-HOSPITAL", "I-HOSPITAL", "O", "B-DATE", "O"],
        ["B-DOCTOR", "I-DOCTOR", "O", "O", "O", "O", "B-LOCATION", "O"],
        ["O", "O", "O", "B-CONTACT", "O", "O", "O", "B-CONTACT", "O"],
        ["O", "B-PATIENT", "I-PATIENT", "O", "O", "O", "B-HOSPITAL", "I-HOSPITAL", "O", "B-DATE", "O"],
    ]
    return sentences, labels


# =============================
# DATASET CLASS
# =============================

class PHIDataset(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], tokenizer: AutoTokenizer, max_len: int):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        labels = self.tags[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=False,
            return_tensors="pt",
            padding="max_length",
        )

        word_ids = encoding.word_ids(batch_index=0)

        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                if word_id != previous_word_id:
                    label_ids.append(LABEL2ID[labels[word_id]])
                else:
                    label_ids.append(-100)
            previous_word_id = word_id

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label_ids, dtype=torch.long)

        return item


# =============================
# MODEL CREATION
# =============================

def create_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)
    return model, tokenizer


# =============================
# TRAINING / EVALUATION
# =============================

def evaluate_model(model, data_loader: DataLoader) -> float:
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for true_seq, pred_seq in zip(labels, preds):
                true_tags = []
                pred_tags = []
                for t, p in zip(true_seq, pred_seq):
                    if t == -100:
                        continue
                    true_tags.append(ID2LABEL[int(t)])
                    pred_tags.append(ID2LABEL[int(p)])
                all_true.append(true_tags)
                all_pred.append(pred_tags)

    print("\nValidation Classification Report (tiny demo):")
    print(classification_report(all_true, all_pred))
    return f1_score(all_true, all_pred)


def train_model(
    model,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=max(1, total_steps),
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = loss_fn(active_logits, active_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_f1 = evaluate_model(model, val_loader)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

    return model


def train_and_save_model(checkpoint_path=CHECKPOINT_PATH):
    print(">>> Training demo model on tiny in-code dataset...")
    model, tokenizer = create_model_and_tokenizer()

    train_sents, train_tags = build_tiny_demo_dataset()
    val_sents, val_tags = build_tiny_demo_dataset()  # reuse tiny data for val

    train_dataset = PHIDataset(train_sents, train_tags, tokenizer, MAX_LEN)
    val_dataset = PHIDataset(val_sents, val_tags, tokenizer, MAX_LEN)

    model = train_model(model, train_dataset, val_dataset)

    print(f">>> Saving demo model to {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)
    return model, tokenizer


def load_trained_model(checkpoint_path=CHECKPOINT_PATH):
    model, tokenizer = create_model_and_tokenizer()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.to(DEVICE)
        print(f">>> Loaded trained weights from {checkpoint_path}")
    else:
        print(f">>> Checkpoint {checkpoint_path} not found; using base model (not fine-tuned).")
    return model, tokenizer


# =============================
# INFERENCE HELPERS
# =============================

def predict_tokens(model, tokenizer, text: str) -> List[Tuple[str, str, float]]:
    model.eval()
    tokens = text.split()
    if len(tokens) == 0:
        return []

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    word_ids = encoding.word_ids(batch_index=0)
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        preds = np.argmax(probs, axis=-1)

    results = []
    used_word_ids = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id in used_word_ids:
            continue
        used_word_ids.add(word_id)

        token = tokens[word_id]
        label_id = preds[idx]
        label = ID2LABEL[label_id]
        conf = float(np.max(probs[idx]))
        results.append((token, label, conf))

    return results


def mask_with_ner_and_regex(
    model,
    tokenizer,
    text: str,
    use_rag: bool = False,
    rag_resolver=None,
    options: Optional[Dict[str, bool]] = None,
) -> Tuple[str, str]:
    """
    Combined masking:
    - NER (BioClinicalBERT) for PHI entities
    - Rule-based AGE and GENDER classification
    - Respects user-selected PHI options
    """
    if options is None:
        options = default_phi_options()

    token_preds = predict_tokens(model, tokenizer, text)

    if use_rag and rag_resolver is not None:
        token_preds = rag_resolver(token_preds)

    deid_tokens = []
    html_tokens = []

    color_map = {
        "PATIENT": "#ffcccc",
        "DOCTOR": "#ffcc99",
        "HOSPITAL": "#ccffcc",
        "LOCATION": "#ccffff",
        "DATE": "#ccccff",
        "ID": "#ffccff",
        "CONTACT": "#ffffcc",
        "AGE": "#e0b0ff",
        "GENDER": "#ffc0e0",
    }

    for (tok, label, conf) in token_preds:
        # -------- RULE-BASED AGE/GENDER CLASSIFICATION (OVERRIDE) --------
        # Gender token?
        if options.get("gender", True) and re.fullmatch(GENDER_TOKEN_PATTERN, tok, flags=re.IGNORECASE):
            ent_type = "GENDER"
            placeholder = "[GENDER]"
            color = color_map.get(ent_type, "#ffd700")
            deid_tokens.append(placeholder)
            html_tokens.append(
                f'<span style="background-color:{color}; padding:2px; border-radius:3px;" '
                f'title="GENDER ({conf:.2f})">{tok}</span>'
            )
            continue

        # Age token?
        if options.get("age", True) and (
            re.fullmatch(AGE_TOKEN_PATTERN, tok, flags=re.IGNORECASE)
            or re.fullmatch(AGE_TOKEN_PATTERN_2, tok, flags=re.IGNORECASE)
        ):
            ent_type = "AGE"
            placeholder = "[AGE]"
            color = color_map.get(ent_type, "#ffd700")
            deid_tokens.append(placeholder)
            html_tokens.append(
                f'<span style="background-color:{color}; padding:2px; border-radius:3px;" '
                f'title="AGE ({conf:.2f})">{tok}</span>'
            )
            continue

        # ----------------- NER-BASED MASKING -----------------
        if label == "O":
            deid_tokens.append(tok)
            html_tokens.append(tok)
            continue

        ent_type = label.split("-", 1)[-1]

        # Determine if this entity type should be masked based on options
        should_mask = True
        if ent_type in ["PATIENT", "NAME"]:
            should_mask = options.get("patient", True)
        elif ent_type in ["DOCTOR"]:
            should_mask = options.get("doctor", True)
        elif ent_type in ["DATE"]:
            should_mask = options.get("dates", True)
        elif ent_type in ["ID"]:
            should_mask = options.get("contact", True)
        elif ent_type in ["CONTACT"]:
            should_mask = options.get("contact", True)
        elif ent_type in ["HOSPITAL", "LOCATION"]:
            should_mask = options.get("location", True)
        else:
            should_mask = True  # default for other entity types

        if not should_mask:
            # Leave token untouched
            deid_tokens.append(tok)
            html_tokens.append(tok)
            continue

        # Normalize placeholders
        if ent_type in ["PATIENT", "NAME"]:
            placeholder = "[PATIENT]"
        elif ent_type == "DATE":
            placeholder = "[DATE]"
        elif ent_type == "DOCTOR":
            placeholder = "[DOCTOR]"
        else:
            placeholder = f"[{ent_type}]"

        deid_tokens.append(placeholder)
        color = color_map.get(ent_type, "#ffd700")
        html_tokens.append(
            f'<span style="background-color:{color}; padding:2px; border-radius:3px;" '
            f'title="{label} ({conf:.2f})">{tok}</span>'
        )

    deid_text = " ".join(deid_tokens)

    # Post-processing rules (honorifics & doctor signatures)
    if options.get("patient", True):
        deid_text = mask_honorific_patient_names(deid_text)
    if options.get("doctor", True):
        deid_text = mask_doctor_signatures_full(deid_text)

    highlighted_html = " ".join(html_tokens)
    return highlighted_html, deid_text


# =============================
# SIMPLE RAG RESOLVER (OPTIONAL)
# =============================

class SimpleRAGResolver:
    def __init__(self, knowledge_entries: List[str], threshold: float = 0.5):
        if not RAG_AVAILABLE:
            raise RuntimeError("RAG not available: sentence_transformers/faiss not installed.")
        self.threshold = threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.knowledge_entries = knowledge_entries
        self.embeddings = self.model.encode(knowledge_entries, convert_to_numpy=True)

        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def __call__(self, token_preds: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        adjusted = []
        for tok, label, conf in token_preds:
            if label in AMBIGUOUS_LABELS and conf < self.threshold:
                query_vec = self.model.encode([tok], convert_to_numpy=True)
                faiss.normalize_L2(query_vec)
                D, I = self.index.search(query_vec, k=1)
                sim = float(D[0][0])
                matched = self.knowledge_entries[I[0][0]]

                if "hospital" in matched.lower() or "medical center" in matched.lower():
                    new_label = "B-HOSPITAL"
                    new_conf = max(conf, sim)
                    adjusted.append((tok, new_label, new_conf))
                else:
                    adjusted.append((tok, label, conf))
            else:
                adjusted.append((tok, label, conf))
        return adjusted


# =============================
# PIPELINE ENTRY FUNCTIONS
# =============================

def process_clinical_text(model, tokenizer, text: str, use_rag: bool = False, rag_resolver=None, options=None):
    return mask_with_ner_and_regex(model, tokenizer, text, use_rag=use_rag, rag_resolver=rag_resolver, options=options)


def process_radiology_text(model, tokenizer, text: str, use_rag: bool = False, rag_resolver=None, options=None):
    # same logic now; later you can customize for radiology-specific PHI
    return mask_with_ner_and_regex(model, tokenizer, text, use_rag=use_rag, rag_resolver=rag_resolver, options=options)


def process_xray_image_placeholder(image_path: str) -> str:
    """
    Placeholder for future image-based extension:
    - Load DICOM/PNG
    - Run a vision-language model to produce a report
    - Then call process_radiology_text on the generated text
    """
    return "Image-to-text X-ray captioning is not implemented in this demo."


# =============================
# STREAMLIT APP
# =============================

def run_streamlit_app():
    st.set_page_config(page_title="Clinical De-ID Demo", layout="wide")
    st.title("🔒 Automated Clinical Text De-Identification (Demo)")

    @st.cache_resource
    def load_resources():
        model, tokenizer = load_trained_model(CHECKPOINT_PATH)
        if RAG_AVAILABLE:
            knowledge_entries = [
                "Mercy Hospital",
                "St. Mary Medical Center",
                "Johns Hopkins Hospital",
                "Cleveland Clinic",
                "Mayo Clinic",
                "Boston",
                "New York",
                "Los Angeles",
            ]
            rag_resolver = SimpleRAGResolver(knowledge_entries=knowledge_entries, threshold=0.7)
        else:
            rag_resolver = None
        return model, tokenizer, rag_resolver

    model, tokenizer, rag_resolver = load_resources()

    st.sidebar.header("Input Options")
    input_type = st.sidebar.selectbox("Input Type", ["Clinical Note", "Radiology Report (Text)", "X-Ray Image (Future)"])
    input_mode = st.sidebar.radio("Input Mode", ["Manual Text", "Upload Document (.txt)"])

    if RAG_AVAILABLE:
        use_rag = st.sidebar.checkbox("Use RAG for ambiguous entities", value=True)
    else:
        st.sidebar.checkbox("Use RAG for ambiguous entities", value=False, disabled=True)
        st.sidebar.info("RAG is disabled (sentence_transformers/faiss not installed).")
        use_rag = False

    st.sidebar.markdown("---")
    st.sidebar.header("PHI categories to de-identify")
    opt_patient = st.sidebar.checkbox("Patient names", True)
    opt_doctor = st.sidebar.checkbox("Doctor / provider names", True)
    opt_dates = st.sidebar.checkbox("Dates", True)
    opt_contact = st.sidebar.checkbox("Contact info (phone, email, IDs, ZIP)", True)
    opt_age = st.sidebar.checkbox("Age", True)
    opt_gender = st.sidebar.checkbox("Gender", True)
    opt_location = st.sidebar.checkbox("Locations / hospitals", True)

    phi_options = {
        "patient": opt_patient,
        "doctor": opt_doctor,
        "dates": opt_dates,
        "contact": opt_contact,
        "age": opt_age,
        "gender": opt_gender,
        "location": opt_location,
    }

    st.sidebar.markdown("---")
    st.sidebar.write("These options control both model-based and regex-only de-identification.")

    if input_type == "X-Ray Image (Future)":
        st.info("Image-based X-ray de-identification is a future extension.\n"
                "For this demo, please paste the radiology *report text* instead.")
        uploaded_image = st.file_uploader("Upload X-ray / DICOM (placeholder, not processed yet)", type=["png", "jpg", "jpeg", "dcm"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image (no de-identification yet).")
        st.stop()

    # ---------- Input handling ----------
    uploaded_text = None
    if input_mode == "Manual Text":
        default_text = (
            "Patient Mary Coleman visited Rockford Medical Center on 03/14/2021. "
            "Contact number is 555-123-4567 and email is mary.coleman@example.com. "
            "He is a 67-year-old male with hypertension."
        )
        text = st.text_area("Paste clinical or radiology text here:", height=200, value=default_text)
    else:
        uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])
        text = ""
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            uploaded_text = bytes_data.decode("utf-8", errors="ignore")
            text = uploaded_text
            with st.expander("Preview of uploaded document"):
                st.text_area("Document preview", value=uploaded_text[:5000], height=200)

    # ---------- Action ----------
    if st.button("De-identify Text / Document"):
        if not text or not text.strip():
            st.warning("Please provide some text (paste it or upload a .txt file).")
        else:
            if input_type == "Clinical Note":
                highlighted_html, deid_text = process_clinical_text(
                    model, tokenizer, text, use_rag=use_rag, rag_resolver=rag_resolver, options=phi_options
                )
            else:
                highlighted_html, deid_text = process_radiology_text(
                    model, tokenizer, text, use_rag=use_rag, rag_resolver=rag_resolver, options=phi_options
                )

            st.subheader("Highlighted PHI (Model + Rules Output)")
            if highlighted_html.strip():
                st.markdown(
                    f"<div style='font-family: monospace; font-size: 14px;'>{highlighted_html}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No tokens predicted by the model (empty or very short input?).")

            st.subheader("De-identified Output (Model + Options)")
            st.code(deid_text, language="text")

            st.subheader("Regex-only De-identification (Baseline, respects options)")
            st.code(regex_deidentify(text, options=phi_options), language="text")

            # Download button for de-identified document
            st.subheader("Download De-identified Document")
            st.download_button(
                label="Download de-identified .txt",
                data=deid_text,
                file_name="deidentified_document.txt",
                mime="text/plain",
            )

    st.markdown("---")
    st.caption(
        "Demo pipeline: tiny BioClinicalBERT NER + AGE/GENDER rules + optional RAG + configurable PHI categories. "
        "Designed for automated PHI masking in clinical and radiology text, with document upload support and future X-ray extensions."
    )


# =============================
# MAIN
# =============================

if __name__ == "__main__":
    # Auto-train ONCE on tiny dataset if no checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        train_and_save_model(CHECKPOINT_PATH)

    # Then run the Streamlit app
    run_streamlit_app()
