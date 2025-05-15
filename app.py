import streamlit as st
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download resource NLTK jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load model dan tokenizer
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")

# Stopwords Bahasa Indonesia
stopwords = set(StopWordRemoverFactory().get_stop_words())

# Kata negasi sederhana
negation_words = ['tidak', 'bukan', 'kurang', 'belum', 'jangan']

def cleaning(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text):
    return text.lower()

def handle_negation(text):
    tokens = word_tokenize(text)
    result = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if tokens[i] in negation_words and i + 1 < len(tokens):
            result.append(tokens[i] + '_' + tokens[i + 1])
            skip = True
        else:
            result.append(tokens[i])
    return ' '.join(result)

def stopword_removal(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stopwords]
    return ' '.join(filtered)

st.title("Deteksi Ulasan Palsu Bahasa Indonesia")

st.markdown("Masukkan ulasan yang ingin dicek dan pilih langkah preprocessing:")

text_input = st.text_area("Masukkan Ulasan")

with st.expander("Pilih Langkah Preprocessing"):
    do_cleaning = st.checkbox("Cleaning", value=True)
    do_case_folding = st.checkbox("Case Folding", value=True)
    do_negation = st.checkbox("Handling Negation", value=True)
    do_stopwords = st.checkbox("Stopword Removal", value=True)

if st.button("Deteksi Ulasan"):
    if not text_input.strip():
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        processed_text = text_input
        steps = []

        if do_cleaning:
            processed_text = cleaning(processed_text)
            steps.append("Pembersihan teks dari URL, angka, tanda baca, dan spasi berlebih.")

        if do_case_folding:
            processed_text = case_folding(processed_text)
            steps.append("Teks diubah menjadi huruf kecil semua.")

        if do_negation:
            processed_text = handle_negation(processed_text)
            steps.append("Negasi digabung menjadi satu token, misal 'tidak_bagus'.")

        if do_stopwords:
            processed_text = stopword_removal(processed_text)
            steps.append("Penghapusan kata-kata umum yang tidak penting (stopword).")

        # Tokenisasi & prediksi
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()

        label = "ULASAN PALSU" if pred_label == 1 else "ULASAN ASLI"
        color = "red" if pred_label == 1 else "green"

        st.markdown(f"### Hasil Deteksi: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        with st.expander("Penjelasan Preprocessing"):
            for step in steps:
                st.markdown(f"- {step}")

        st.text("Teks Setelah Preprocessing:")
        st.code(processed_text)
