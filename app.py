import streamlit as st
import re
import torch
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Pastikan resource nltk 'punkt' sudah terunduh, lakukan sekali saja
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load model dan tokenizer (pastikan folder "saved_model" berisi model yang benar)
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")

# Stopwords dan kata negasi
stopwords = set(StopWordRemoverFactory().get_stop_words())
negation_words = ['tidak', 'bukan', 'kurang', 'belum', 'jangan']

# Fungsi preprocessing

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
    filtered = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered)

# UI Streamlit

st.title("Deteksi Ulasan Palsu (Bahasa Indonesia)")

st.markdown("Masukkan ulasan dan pilih langkah-langkah preprocessing yang ingin digunakan.")

text_input = st.text_area("Masukkan Ulasan")

with st.expander("Pilih Langkah Preprocessing"):
    do_cleaning = st.checkbox("Cleaning", value=True)
    do_case_folding = st.checkbox("Case Folding", value=True)
    do_negation = st.checkbox("Handling Negation", value=True)
    do_stopwords = st.checkbox("Stopword Removal", value=True)

if st.button("Deteksi Ulasan"):
    if not text_input.strip():
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        original_text = text_input
        explanation = []

        if do_cleaning:
            text_input = cleaning(text_input)
            explanation.append("Dilakukan pembersihan teks dari URL, angka, tanda baca, dan spasi berlebih.")

        if do_case_folding:
            text_input = case_folding(text_input)
            explanation.append("Teks diubah menjadi huruf kecil seluruhnya.")

        if do_negation:
            text_input = handle_negation(text_input)
            explanation.append("Negasi seperti 'tidak baik' digabung menjadi satu token: 'tidak_baik'.")

        if do_stopwords:
            text_input = stopword_removal(text_input)
            explanation.append("Kata-kata umum yang tidak penting dihapus dari teks.")

        # Tokenisasi dan prediksi
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "ULASAN PALSU" if prediction == 1 else "ULASAN ASLI"
        color = "red" if prediction == 1 else "green"

        st.markdown(f"### Hasil Deteksi: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

        with st.expander("Penjelasan Preprocessing"):
            for step in explanation:
                st.markdown(f"- {step}")

        st.text("Teks Setelah Preprocessing:")
        st.code(text_input, language='text')
