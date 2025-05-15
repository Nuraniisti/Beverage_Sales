import streamlit as st
import re
import nltk
import torch
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Download NLTK resource kalau belum ada
nltk.download('punkt')

# Load model dan tokenizer yang sudah kamu latih & simpan di folder 'saved_model'
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")

# Stopwords dan daftar kata negasi
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
        if tokens[i] in negation_words and i+1 < len(tokens):
            result.append(tokens[i] + '_' + tokens[i+1])
            skip = True
        else:
            result.append(tokens[i])
    return ' '.join(result)

def stopword_removal(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="Deteksi Ulasan Palsu", layout="centered")
st.title("Deteksi Ulasan Palsu (Bahasa Indonesia)")

st.markdown("Masukkan ulasan dan pilih langkah preprocessing yang ingin digunakan.")

text_input = st.text_area("Masukkan Ulasan", height=150)

with st.expander("Pilih Langkah Preprocessing"):
    do_cleaning = st.checkbox("Cleaning (hapus URL, angka, tanda baca)", value=True)
    do_case_folding = st.checkbox("Case Folding (huruf kecil semua)", value=True)
    do_negation = st.checkbox("Handling Negation (gabung kata negasi)", value=True)
    do_stopwords = st.checkbox("Stopword Removal (hapus kata umum)", value=True)

if st.button("Deteksi Ulasan"):
    if not text_input.strip():
        st.warning("Tolong masukkan ulasan terlebih dahulu!")
    else:
        explanation = []
        processed_text = text_input

        if do_cleaning:
            processed_text = cleaning(processed_text)
            explanation.append("Teks dibersihkan dari URL, angka, tanda baca, dan spasi berlebih.")
        if do_case_folding:
            processed_text = case_folding(processed_text)
            explanation.append("Teks diubah menjadi huruf kecil semua.")
        if do_negation:
            processed_text = handle_negation(processed_text)
            explanation.append("Kata negasi digabung dengan kata setelahnya (misal: 'tidak_bagus').")
        if do_stopwords:
            processed_text = stopword_removal(processed_text)
            explanation.append("Kata-kata umum yang kurang penting dihapus.")

        # Tokenisasi dan prediksi
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        label = "ULASAN PALSU" if pred == 1 else "ULASAN ASLI"
        color = "red" if pred == 1 else "green"

        st.markdown(f"### Hasil Deteksi: <span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)

        with st.expander("Penjelasan Langkah Preprocessing"):
            for step in explanation:
                st.markdown(f"- {step}")

        st.text("Teks Setelah Preprocessing:")
        st.code(processed_text, language='text')

        st.markdown("---")
        st.markdown(
            """
            **Tentang Model:**  
            Model ini menggunakan DistilBERT, sebuah model transformer yang telah dilatih pada data bahasa Indonesia untuk mendeteksi apakah ulasan termasuk palsu atau asli berdasarkan pola bahasa dan konteks.  
            Deteksi ini membantu mengidentifikasi opini yang mungkin dibuat secara tidak jujur atau manipulatif.
            """
        )
