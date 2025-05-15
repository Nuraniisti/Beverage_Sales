import streamlit as st
import pandas as pd

def main():
    st.title("Deteksi Ulasan Palsu Sederhana")
    menu = st.sidebar.selectbox("Menu", ["Beranda", "Deteksi Ulasan", "Hasil Deteksi", "Statistik", "Tentang"])
    
    if menu == "Beranda":
        st.header("Selamat datang di aplikasi Deteksi Ulasan Palsu")
        st.write("""
            Aplikasi ini membantu Anda mendeteksi apakah sebuah ulasan merupakan ulasan asli atau palsu.
            Gunakan menu 'Deteksi Ulasan' untuk memasukkan ulasan dan mendapatkan hasil prediksi.
            Anda juga bisa melihat statistik ulasan yang sudah dianalisis pada menu 'Statistik'.
        """)
        
    elif menu == "Deteksi Ulasan":
        st.header("Masukkan Ulasan untuk Deteksi")
        ulasan = st.text_area("Ketik ulasan di sini:", height=150)
        
        if st.button("Deteksi"):
            if ulasan.strip() == "":
                st.warning("Silakan masukkan ulasan terlebih dahulu!")
            else:
                # Contoh prediksi dummy, ganti dengan model nyata
                hasil_prediksi = "Palsu" if "spam" in ulasan.lower() else "Asli"
                st.success(f"Hasil deteksi: {hasil_prediksi}")
    
    elif menu == "Hasil Deteksi":
        st.header("Daftar Ulasan dan Hasil Deteksi")
        st.info("Fitur ini akan menampilkan daftar ulasan yang sudah dideteksi jika ada data tersimpan.")
        # Contoh kosong, bisa dikembangkan dengan penyimpanan hasil
        
    elif menu == "Statistik":
        st.header("Statistik Deteksi Ulasan")
        st.write("Visualisasi dan statistik terkait ulasan yang sudah dianalisis akan ditampilkan di sini.")
        # Contoh dummy
        data = {'Kategori': ['Asli', 'Palsu'], 'Jumlah': [70, 30]}
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index('Kategori'))
        
    elif menu == "Tentang":
        st.header("Tentang Aplikasi")
        st.write("""
            Aplikasi ini dibuat dengan Streamlit untuk demonstrasi deteksi ulasan palsu.
            Model deteksi mengandalkan teknik sederhana (contoh dummy), 
            namun bisa dikembangkan dengan algoritma NLP dan machine learning.
        """)

if __name__ == "__main__":
    main()
