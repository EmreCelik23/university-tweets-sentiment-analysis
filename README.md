````markdown
# 🌌 YTU CE COSMOS | Sentiment Analysis Project

## Contributors
- Emre Çelik
- Alihan Uludağ

Bu proje, Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği bitirme projesi kapsamında geliştirilmiştir. 
Derin Öğrenme (CNN, BiLSTM, Hybrid) ve BERTurk modelleri kullanılarak, Twitter verileri üzerinden Türk Üniversitelerine yönelik duygu analizi yapar.

---

## 📂 Proje Dosya Yapısı

Projenin hatasız çalışması için dosyaların aşağıdaki düzende olduğundan emin olun:

```text
YTU_CE_Cosmos/
├── app.py                # Ana Arayüz (Streamlit)
├── predict.py            # Yapay Zeka Motoru (PyTorch)
├── requirements.txt      # Gerekli Kütüphaneler
├── README.md             # Bu Dosya
├── .streamlit/           # [ÖNEMLİ] Tema Klasörü
│   └── config.toml       # Renk ayarları
├── models/               # EĞİTİLMİŞ MODELLER
│   ├── berturk_model/    # BERTurk dosyaları
│   ├── cnn_model.pt      # CNN Ağırlıkları
│   ├── bilstm_model.pt   # BiLSTM Ağırlıkları
│   └── tokenizer.pickle  # Kelime Sözlüğü (CNN/LSTM için)
└── data/                 # Veri Setleri
````

-----

## 🚀 Kurulum (Adım Adım)

Projeyi çalıştırmak için bilgisayarınızda **Python 3.8+** yüklü olmalıdır.

### 🍎 Mac / Linux Kullanıcıları İçin

Terminali proje klasöründe açın ve şu komutları sırasıyla uygulayın:

1.  **Sanal Ortamı Oluşturun:**

    ```bash
    python3 -m venv .venv
    ```

2.  **Ortamı Aktif Edin:**

    ```bash
    source .venv/bin/activate
    ```

    *(Terminal satırının başında `(.venv)` yazısını görmelisiniz)*

3.  **Kütüphaneleri Yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```

-----

### 🪟 Windows Kullanıcıları İçin

CMD veya PowerShell'i proje klasöründe açın ve şu komutları uygulayın:

1.  **Sanal Ortamı Oluşturun:**

    ```cmd
    python -m venv .venv
    ```

2.  **Ortamı Aktif Edin:**

    ```cmd
    .venv\Scripts\activate
    ```

3.  **Kütüphaneleri Yükleyin:**

    ```cmd
    pip install -r requirements.txt
    ```

-----

## 🎮 Uygulamayı Başlatma

Kurulum tamamlandıktan sonra (ve sanal ortam `.venv` aktifken) arayüzü başlatmak için:

```bash
streamlit run app.py
```

Tarayıcınız otomatik açılacaktır. Açılmazsa terminaldeki `http://localhost:8501` linkine tıklayın.

-----

## ⚠️ Olası Sorunlar ve Çözümleri

  * **"Module not found" Hatası:** Sanal ortamı aktif etmeyi unutmuşsunuzdur. `source .venv/bin/activate` (Mac) veya `.venv\Scripts\activate` (Windows) komutunu tekrar girin.
  * **Model Yükleme Hatası:** `models/` klasörünün içinde `.pt` dosyalarının ve `tokenizer.pickle` dosyasının eksik olmadığından emin olun.
  * **Renkler Gelmiyor:** `.streamlit/config.toml` dosyasının oluşturulduğundan emin olun.

<!-- end list -->

```
```
