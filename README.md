# 🏢 İdenfit İK ve Şirket Kuralları RAG Asistanı (%100 Yerel & Çevrimdışı)

Bu proje, İdenfit şirket içi prosedürlerini (izin kuralları, prim hak edişleri, MoneyPay kullanımı, İSG talimatları ve ahlaki ilkeler) **tamamen yerel ve çevrimdışı** olarak sorgulayan akıllı bir **RAG (Retrieval-Augmented Generation)** asistanıdır.

Herhangi bir harici API (Gemini, OpenAI vb.) veya internet bağlantısına ihtiyaç duymaz.

---

## 🚀 Özellikler

- **%100 Yerel LLM:** Ollama üzerinde çalışan **Qwen 2.5 (3B)** modeli ile kurumsal ve profesyonel Türkçe yanıtlar.
- **Akıllı Hibrit Arama:** **ChromaDB** ONNX vektör benzerliği ile Türkçe morfolojik anahtar kelime eşleşmesini birleştirerek kural maddelerini hatasız bulur.
- **Canlı Akış (Streaming):** Yanıtlar kullanıcıya daktilo efektiyle anlık olarak sunulur.
- **Docker Desteği:** Tek komutla veya tek tıkla konteyner olarak ayağa kaldırılabilir.
- **Kaynak Gösterimi:** Her yanıtın altında referans alınan PDF dokümanı, sayfa numarası ve alaka skoru gösterilir.

---

## 📦 Kurulum ve Çalıştırma

### 1. Ön Gereksinim: Ollama Kurulumu
1. [Ollama](https://ollama.com/) uygulamasını indirin ve kurun.
2. Komut satırından Qwen 2.5 modelini indirin:
   ```bash
   ollama run qwen2.5:3b
   ```

---

### 2. Çalıştırma Yöntemleri

#### Yöntem A: Doğrudan Windows Üzerinde (Hızlı)
Gereken paketleri kurun:
```bash
pip install -r requirements.txt
```
Ardından:
- Doğrudan [**`baslat.bat`**](baslat.bat) dosyasına çift tıklayın **veya**
- Terminalden çalıştırın:
  ```bash
  streamlit run app.py
  ```

#### Yöntem B: Docker ile Çalıştırma
Docker Desktop'ı açtıktan sonra:
- [**`docker_baslat.bat`**](docker_baslat.bat) dosyasına çift tıklayın **veya**
- Terminalden:
  ```bash
  docker compose up --build -d
  ```

Tarayıcınızdan **`http://localhost:8501`** adresine giderek asistanı kullanabilirsiniz.

---

## 📂 Proje Yapısı

```
├── app.py                     # Streamlit kullanıcı arayüzü ve RAG motoru
├── requirements.txt           # Python bağımlılıkları
├── Dockerfile                 # Docker konteyner tanımı
├── docker-compose.yml         # Docker Compose servis yapılandırması
├── baslat.bat                 # Windows yerel tek tık başlatıcı
├── docker_baslat.bat          # Docker tek tık başlatıcı
├── chroma_db/                 # Önceden indekslenmiş ChromaDB vektör veritabanı
├── vector_index.pkl           # Yapılandırılmış doküman kural parçaları
└── docs/                      # Doküman klasörü (PDF'ler yerel tutulur, gitignore edilmiştir)
```

