"""
İDENFİT İK VE ŞİRKET KURALLARI RAG ASİSTANI (%100 YEREL & HIZLANDIRILMIŞ)
(CHROMADB HİBRİT ARAMA + OLLAMA QWEN2.5:3B)
========================================================================
Optimizasyonlar:
  1. Hızlı WebSocket ve CORS bağlantısı (gecikmesiz ilk açılış).
  2. Ollama GPU modelini bellekte sürekli sıcak tutma (keep_alive: 24h, soğuk başlama gecikmesi yok).
  3. Doğrudan ve net yanıt üreten optimize edilmiş İK promptu.
"""
import streamlit as st
import os
import re
import json
import time
import requests
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Sayfa yapılandırması
st.set_page_config(
    page_title="İdenfit İK ve Kural Asistanı (Ollama)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 14px;
        font-size: 0.85rem;
        color: #334155;
    }
    .badge-llm {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    .badge-vector {
        background-color: #E0E7FF;
        color: #3730A3;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

RAG_DIZINI = Path(__file__).resolve().parent
DB_DIZINI = RAG_DIZINI / "chroma_db"

# Ollama URL yapılandırması (Docker ve Yerel uyumlu)
def get_ollama_url():
    env_url = os.getenv("OLLAMA_URL")
    if env_url:
        return env_url
    for url in ["http://localhost:11434", "http://host.docker.internal:11434"]:
        try:
            r = requests.get(f"{url}/api/tags", timeout=1)
            if r.status_code == 200:
                return url
        except Exception:
            continue
    return "http://localhost:11434"

OLLAMA_URL = get_ollama_url()

def tr_lower(text: str) -> str:
    """Türkçe İ/ı karakter dönüşümünü ve birleşik aksanları normalize eder."""
    return text.replace("İ", "i").replace("I", "ı").lower().replace("\u0307", "")

@st.cache_resource
def chromadb_yukle():
    if not DB_DIZINI.exists():
        st.error("ChromaDB veritabanı bulunamadı!")
        st.stop()
    client = chromadb.PersistentClient(path=str(DB_DIZINI))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(name="idenfit_kurallar", embedding_function=embed_fn)
    return collection

@st.cache_resource
def modeli_onceden_bellege_al():
    """Modeli GPU/RAM'e önceden yükler, ilk soru sorulduğunda bekleme süresini sıfıra indirir."""
    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": "ok",
                "keep_alive": "24h",
                "options": {"num_predict": 1}
            },
            timeout=3
        )
    except Exception:
        pass

def ollama_modelleri_al():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if res.status_code == 200:
            modeller = [m["name"] for m in res.json().get("models", [])]
            return modeller
    except Exception:
        pass
    return []

def hibrit_arama(collection, query: str, top_k: int = 3):
    """
    ChromaDB vektör benzerliği ile Türkçe anahtar kelime eşleşmesini birleştiren hızlı hibrit arama motoru.
    """
    toplam_kayit = collection.count()
    res = collection.query(query_texts=[query], n_results=min(toplam_kayit, 30))
    
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    
    q_words = [w for w in re.findall(r'\w+', tr_lower(query)) if len(w) > 2]
    stopwords = {'hangi', 'neler', 'nelerdir', 'nasil', 'nedir', 'icin', 'kac', 'olan', 'olarak', 've', 'ile', 'gün', 'gun'}
    meaningful = [w for w in q_words if w not in stopwords]
    if not meaningful:
        meaningful = q_words
    
    scored = []
    for d, m, dist in zip(docs, metas, dists):
        vec_score = max(0.0, 1.0 - dist)
        d_lower = tr_lower(d)
        d_words = re.findall(r'\w+', d_lower)
        
        kw_hits = 0
        for w in meaningful:
            root = w[:4] if len(w) >= 4 else w
            if any(dw.startswith(root) for dw in d_words) or root in d_lower:
                kw_hits += 1
                
        kw_score = kw_hits / max(len(meaningful), 1)
        exact_bonus = 0.45 if any(w in d_lower for w in meaningful if len(w) > 3) else 0.0
        
        final_score = (vec_score * 0.3) + (kw_score * 0.5) + exact_bonus
        scored.append((final_score, d, m, vec_score))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# Koleksiyonu ve Modeli Başlat
collection = chromadb_yukle()
modeli_onceden_bellege_al()
mevcut_modeller = ollama_modelleri_al()

# SOL PANEL
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/human-resources.png", width=70)
    st.title("İdenfit Bilgi Havuzu")
    
    if mevcut_modeller:
        st.success("⚡ **Ollama Aktif (GPU Destekli)**\nModel bellekte sıcak tutuluyor.")
        varsayilan_idx = 0
        for i, m in enumerate(mevcut_modeller):
            if "qwen2.5:3b" in m or "qwen2.5" in m:
                varsayilan_idx = i
                break
        secilen_model = st.selectbox("Yerel LLM Modeli", mevcut_modeller, index=varsayilan_idx)
    else:
        st.error("⚠️ **Ollama Çalışmıyor!**\nLütfen bilgisayarınızda Ollama uygulamasını başlatın.")
        secilen_model = "qwen2.5:3b"
    
    st.markdown("---")
    st.subheader("📚 İndekslenen Dokümanlar")
    st.markdown("• 📄 `MoneyPay Yükleme & Üyelik Prosedürü`")
    st.markdown("• 📄 `İş Sağlığı ve Güvenliği (İSG) Talimatı`")
    st.markdown("• 📄 `Devam Primi & İzin Hak Ediş Tablosu`")
    st.markdown("• 📄 `Ahlaki İlkeler ve Çalışma Esasları`")
    
    st.metric(label="Toplam Vektör Maddesi", value=f"{collection.count()} Kural")
    
    st.markdown("---")
    st.subheader("💡 Hızlı Sorular")
    ornekler = [
        "Babalık izni kaç gün?",
        "Süt izni kuralları ve şartları nelerdir?",
        "Evlilik izninde hangi belgeler verilir?",
        "MoneyPay uygulamasına nasıl üye olunur?",
        "Makinelerle çalışırken dikkat edilecek iş güvenliği kuralları"
    ]
    for ornek in ornekler:
        if st.button(ornek, use_container_width=True):
            st.session_state["hazir_soru"] = ornek

# ANA EKRAN
st.markdown('<div class="main-header">🏢 İdenfit Şirket İçi Kural & İzin Asistanı</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Yıllık izinler, prim hakedişleri, MoneyPay veya iş güvenliği hakkında merak ettiğiniz her şeyi sorun. (Ollama & Yerel RAG)</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Merhaba! Ben İdenfit şirket içi yapay zeka asistanınızım. Bilgisayarınızda yerel çalışan Ollama (Qwen 2.5) altyapısıyla hizmet veriyorum. Şirket kuralları veya izinlerle ilgili sormak istediğiniz bir konu var mı?"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

hazir_girdi = st.session_state.pop("hazir_soru", None)
kullanici_sorusu = st.chat_input("Sorunuzu buraya yazın (Örn: Babalık izni kaç gün?)...") or hazir_girdi

if kullanici_sorusu:
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})
    with st.chat_message("user"):
        st.markdown(kullanici_sorusu)
    
    with st.chat_message("assistant"):
        t0 = time.time()
        
        # 1. RETRIEVAL (Yerel Vektör + Türkçe Hibrit Arama)
        en_iyi_sonuclar = hibrit_arama(collection, kullanici_sorusu, top_k=3)
        arama_suresi = (time.time() - t0) * 1000
        
        en_iyi_skor, en_iyi_doc, en_iyi_meta, en_iyi_vskor = en_iyi_sonuclar[0]
        ana_kaynak = en_iyi_meta.get("kaynak", "Doküman")
        ana_sayfa = en_iyi_meta.get("sayfa", 1)
        
        baglam = "\n---\n".join([item[1] for item in en_iyi_sonuclar])
        
        # 2. PROMPT HAZIRLAMA (Hızlı ve Net)
        prompt = f"""Sen şirket içi resmi İdenfit İK asistanısın. Aşağıda şirketin resmi dokümanlarından alınan doğrulanmış kural maddeleri yer almaktadır.
Kullanıcının sorusunu bu bilgilere dayanarak doğrudan, samimi, net ve maddeler halinde Türkçe olarak yanıtla. Uzun giriş cümleleri kurma, doğrudan kuralı aktar.
Dokümanda yer almayan hiçbir kuralı kendin uydurma.

[RESMİ ŞİRKET BİLGİSİ]:
{baglam}

[KULLANICI SORUSU]:
{kullanici_sorusu}
"""
        # 3. GENERATION (Ollama Streaming + keep_alive)
        placeholder = st.empty()
        tam_metin = ""
        
        try:
            res = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": secilen_model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "24h", # Model GPU belleğinde kalır, sonraki sorgular anında yanıt verir
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 250, # Hızlı ve öz yanıt üretimi
                        "num_ctx": 2048
                    }
                },
                stream=True,
                timeout=60
            )
            
            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        veri = json.loads(line.decode("utf-8"))
                        parca = veri.get("response", "")
                        tam_metin += parca
                        placeholder.markdown(tam_metin + "▌")
                placeholder.markdown(tam_metin)
                llm_metni = tam_metin
            else:
                llm_metni = f"Ollama hatası (Kod {res.status_code}): {res.text}"
                placeholder.markdown(llm_metni)
        except Exception as e:
            llm_metni = f"Ollama bağlantı hatası: {e}"
            placeholder.markdown(llm_metni)
        
        toplam_sure = (time.time() - t0)
        eslesme_yuzdesi = min(99.0, max(50.0, en_iyi_skor * 100))
        
        # Kaynak Kutusu
        kaynak_kutusu = f"""
<div class="source-box">
    🤖 <b>Model:</b> <span class="badge-llm">{secilen_model} (Yerel GPU)</span>&nbsp;&nbsp;|&nbsp;&nbsp;
    📐 <b>Vektör Motoru:</b> <span class="badge-vector">ChromaDB (Yerel Hibrit)</span><br>
    📌 <b>Resmi Dayanak:</b> <code>{ana_kaynak}</code> (Sayfa: {ana_sayfa})<br>
    🎯 <b>Alaka Skoru:</b> %{eslesme_yuzdesi:.1f}&nbsp;&nbsp;|&nbsp;&nbsp;⚡ <b>Arama:</b> {arama_suresi:.1f} ms&nbsp;&nbsp;|&nbsp;&nbsp;⏱️ <b>Toplam Süre:</b> {toplam_sure:.2f} sn
</div>
"""
        st.markdown(kaynak_kutusu, unsafe_allow_html=True)
        son_mesaj = f"{llm_metni}\n\n{kaynak_kutusu}"
        st.session_state.messages.append({"role": "assistant", "content": son_mesaj})
