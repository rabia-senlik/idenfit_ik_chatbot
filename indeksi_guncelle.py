"""
İDENFİT VE ŞİRKET PROSEDÜRLERİ İNDEKS OLUŞTURUCU
=================================================
Tüm PDF'leri (metin + taranmış anket/seçim görselleri) okur,
maddeler ve bölümler halinde ayırır,
Google Gemini Embedding (3072-D) ile vektörleştirip 'vector_index.pkl'ye kaydeder.
"""
import pypdf
import re
import os
import pickle
import time
from pathlib import Path
from google import genai
import numpy as np

KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=KEY) if KEY else None
RAG_DIR = Path(__file__).resolve().parent

def indeks_olustur():
    docs = []
    metas = []
    pdf_files = sorted(RAG_DIR.glob("*.pdf"))
    print(f"Toplam {len(pdf_files)} PDF dosyası taranıyor...")

    for pdf_path in pdf_files:
        fname = pdf_path.name
        print(f"\n📂 Dosya: {fname}")
        reader = pypdf.PdfReader(str(pdf_path))
        
        # 1. DEVAM PRİMİ VE İZİNLER TABLOSU
        if "104656" in fname:
            full_text = "\n".join((p.extract_text() or "") for p in reader.pages)
            items = re.split(r'\n(?=\d{1,2}\s+[A-ZÇĞİÖŞÜ])', full_text)
            for item in items:
                clean = item.strip()
                if len(clean) > 20:
                    docs.append(f"DEVAM PRİMİ VE İZİN KURALI:\n{clean}")
                    metas.append({"kaynak": fname, "sayfa": 1, "baslik": clean.split("\n")[0][:60]})
            print(f"   -> {len(items)} izin kuralı ayrıştırıldı.")
            continue

        # 2. AHLAKİ İLKELER PROSEDÜRÜ
        if "135538" in fname:
            full_text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
            # Bölümlere göre ayır (1. AMAÇ, 2. KAPSAM, 3. TANIMLAR, 4. UYGULAMA vb.)
            sections = re.split(r'\n(?=[0-9]\.\s+[A-ZÇĞİÖŞÜ\s]{3,})', full_text)
            for sec in sections:
                clean = sec.strip()
                if len(clean) > 30:
                    docs.append(f"AHLAKİ İLKELER PROSEDÜRÜ:\n{clean}")
                    metas.append({"kaynak": fname, "sayfa": 1, "baslik": clean.split("\n")[0][:60]})
            print(f"   -> {len(sections)} ahlaki ilke bölümü ayrıştırıldı.")
            continue

        # 3. DİĞER PDF'LER (MoneyPay, İSG, Seçim Anketleri)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            
            # Sayfa görsel ise Gemini Vision ile metne dök
            if len(text) < 50 and len(page.images) > 0:
                print(f"   Sayfa {i+1} taranmış görsel/anket, Gemini Multimodal ile okunuyor...")
                img_data = page.images[0].data
                try:
                    ocr_res = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=[
                            genai.types.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
                            "Bu şirket dokümanı görselindeki tüm metinleri, başlıkları, anket/seçim sonuçlarını ve oranları eksiksiz Türkçe metin olarak çıkar."
                        ]
                    )
                    text = ocr_res.text.strip()
                except Exception as e:
                    print(f"   OCR Hatası: {e}")

            if len(text) > 30:
                # Paragraflara veya adımlara göre böl
                paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 35]
                if not paras:
                    paras = [text]
                for p in paras:
                    docs.append(p)
                    metas.append({"kaynak": fname, "sayfa": i+1, "baslik": p.split("\n")[0][:60]})

    print(f"\n✅ Toplam {len(docs)} bağımsız vektör parçası hazırlandı.")
    print("Gemini Embedding (3072-D) vektörleri hesaplanıyor...")
    
    t0 = time.time()
    # Batch embedding (tek seferde veya 50'şerli)
    all_embeddings = []
    chunk_size = 40
    for i in range(0, len(docs), chunk_size):
        sub_docs = docs[i:i+chunk_size]
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=sub_docs
        )
        for e in res.embeddings:
            all_embeddings.append(e.values)
            
    embeddings = np.array(all_embeddings)
    output_path = RAG_DIR / "vector_index.pkl"
    with open(output_path, "wb") as f:
        pickle.dump({"docs": docs, "metas": metas, "embeddings": embeddings}, f)
        
    print(f"🎉 Başarıyla tamamlandı! Vektör veritabanı güncellendi ({time.time()-t0:.2f} saniye).")
    return len(docs)

if __name__ == "__main__":
    indeks_olustur()
