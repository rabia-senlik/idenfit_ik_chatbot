"""
ADIM 2: CHROMADB ÜZERİNDE ANLAMSAL ARAMA (SEMANTIC SEARCH / RETRIEVAL TESTİ)
================================================================================
Bu script:
  1. ChromaDB'ye bağlanır.
  2. Kullanıcının sorduğu Türkçe soruyu vektöre çevirir.
  3. PDF parçaları içinden en alakalı ilk 2-3 sonucu ve benzerlik skorunu getirir.

Çalıştırma: python 02_arama_testi.py
"""
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import time

RAG_DIZINI = Path(__file__).resolve().parent
DB_DIZINI = RAG_DIZINI / "chroma_db"

print("=" * 70)
print("ADIM 2: İDENFİT VEKTÖR VERİTABANINDA ANLAMSAL ARAMA (RETRIEVAL)")
print("=" * 70)

# 1) Embedding Modelini ve ChromaDB'yi Yükle
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=str(DB_DIZINI))
collection = client.get_collection(name="idenfit_kurallar", embedding_function=embedding_func)

def kural_ara(soru, top_k=2):
    """
    Soruyu vektöre çevirip en alakalı top_k adet PDF parçasını bulur.
    """
    t0 = time.time()
    sonuclar = collection.query(
        query_texts=[soru],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    sure = (time.time() - t0) * 1000 # milisaniye
    
    print(f"\n" + "-" * 70)
    print(f"SORU: {soru}")
    print(f"Arama Süresi: {sure:.1f} ms | Bulunan İlgili Madde Sayısı: {top_k}")
    print("-" * 70)
    
    docs = sonuclar["documents"][0]
    metas = sonuclar["metadatas"][0]
    mesafeler = sonuclar["distances"][0]
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, mesafeler), 1):
        benzerlik = (1 - dist) * 100 # Mesafe ne kadar küçükse benzerlik o kadar yüksek
        print(f"\n  [BULUNAN MADDE {i}] -> Benzerlik: %{benzerlik:.1f}")
        print(f"  Kaynak Dosya: {meta['kaynak']} (Sayfa: {meta['sayfa']})")
        print(f"  İçerik:")
        print(f"  \"{doc}\"")

# 3 Kritik İK Sorusunu Test Ediyoruz:
kural_ara("Babalık izni kaç gündür ve ne zaman talep edilmelidir?")
kural_ara("MoneyPay uygulamasında GSM ile üye olma adımları nelerdir?")
kural_ara("Makinada çalışırken saçlar ve kişisel koruyucu donanım nasıl olmalıdır?")

print("\n" + "=" * 70)
print("Arama Testi Tamamlandı!")
print("=" * 70)
