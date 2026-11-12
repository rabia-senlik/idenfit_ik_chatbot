"""
ADIM 1 (ONNX GÜVENLİ SÜRÜM): İDENFİT DOKÜMANLARINI ONNX İLE CHROMADB'YE YAZMA
================================================================================
PyTorch DLL hatalarını önlemek için doğrudan ChromaDB'nin hafif ONNX motorunu kullanır.
"""
import os
import glob
import pypdf
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import time

t0 = time.time()
RAG_DIZINI = Path(__file__).resolve().parent
DB_DIZINI = RAG_DIZINI / "chroma_db"

print("=" * 70)
print("ONNX TABANLI GÜVENLİ VEKTÖR VERİTABANI OLUŞTURULUYOR")
print("=" * 70)

# ONNX tabanlı yerleşik embedding fonksiyonu (PyTorch DLL bağımlılığı YOKTUR!)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path=str(DB_DIZINI))
koleksiyon_adi = "idenfit_kurallar"

try:
    client.delete_collection(name=koleksiyon_adi)
except Exception:
    pass

collection = client.create_collection(
    name=koleksiyon_adi,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"}
)

pdf_dosyalari = glob.glob(str(RAG_DIZINI / "*.pdf"))

def metni_parcala(metin, parca_boyutu=450, ortusme=80):
    parcalar = []
    baslangic = 0
    metin_uzunluk = len(metin)
    while baslangic < metin_uzunluk:
        bitis = baslangic + parca_boyutu
        parca = metin[baslangic:bitis].strip()
        if len(parca) > 30:
            parcalar.append(parca)
        baslangic += (parca_boyutu - ortusme)
    return parcalar

documents = []
metadatas = []
ids = []
chunk_sayaci = 0

for pdf_yolu in pdf_dosyalari:
    dosya_adi = os.path.basename(pdf_yolu)
    reader = pypdf.PdfReader(pdf_yolu)
    for sayfa_no, sayfa in enumerate(reader.pages, 1):
        ham_metin = sayfa.extract_text() or ""
        temiz_metin = " ".join(ham_metin.split())
        if not temiz_metin:
            continue
        parcalar = metni_parcala(temiz_metin)
        for i, parca in enumerate(parcalar):
            chunk_sayaci += 1
            documents.append(parca)
            metadatas.append({
                "kaynak": dosya_adi,
                "sayfa": sayfa_no,
                "parca_no": i + 1
            })
            ids.append(f"doc_{chunk_sayaci}")

print(f">>> {len(documents)} adet metin parçası ONNX ile indeksleniyor...")
collection.add(documents=documents, metadatas=metadatas, ids=ids)

print(f"[BAŞARILI] ONNX Vektör Veritabanı Hazır! (Süre: {time.time() - t0:.2f} sn)")
