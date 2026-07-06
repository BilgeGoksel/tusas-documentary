# Yerel Belge Analiz ve Soru-Cevap Sistemi

Bu proje, kullanıcıların PDF, JPG ve PNG formatındaki belgeleri yükleyerek bu belgeler hakkında doğal dilde soru sorabilmesini sağlayan yerel çalışan bir Belge Analiz ve Soru-Cevap sistemidir.

Sistem; belge yükleme, PDF metin çıkarımı, görsellerden OCR ile metin okuma, metni parçalara ayırma, embedding üretme, ChromaDB üzerinde arama yapma ve Ollama üzerinden belgeye dayalı cevap üretme adımlarını içerir.

---

## Özellikler

- PDF, JPG, JPEG ve PNG belge yükleme
- PDF dosyalarından sayfa bazlı metin çıkarımı
- Görsellerden OCR ile metin çıkarımı
- Türkçe ve İngilizce belge desteği
- Yerel Ollama modeli ile cevap üretimi
- Yerel Ollama embedding modeli ile vektör üretimi
- ChromaDB ile belge içi anlamsal arama
- Kaynak gösterimli cevap üretimi
- Belgede olmayan bilgi için cevap uydurmama yaklaşımı
- Çoklu belge seçimi
- Kullanıcı dostu Streamlit arayüzü
- FastAPI tabanlı backend
- Test edilebilir ve modüler mimari

---

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF İşleme | PyMuPDF |
| OCR | PaddleOCR |
| Görüntü İşleme | Pillow / OpenCV |
| Embedding | Ollama `qwen3-embedding:0.6b` |
| LLM | Ollama `qwen3:4b` |
| Vector Store | ChromaDB |
| Test | Pytest |
| Konfigürasyon | `.env` |

---

## Sistem Mimarisi

```text
Kullanıcı
↓
Streamlit Arayüzü
↓
FastAPI Backend
↓
Belge Yükleme
↓
PDF Metin Çıkarımı / OCR
↓
Sayfa Bazlı Metin
↓
Chunking
↓
Embedding
↓
ChromaDB
↓
Retriever
↓
Prompt Builder
↓
Ollama / Qwen
↓
Kaynaklı Cevap
```

---

## Proje Yapısı

```text
CaseStudy/
│
├── app/
│   ├── api/
│   │   ├── routes_documents.py
│   │   ├── routes_health.py
│   │   ├── routes_qa.py
│   │   └── routes_search.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging_config.py
│   │
│   ├── document_processing/
│   │   ├── document_processor.py
│   │   ├── image_preprocessor.py
│   │   ├── ocr_service.py
│   │   ├── pdf_extractor.py
│   │   └── text_cleaner.py
│   │
│   ├── rag/
│   │   ├── answer_generator.py
│   │   ├── chunker.py
│   │   ├── embedding_service.py
│   │   ├── prompt_builder.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── services/
│   │   ├── document_processing_service.py
│   │   ├── file_service.py
│   │   ├── indexing_service.py
│   │   └── qa_service.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── tests/
│
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── indexed/
│   └── chroma/
│
├── logs/
│
├── .env.example
├── requirements.txt
├── README.md
├── DEVLOG.md
└── TESTING.md
```

---

## Kurulum

### 1. Repository'yi klonlayın

```bash
git clone <REPOSITORY_URL>
cd CaseStudy
```

---

### 2. Sanal ortam oluşturun

Windows için:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS için:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Bağımlılıkları kurun

```bash
pip install -r requirements.txt
```

---

## Ollama Kurulumu

Bu proje ücretli veya bulut tabanlı LLM API kullanmaz. Model ve embedding işlemleri yerel Ollama üzerinden çalışır.

Önce Ollama'nın kurulu ve çalışıyor olduğundan emin olun.

Ollama servis kontrolü:

```bash
ollama list
```

Gerekli modelleri indirin:

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

Ollama servisinin çalıştığını kontrol etmek için tarayıcıdan şu adrese gidilebilir:

```text
http://localhost:11434
```

Beklenen çıktı:

```text
Ollama is running
```

---

## Ortam Değişkenleri

`.env.example` dosyasını `.env` olarak kopyalayın.

Windows için:

```powershell
Copy-Item .env.example .env
```

Linux/macOS için:

```bash
cp .env.example .env
```


---

## Uygulamayı Çalıştırma

### 1. Backend'i başlatın

```bash
uvicorn app.main:app --reload --reload-dir app
```

Backend şu adreste çalışır:

```text
http://127.0.0.1:8000
```

Swagger dokümantasyonu:

```text
http://127.0.0.1:8000/docs
```

---

### 2. Frontend'i başlatın

Ayrı bir terminal açın ve sanal ortamı tekrar aktif edin.

```bash
streamlit run frontend/streamlit_app.py
```

Streamlit arayüzü genellikle şu adreste açılır:

```text
http://localhost:8501
```

---

## Kullanım

1. Uygulamayı açın.
2. PDF, JPG veya PNG belgelerinizi seçin.
3. **Belgeleri Yükle ve Hazırla** butonuna basın.
4. Sistem arka planda belgeyi:
   - yükler,
   - işler,
   - OCR veya PDF metin çıkarımı uygular,
   - chunk'lara ayırır,
   - embedding üretir,
   - ChromaDB'ye indexler.
5. Belge hazır olduktan sonra soru sorabilirsiniz.
6. Sistem belgeye dayalı cevap ve kaynakları gösterir.

---

## API Endpoint'leri

### Sağlık Kontrolü

```http
GET /api/v1/health
```

Backend ve Ollama bağlantı durumunu kontrol eder.

---

### Belge Yükleme

```http
POST /api/v1/documents/upload
```

PDF, JPG, JPEG veya PNG belge yükler.

---

### Belge İşleme

```http
POST /api/v1/documents/{document_id}/process
```

Yüklenen belgeyi işler. PDF ise metin çıkarımı yapar, görsel ise OCR uygular.

---

### Belge Indexleme

```http
POST /api/v1/documents/{document_id}/index
```

İşlenmiş belgeyi chunk'lara ayırır, embedding üretir ve ChromaDB'ye kaydeder.

---

### Search / Retrieval

```http
POST /api/v1/search
```

Kullanıcı sorgusuna en alakalı chunk'ları döndürür. Debug ve retrieval doğrulama için kullanılır.

---

### Soru-Cevap

```http
POST /api/v1/qa
```

Belgeye dayalı cevap üretir.

Örnek request:

```json
{
  "query": "Bu belgedeki ana başlıklar nelerdir?",
  "document_ids": ["document-id"],
  "top_k": 5
}
```

Örnek response:

```json
{
  "answer": "Belgede ana başlıklar ...",
  "found_in_documents": true,
  "sources": [
    {
      "source_number": 1,
      "original_filename": "ornek.pdf",
      "page_number": 2,
      "similarity_score": 0.42,
      "snippet": "..."
    }
  ],
  "retrieved_chunk_count": 5,
  "model": "qwen3:4b",
  "top_k": 5
}
```

---

## Testleri Çalıştırma

Tüm testleri çalıştırmak için:

```bash
pytest -v
```

Test kapsamı genel olarak şunları içerir:

- Belge yükleme
- Dosya doğrulama
- Duplicate belge kontrolü
- PDF metin çıkarımı
- OCR servisleri
- Chunking
- Embedding servisi
- ChromaDB işlemleri
- Retrieval
- Prompt builder
- Ollama cevap üretimi
- QA servisleri
- API endpoint'leri

---

## Test Edilen Senaryolar

Sistem aşağıdaki belge ve soru tipleriyle test edilmiştir:

- Uzun ders notu PDF'i
- CV PDF'i
- Grafik PNG dosyası
- Türkçe ve İngilizce içeren örnek soru ekran görüntüsü
- Belgede olmayan bilgi soruları
- Çoklu belge seçimi
- OCR tabanlı belgeler
- Kaynak gösterme davranışı

Detaylı test sonuçları için:

```text
TESTING.md
```

dosyasına bakabilirsiniz.

---

## Geliştirme Süreci

Geliştirme sürecinde alınan kararlar, karşılaşılan hatalar ve çözümler için:

```text
DEVLOG.md
```

dosyasına bakabilirsiniz.

Bu dosyada özellikle şu konular anlatılmıştır:

- Problemin nasıl parçalara ayrıldığı
- FastAPI + Streamlit mimari kararı
- Ollama kullanım kararı
- PaddleOCR sürüm problemleri
- Threshold ayarı
- Timeout problemi
- Kullanıcı arayüzünün sadeleştirilmesi
- Sistem sınırları ve öğrenilenler

---

## Güvenlik ve Gizlilik

- Belgeler yerel ortamda işlenir.
- Ücretli veya bulut tabanlı LLM API kullanılmaz.
- Kullanıcı belgeleri harici servislere gönderilmez.
- API anahtarına ihtiyaç yoktur.
- `.env`, yüklenen belgeler, ChromaDB verileri ve log dosyaları Git'e eklenmez.

---

## Git'e Eklenmemesi Gerekenler

Aşağıdaki dosya ve klasörler `.gitignore` içinde tutulmalıdır:

```text
.env
.venv/
data/uploads/
data/processed/
data/indexed/
data/chroma/
logs/
__pycache__/
.pytest_cache/
```

---

## Gelecek Geliştirmeler

İleride eklenebilecek özellikler:

- OCR çıktısını kullanıcıya gösterme
- OCR metnini manuel düzeltme
- Tablo çıkarımı
- Grafik yorumlama için özel modül
- Daha gelişmiş reranking
- Çok kullanıcılı belge yönetimi
- PostgreSQL tabanlı metadata yönetimi
- Asenkron belge işleme kuyruğu
- Docker Compose ile tek komutla çalıştırma
- Daha güçlü yerel modellerle karşılaştırma
- Otomatik benchmark ve değerlendirme metrikleri

---

## Proje Durumu

Proje şu anda işlevsel bir MVP seviyesindedir.

Çalışan ana akış:

```text
Belge Yükle
↓
Otomatik Hazırla
↓
Soru Sor
↓
Kaynaklı Cevap Al
```

---

## Geliştirici

Bilge Göksel

---

## Lisans

Bu proje teknik değerlendirme amacıyla geliştirilmiştir.
