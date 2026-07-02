# AGENTS.md

Bu repository, TUSAS teknik degerlendirmesi icin gelistirilen yerel calisan bir Belge Analiz ve Soru-Cevap sistemidir.

## Proje Baglami

- Kullanicilar PDF, JPG ve PNG dosyalari yukleyebilecektir.
- PDF metin cikarimi ve Turkce/Ingilizce OCR planlanmaktadir.
- Sistem ileride RAG mimarisi kullanacaktir.
- Cevaplar yalnizca yuklenen belgelerdeki bilgilere dayanmalidir.
- Belgede olmayan bilgi soruldugunda sistem acikca bilgi bulunamadigini soylemelidir.
- Her cevap dosya adi ve sayfa numarasi ile kaynak gostermelidir.
- Ucretli veya bulut LLM API kullanilmayacaktir.
- LLM: yerel Ollama `qwen3:4b`.
- Embedding modeli: yerel Ollama `qwen3-embedding:0.6b`.
- Backend: FastAPI.
- Frontend: Streamlit.
- Vector store: ChromaDB.
- PDF isleme: PyMuPDF.
- OCR: PaddleOCR.
- Test: pytest.
- Proje daha sonra Docker ile calistirilacaktir.

## Calisma Kurallari

1. Python 3.11 kullan ve type hint ekle.
2. Kod moduler, okunabilir ve test edilebilir olsun.
3. Fonksiyonlar tek sorumluluk ilkesine uysun.
4. Public fonksiyonlara kisa docstring ekle.
5. Gizli anahtarlar, kullanici belgeleri ve hassas dosyalar Git'e eklenmemelidir.
6. Buyuk framework veya gereksiz bagimlilik ekleme.
7. Yeni bagimlilik eklemeden once gerekcesini acikla.
8. Mevcut calisan kodu gereksiz yere yeniden yazma.
9. Her gorev sonunda degistirilen dosyalari ve yapilanlari ozetle.
10. Kod degisikliklerinden sonra ilgili testleri calistir.
11. Kullanicinin belgesinin tam icerigini loglama.
12. Hatalari sessizce yutma; anlasilir hata mesajlari ve uygun logging kullan.
13. Kod yazmadan once mevcut repository dosyalarini incele.
14. Yapilan teknik kararlarin `DEVLOG.md` icine kaydedilmeye uygun ozetini raporla.

## Ilk Gun Kapsami

Ilk gun yalnizca su altyapi gelistirilecektir:

- Proje iskeleti.
- Saglik kontrolu endpoint'i.
- Ollama baglanti testi.
- Dosya yukleme altyapisi.

Ilk gun henuz su ozellikler eklenmeyecektir:

- OCR.
- RAG.
- ChromaDB entegrasyonu.
- Gercek soru-cevap ozelligi.
