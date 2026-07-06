# TESTING.md

# Yerel Belge Analiz ve Soru-Cevap Sistemi — Test ve Doğrulama

Bu dosya, geliştirilen belge analiz ve soru-cevap sisteminin nasıl test edildiğini, hangi belge tiplerinde nasıl davrandığını, başarılı olduğu durumları ve sınırlı kaldığı noktaları açıklamak için hazırlanmıştır.

Testlerde amaç yalnızca sistemin cevap verip vermediğini görmek değil; aynı zamanda aşağıdaki noktaları da değerlendirmektir:

- PDF, JPG ve PNG belge yükleme
- PDF metin çıkarımı
- Görsel belgelerde OCR
- Türkçe ve İngilizce metin okuma
- Uzun belgelerde retrieval performansı
- Grafik/görsel yorumlama
- CV gibi yapılandırılmış belgelerde bilgi çıkarımı
- Belgede olmayan bilgi sorulduğunda sistem davranışı
- Timeout, threshold ve OCR kaynaklı sınırlar
- Kaynak gösterme davranışı

---

## Test Ortamı

Testler yerel geliştirme ortamında yapılmıştır.

Kullanılan ana bileşenler:

- Backend: FastAPI
- Frontend: Streamlit
- OCR: PaddleOCR
- PDF işleme: PyMuPDF
- Embedding: Ollama `qwen3-embedding:0.6b`
- Chat modeli: Ollama `qwen3:4b`
- Vector Store: ChromaDB
- İşletim sistemi: Windows
- Arayüz: Web tabanlı Streamlit UI

Sistem tamamen yerel çalışacak şekilde test edilmiştir. Ücretli veya bulut tabanlı LLM API kullanılmamıştır.

---

# Test Senaryoları

## T01 — Uzun Ders Notu PDF Testi

### Amaç

Sistemin uzun bir PDF dosyasında ilgili bilgiyi bulup kaynaklı cevap üretebilmesini test etmek.

### Kullanılan belge

- `Bilgisayarlı görü hafta 12.pdf`

### Soru

```text
otonom sistemlerde mimari nasıl oluşturulur?
```

### Beklenen davranış

Sistemin uzun PDF içerisinden otonom sistem mimarisiyle ilgili bölümleri bulması ve cevapta belgeye dayalı açıklama yapması beklenmiştir.

### Gözlenen sonuç

Sistem, otonom sistemlerde mimarinin katmanlı bir yapıya dayandığını, "Algı-Plan-Kontrol" döngüsünden bahsettiğini ve yedekli yapı kullanılması gerektiğini açıklamıştır. Cevapta kaynak numaraları verilmiştir.

### Değerlendirme

Başarılı.

Bu test, uzun PDF üzerinde retrieval ve kaynaklı cevap üretme akışının çalıştığını göstermiştir.

---
<img width="976" height="614" alt="image" src="https://github.com/user-attachments/assets/26831ec0-605d-48c0-8a65-67c316ed2ab5" />


## T02 — Grafik PNG Testi

### Amaç

Sistemin yalnızca metin tabanlı PDF'lerde değil, görsel olarak yüklenen bir grafikte de OCR/retrieval/QA akışını çalıştırabilmesini test etmek.

### Kullanılan belge

- `isik_sureleri.png`
<img width="712" height="424" alt="isik_sureleri" src="https://github.com/user-attachments/assets/0a6a3dc7-a4b3-457a-aeff-5fe979276a00" />

<img width="952" height="359" alt="image" src="https://github.com/user-attachments/assets/87ec5912-290c-4341-bca6-a41b0532eda3" />

---

## T03 — CV  Testi

### Amaç

Sistemin CV gibi yapılandırılmış belgelerden adayın bilgilerini doğru çıkarabilmesini test etmek.

### Kullanılan belge

- `CV.pdf`

<img width="898" height="390" alt="image" src="https://github.com/user-attachments/assets/98399958-acc9-4f45-9b60-3f60dc9f85a8" />

### Değerlendirme

Sistem CV içerisindeki eğitim bölümünü doğru şekilde bulmuş ve cevapta kaynak göstermiştir.

---

## T04 

### Amaç

Sistemin CV içerisindeki proje ve deneyim bilgilerini listeleyebilmesini test etmek.

### Kullanılan belge

- `CV.pdf`

### Soru

```text
Which projects has the candidate worked on?
```

### Beklenen davranış

Sistemin CV'deki projeleri ve teknik içerikleri doğru şekilde özetlemesi beklenmiştir.

### Gözlenen cevap

Sistem aşağıdaki projelerden bahsetmiştir:

- TÜBİTAK 2209-A University Students Research Projects
- TEKNOFEST Robotaxi Autonomous Vehicle Competition
- TUSAŞ çalışmaları
- HAVELSAN NER projesi

Ayrıca reinforcement learning, graph learning, NLP ve autonomous systems gibi alanları da belirtmiştir.

<img width="954" height="737" alt="image" src="https://github.com/user-attachments/assets/39558c12-da6d-4d01-9f23-696524f63d39" />

---

## T05 — CV Toplam İş Deneyimi Süresi Testi

### Amaç

Sistemin CV içerisindeki tarih aralıklarını kullanarak toplam deneyim süresini hesaplayıp hesaplayamadığını test etmek.

### Kullanılan belge

- `CV.pdf`

### Soru

```text
What is the total duration of Bilge Göksel's work experience?
```

### İlk gözlem

İlk denemede Ollama tarafında timeout problemi yaşanmıştır. Model cevap üretirken belirlenen süreyi aşmıştır.

### Çözüm

Timeout süresi artırılmıştır:

```env
OLLAMA_CHAT_TIMEOUT_SECONDS=180
```

Timeout artırıldıktan sonra sistem cevap üretmiştir.

### Gözlenen cevap
<img width="906" height="230" alt="image" src="https://github.com/user-attachments/assets/d5cedd45-b606-4176-a794-4a82e5e8b263" />


Sistem toplam deneyimi 14 ay olarak hesaplamıştır. Ancak hesaplama hatalıdır. TUSAŞ deneyimini 12 ay, HAVELSAN deneyimini 2 ay olarak yorumlamıştır.
Ancak TUSAŞ deneyimi 1 yıl değil, yaklaşık 6 aydır. HSD tarafındaki deneyim 1 yıl olarak değerlendirilmelidir.

Sistem ilgili bölümleri bulabilmiş ancak süre hesaplamasında yanlış yorum yapmıştır.

### Çıkarım

Bu test, sistemin doğrudan belge içinde yazan bilgileri aktarmada daha başarılı olduğunu; ancak tarih aralıklarından süre hesaplama gibi ek muhakeme gerektiren sorularda hata yapabildiğini göstermiştir.

---

## T06 — İngilizce  OCR Testi

### Amaç

Sistemin Türkçe ve İngilizce metin içeren bir sınav örnek sorusu ekran görüntüsünden OCR ile metin çıkarıp çıkaramadığını test etmek.

### Kullanılan belge

- `YDS-deneme.png`
<img width="1469" height="670" alt="YDS-deneme" src="https://github.com/user-attachments/assets/a969782e-759d-44b5-8d6f-47bb7f01e9a7" />


### Soru

```text
İngilizce yönergede ne yazıyor?
```

### İlk gözlem

İlk denemede sistem:

```text
Bu bilgi yüklenen belgelerde bulunamadı.
```

cevabını vermiştir.

### Sorun analizi

`/api/v1/search` endpoint'i ile sorgu test edildiğinde ilgili OCR chunk'larının aslında geldiği görülmüştür. Ancak similarity score değerleri düşük kalmıştır.

Örnek skorlar yaklaşık şu aralıktaydı:

```text
0.18 - 0.22
```

Bu nedenle QA servisi threshold nedeniyle LLM'yi çağırmadan "belgede bulunamadı" cevabı dönmüştür.

### Çözüm

`RETRIEVAL_MIN_SCORE` değeri daha toleranslı hale getirilmiştir:

```env
RETRIEVAL_MIN_SCORE=0.15
```

### Sonraki gözlenen cevap

Sistem İngilizce yönergeyi Türkçe olarak şu şekilde açıklamıştır:

```text
Bu sorular için İngilizce cümlelerin en doğru Türkçe çevirilerini ve Türkçe cümlelerin en doğru İngilizce çevirilerini seçin.
```

### Değerlendirme

Başarılı. Sistem İngilizce yönergeyi Türkçe olarak açıklayabilmiştir.

Bu test OCR, retrieval threshold ve QA entegrasyonu açısından önemli bir düzeltme sağlamıştır.

---

## T07 — Görselde Belirli İfade Arama Testi

### Amaç

Sistemin görsel içindeki belirli bir ifadeyi bulup bulamadığını test etmek.

### Kullanılan belge

- `YDS-deneme.png`

### Soru

```text
D seçeneğinde "day by day" ifadesi geçiyor mu?
```

### Gözlenen cevap

```text
Evet, D seçeneğinde "day by day" ifadesi bulunmaktadır. [1]
```

### Değerlendirme

Başarılı.

Bu test, OCR ile çıkarılan metin üzerinde noktasal bilgi aramanın çalıştığını göstermiştir.

---

## T08 

### Amaç

Sistemin sınav sorusu gibi yorumlama gerektiren bir görselde nasıl davrandığını gözlemlemek.

### Kullanılan belge

- `YDS-deneme.png`

### Soru

```text
Bu sorunun doğru cevabı hangisidir?
```

### Gözlenen cevap

Sistem doğru cevabın C olduğunu söylemiştir ve açıklama üretmiştir.

<img width="704" height="561" alt="image" src="https://github.com/user-attachments/assets/56a9ecbc-f963-4c7f-a671-264361158e0a" />


### Değerlendirme

Bu cevap kullanıcı açısından faydalı görünse de sistemin asıl amacı sınav sorusu çözmek değildir. Bu projede sistemin ana hedefi belge içeriğine dayalı cevap üretmektir.

Eğer belgede doğru cevap açıkça belirtilmiyorsa sistemin kesin ifade kullanması risklidir.

### Çıkarım

Bu test, LLM'nin belgeye dayalı bilgiyi yorumlayarak ek sonuç üretmeye çalışabileceğini göstermiştir.

---

## T09

### Amaç

Sistemin görseldeki İngilizce seçeneklerden belirli birini çıkarıp açıklayabilmesini test etmek.

### Kullanılan belge

- `YDS-deneme.png`

### Soru

```text
What does option C say?
```

### Gözlenen cevap

Sistem C seçeneğinin İngilizce metnini çıkarmış ve anlamını açıklamıştır.

### Değerlendirme
Bu test, İngilizce OCR ve seçili metin retrieval davranışının çalıştığını göstermiştir.

---

# Belgede Olmayan Bilgi Testleri

### T10 — Belgede Olmayan Bilgi Testi

#### Amaç

Sistemin aynı anda birden fazla belge seçiliyken belgede bulunmayan bilgileri uydurup uydurmadığını test etmek.

Bu testte aynı anda iki farklı belge seçilmiştir:

- `CV.pdf`
- `YDS-deneme.png`

Bu senaryoda sistemin hem adayla ilgili CV belgesini hem de sınav sorusu içeren görsel belgeyi birlikte dikkate alması beklenmiştir.

<img width="684" height="618" alt="Ekran görüntüsü 2026-07-06 183127" src="https://github.com/user-attachments/assets/62192139-2e8c-4da0-ab66-e69d1395329b" />


#### Gözlenen Sonuç

Sistem üç soruda da aynı davranışı göstermiştir:

```text
Bu bilgi yüklenen belgelerde bulunamadı.
```

#### Değerlendirme

Başarılı.

Bu test, sistemin birden fazla belge seçili belgelerde bulunmayan bilgileri üretmemeye çalıştığını göstermiştir.

Bu davranış, sistemin halüsinasyon azaltma yaklaşımının temel senaryolarda çalıştığını göstermektedir.

#### Not

Bazı cevaplarda kaynaklar expander'ı görünse de cevap metni belgede bilgi bulunmadığını açıkça belirtmiştir. Bu durum ileride arayüz tarafında iyileştirilebilir; `found_in_documents=false` olduğunda kaynaklar alanı tamamen gizlenebilir.


# Genel Sonuç

Testler sonucunda sistemin işlevsel bir MVP seviyesine ulaştığı görülmüştür.

Sistem:

- Belge yükleyebiliyor.
- PDF ve görsel belgelerden metin çıkarabiliyor.
- Belgeleri chunk'lara ayırıp embedding üretebiliyor.
- ChromaDB üzerinden retrieval yapabiliyor.
- Ollama/Qwen ile belgeye dayalı cevap üretebiliyor.
- Cevaplarda kaynak gösterebiliyor.
- Belgede bulunmayan bilgilerde cevap üretmemeye çalışıyor.

Sistem genel belge analizi ve soru-cevap için uygun bir MVP seviyesindedir. Üretim ortamına taşınmadan önce daha gelişmiş doğrulama, tablo/grafik işleme ve hesaplama destekli modüllerle güçlendirilebilir.
