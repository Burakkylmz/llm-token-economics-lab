# LAB10 — Conversation Summarization & Context Economics

> **Katılımcı rehberi** · Python / Groq API / Docker Compose\
> **Kapsam:** Yalnızca *Conversation Summarization*; Full History, karşılaştırma için kontrol koşuludur.\
> **Amaç:** Özetleme ile context küçültmenin, bilgi kaybının ve toplam API maliyetinin birlikte nasıl değiştiğini **ölçmek**.

## 1. Bu laboratuvarda ne öğreneceksiniz?

Çok turlu chatbotlarda önceki konuşmanın tamamı her çağrıya eklendiğinde input token tüketimi büyüyebilir. LAB10, uzun bir görüşmenin LLM ile **bir kez özetlenip** sonraki sorulara bağlam olarak verilmesini inceler.

Deneyin cevaplamak istediği üç soru:

1. **Context compression:** Özetleme sonraki API çağrılarına gönderilen input token sayısını ne kadar düşürüyor?
2. **Information retention:** Özet, sipariş numarası, güncel karar, tarih ve belirsizlik gibi kritik bilgileri koruyor mu?
3. **Token economics:** Summary üretiminin maliyeti de eklendiğinde toplam maliyet düşüyor mu? Tasarruf hangi follow-up çağrısında başlıyor?

> **Önemli:** Daha kısa context, her zaman daha düşük **toplam** maliyet veya daha iyi cevap demek değildir.

## 2. Deney tasarımı

LAB10 aynı sentetik konuşma ve aynı **golden follow-up questions** üzerinde üç koşulu karşılaştırır:

| Condition | Modelin gördüğü geçmiş | Summary üretimi |
|---|---|---|
| `full_history_baseline` | Görüşmenin tamamı | Yok |
| `summarization_low` | Low effort ile üretilen summary | `reasoning_effort=low` |
| `summarization_medium` | Medium effort ile üretilen summary | `reasoning_effort=medium` |

**Provider:** Groq API\
**Model:** `openai/gpt-oss-20b` (paylaşılan başarılı çalıştırmadaki model)\
**Default dataset:** `customer_support_001` — kurgusal müşteri destek görüşmesi

Summary üretiminde reasoning effort değişir. Sonraki soruları yanıtlayan modelin ayarları ise karşılaştırmanın adil olması için aynı tutulmalıdır. Her summary, ilgili follow-up sorularında yeniden oluşturulmadan kullanılır.

### Default profilin çağrı hesabı

| İşlem | Mantıksal API çağrısı |
|---|---:|
| Full History ile 4 soru | 4 |
| Low summary üretimi | 1 |
| Low summary ile aynı 4 soru | 4 |
| Medium summary üretimi | 1 |
| Medium summary ile aynı 4 soru | 4 |
| **Toplam planlanan** | **14** |

Bu sayı **mantıksal deney çağrısı** sayısıdır. SDK/operational retries varsa gerçek HTTP attempt sayısı farklı olabilir. Gerçek API kullanımı ve fiyatlandırma sağlayıcının koşullarına bağlıdır.

## 3. Ön hazırlık — Windows PowerShell ve Docker

Gereksinimler: Docker Desktop, Docker Compose, projeye erişim ve gerçek deney için geçerli `GROQ_API_KEY`.

Komutları repository kök dizininden çalıştırın:

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

> Yol bilgisayarınızda farklıysa kendi repository yolunuzu kullanın.

### 3.1 API anahtarı

Repository'deki `.env.example` dosyasını `.env` olarak kopyalayıp `GROQ_API_KEY` değerini ekleyin. `.env` içeriğini kimseyle paylaşmayın ve Git'e eklemeyin.

```powershell
Copy-Item .env.example .env
```

`.env` dosyasını düzenleyin. API anahtarını kaynak koda, Markdown dosyasına veya ekran görüntüsüne yazmayın.

### 3.2 Live konfigürasyonu

Güvenli varsayılan ayar gerçek API çağrılarını kapatır. Kendi `live.toml` dosyanızı oluşturun:

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` içinde şu ayarı kontrol edin:

```toml
[execution]
live_api_enabled = true
```

Diğer konfigürasyon bölümlerini silmeyin; örnek yalnızca değiştirilecek alanı gösterir. `config/live.toml` dosyasını Git'e göndermeyin.

### 3.3 Docker image'ını oluşturma/güncelleme

```powershell
docker compose build lab-runner
```

**Not:** Bu projede `config/` Docker image'ına build sırasında kopyalanır. `config/live.toml` dosyasını oluşturduktan veya ilgili LAB10 kodunu güncelledikten sonra image'ı yeniden build edin.

## 4. LAB10'u çalıştırma komutları

### A. CLI seçeneklerini görüntüleme (API maliyeti yok)

```powershell
docker compose run --rm lab-runner lab10 --help
```

`--profile`, `--dry-run` veya `--show-io` gibi seçenekleri kullanmadan önce bu çıktıyla mevcut yerel sürümde desteklendiklerini doğrulayın.

### B. Default deney planını görüntüleme — Dry Run

```powershell
docker compose run --rm lab-runner lab10 all --profile default --dry-run --config config/default.toml
```

Beklenen davranış: Deney planı oluşturulur; gerçek Groq API çağrısı yapılmaz, summary veya yanıt üretilmez. Planlanan çağrı sayısı **14** olmalıdır.

### C. Default Grand Test — Gerçek API çağrıları

**Aşağıdaki komut, LAB10 için başarılı çalıştırma çıktısıyla doğrulanmıştır:**

```powershell
docker compose run --rm lab-runner lab10 all --profile default --live --config config/live.toml
```

Bu deneyde `customer_support_001` üzerinden **14 mantıksal çağrı** planlanır. Gerçek token ve maliyet tüketimi oluşur.

### D. Extended profil — İsteğe bağlı

```powershell
docker compose run --rm lab-runner lab10 all --profile extended --live --config config/live.toml
```

Tasarlanan Extended profil, iki dataset ve dörder soru için **28 mantıksal çağrı** planlar. **Bu komutun yerel sürümünüzdeki davranışını `lab10 --help` ve öncelikle `--dry-run` ile doğrulayın**; paylaşılan çalıştırma çıktısı yalnızca Default profili doğruluyor.

### E. Daha ayrıntılı input/output incelemesi

LAB10'un yerel CLI'sı `--show-io` parametresini destekliyorsa, **yalnızca sentetik dataset ile** çalıştırın:

```powershell
docker compose run --rm lab-runner lab10 all --profile default --live --show-io --config config/live.toml
```

**Dikkat:** Bu komut yeni bir canlı deney başlatabilir; yalnızca mevcut sonuçları görüntülemek için yeniden çalıştırmayın. Önce kaydedilmiş sonuçları inceleyin.

## 5. Deneyin akışı: Önce / Özetleme / Sonra

### Aşama 1 — Original Conversation

Orijinal görüşmenin tamamı baseline condition'da modele gönderilir. Örneğin görüşmede sipariş numarası, teslimat tarihi, değişen teslimat tercihi ve bekleyen iade talebi bulunabilir.

### Aşama 2 — Summary Generation

Aynı orijinal konuşma iki kez özetlenir: `low` ve `medium` reasoning effort. Şunlar kaydedilir:

- Summarization için **gerçekte gönderilen prompt ve conversation**
- Üretilen **summary metni**
- Summary generation `prompt_tokens`, `completion_tokens`, `reasoning_tokens`
- Summary üretim latency'si ve estimated cost

### Aşama 3 — Follow-up Evaluation

Aynı dört golden question üç koşulda cevaplanır: Full History, Low Summary, Medium Summary. İdeal karşılaştırma, hem **gönderilen request'i** hem **model cevabını** hem de **golden expected answer** değerini gösterir.

Örneğin:

```text
Question: What is the customer's order ID?
Expected: ORD-74291

Full History Answer: ORD-74291
Low Summary Answer: ORD-74291
Medium Summary Answer: ORD-74291
```

Bu, yalnızca **bir örnek** sorudur; raporun tüm sorulardaki doğruluğu ayrıca incelenmelidir.

## 6. Metriklerin anlamı

| Metrik | Nasıl yorumlanır? |
|---|---|
| `Summarization Input Tokens` | Summary oluşturmak için modele gönderilen istek tokenları |
| `Summarization Completion Tokens` | Summary çağrısının toplam output tokenları; reasoning tokens dahil olabilir |
| `Reasoning Tokens` | Summary üretiminde harcanan reasoning tokenları; **completion üzerine ikinci kez eklenmez** |
| `Full History Request Tokens` | Full History ile gönderilen downstream request'in toplam input tokenları |
| `Summary Request Tokens` | Summary ile gönderilen eşdeğer downstream request'in toplam input tokenları |
| `Context Reduction` | Eşdeğer downstream request'lerin input token farkının yüzdesi |
| `Fact Retention Rate` | Golden dataset'teki önemli olguların summary'de doğru korunma oranı |
| `Correct Answers` | Sonraki golden questions'ın doğru cevaplandığı sayı |
| `Summary Generation Cost` | Tek seferlik summary üretiminin tahmini maliyeti |
| `Follow-Up Cost` | İlgili condition'daki tüm takip sorularının tahmini maliyeti |
| `Total Workflow Cost` | Summary generation + follow-up; baseline için yalnızca follow-up |
| `Net Savings` | Baseline toplam maliyeti − summarization toplam maliyeti |
| `Break-Even` | Summary'nin baştaki maliyetini, ölçülen takip çağrılarında ne zaman telafi ettiği |

### Formüller

**Request-level input reduction:**

\[
\text{Reduction (\%)} = 100 \times \frac{T_{\text{full request}}-T_{\text{summary request}}}{T_{\text{full request}}}
\]

**Cumulative cost:**

\[
C_{\text{full}}(N)=\sum_{i=1}^{N} C_{\text{full},i}
\]

\[
C_{\text{summary}}(N)=C_{\text{summary generation}}+\sum_{i=1}^{N} C_{\text{summary},i}
\]

**Net savings:**

\[
\text{Net Savings}=C_{\text{full}}-C_{\text{summary}}
\]

Pozitif Net Savings tasarruf, negatif değer daha yüksek toplam maliyet demektir. **Break-Even: NOT REACHED**, ölçülen sorular süresince başlangıç özetleme maliyetinin telafi edilemediği anlamına gelir; gelecekte hiç telafi edilemeyeceği anlamına gelmez.

**Token ölçüm uyarısı:** `prompt_tokens` bütün request'i (instructions, context, soru ve formatting) ölçer; tek başına yalnızca konuşma metninin token sayısı değildir. `Fact Retention` ile `Answer Accuracy` farklı kalite boyutlarıdır.

## 7. Referans canlı çalıştırmanın yorumu

Aşağıdaki değerler **29 Ekim 2026'da paylaşılan Default Grand Test çıktısına** aittir; yeni çalıştırmalar için garanti değildir.

| Metrik | Full History | Summary Low | Summary Medium |
|---|---:|---:|---:|
| Correct Answers | 4/4 | 4/4 | 4/4 |
| Context Reduction | — | %49,9 | %50,6 |
| Fact Retention | — | %71,4 | %85,7 |
| Summary Generation Cost | — | $0,00022222 | $0,00044092 |
| Follow-Up Cost | $0,00037162 | $0,00020272 | $0,00019913 |
| Total Workflow Cost | $0,00037162 | $0,00042495 | $0,00064005 |
| Net Savings | — | -$0,00005333 | -$0,00026842 |
| Break-Even | — | NOT REACHED | NOT REACHED |

> Sayısal değerler rapor ekranında yuvarlanabildiği için gösterilen alt toplamlarla toplam maliyet arasında son basamakta fark olabilir.

**Buradan çıkarılacak üç ders:**

1. Özetleme sonraki request'lerde input context'ini yaklaşık yarıya indirmiştir.
2. Medium, bu örnekte daha yüksek fact retention sağlamıştır; fakat summary üretim maliyeti de artmıştır.
3. Dört follow-up sorusu sonunda özetleme yöntemlerinden hiçbiri toplam maliyet tasarrufuna ulaşmamıştır. Buna rağmen her iki summary koşulu da 4/4 soruyu doğru yanıtlamıştır.

**Yanlış çıkarım yapmayın:** 4/4 doğru cevap, tüm kritik bilgilerin korunmuş olduğu anlamına gelmez. Ayrıca tek dataset ve dört soru, hangi reasoning effort'ın genel olarak daha iyi olduğunu kanıtlamaz.

## 8. Sonuçları ve grafikleri bulma

Önceki çalıştırma şu klasöre sonuç yazmıştır:

```text
results/lab10_conversation_summarization/lab10-20261029T082117Z/
```

Grafikler:

```text
results/lab10_conversation_summarization/lab10-20261029T082117Z/charts/
```

Yeni çalıştırmanızın klasörü farklı olacaktır; **terminalin sonundaki `Results Directory` değerini esas alın.**

PowerShell üzerinden en son sonuç klasörünü bulmak için:

```powershell
$latest = Get-ChildItem .\results\lab10_conversation_summarization -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$latest.FullName
Get-ChildItem $latest.FullName
```

Analiz raporunu açmak için (dosya varsa):

```powershell
Get-Content (Join-Path $latest.FullName 'analysis_report.md') -Encoding UTF8
```

Özellikle şu tür sonuç kayıtlarını arayın: generated summaries, request/response inspection, token comparison, cost comparison, quality evaluation, break-even analysis ve charts. **Dosyaların kesin isimleri yerel LAB10 uygulama sürümüne bağlı olabilir.**

**Önceki sonuçları incelemek için Live komutunu tekrar çalıştırmayın:** Bu yeni Groq çağrıları ve ek maliyet oluşturur.

## 9. Sınıf içi uygulama görevi

1. `--help` çıktısını inceleyin ve varsayılan deneyin kapsamını açıklayın.
2. `--dry-run` ile 14 mantıksal API çağrısının hangi aşamalarda kullanılacağını gösterin.
3. Eğitim görevlisinin onayıyla Default Live Grand Test'i çalıştırın.
4. Bir golden question seçerek **Full History request** ile **Summary request** içeriğini karşılaştırın.
5. Low ve Medium tarafından üretilen summary metinlerinde korunan ve eksik kalan bilgileri bulun.
6. `Context Reduction`, `Fact Retention Rate`, `Correct Answers` ve `Total Workflow Cost` değerlerini yorumlayın.
7. `Break-Even` gerçekleşip gerçekleşmediğini raporlayın. Gerçekleşmediyse bunun nedenini açıklayın.

### Tartışma soruları

- %50 daha kısa context, neden %50 daha düşük toplam maliyet anlamına gelmez?
- Summary üretiminin maliyetini hesaba katmazsak hangi hatalı sonuca varırız?
- Low ve Medium aynı soruları doğru cevapladığı halde fact retention neden farklı olabilir?
- Daha pahalı summary hangi durumlarda tercih edilebilir?
- Daha fazla follow-up soru gelmesi break-even açısından neyi değiştirebilir?
- Eksik veya yanlış bir summary hangi üretim risklerine yol açar?

## 10. Sık karşılaşılan sorunlar

| Sorun | Kontrol edilmesi gerekenler |
|---|---|
| `lab10` komutu tanınmıyor | LAB10 kodlarının yerel repository'de bulunduğunu ve image'ın yeniden build edildiğini doğrulayın. |
| Live execution not allowed | `config/live.toml` içinde `execution.live_api_enabled = true` olduğunu ve image'ın bu dosya oluşturulduktan sonra build edildiğini kontrol edin. |
| `GROQ_API_KEY` eksik | `.env` dosyasını ve Docker Compose environment aktarımını kontrol edin. Anahtarı terminal çıktısında paylaşmayın. |
| HTTP 429 / rate limit | İstekleri üst üste yeniden göndermeyin; mevcut kota durumunu ve provider yanıtlarını kontrol edin. |
| Sonuç dosyası görünmüyor | Terminalin verdiği `Results Directory` yolunu ve `./results` volume mount'unu kontrol edin. |
| `--show-io` tanınmıyor | Önce `lab10 --help` kullanın; yerel sürümde seçenek desteklenmiyor olabilir. |
| `Break-Even: NOT REACHED` | Bu tek başına hata değildir; özetleme maliyeti henüz takip çağrılarında karşılanmamıştır. |

## 11. Güvenlik ve maliyet notu

- `--live` gerçek Groq API çağrıları gönderir. Hesabın kotasını tüketebilir ve maliyet doğurabilir.
- `--dry-run`, live çağrı yapmadan deney planını incelemek içindir.
- Summary çıktılarının ve tam request/response kayıtlarının yalnızca **sentetik eğitim verileriyle** paylaşılması güvenlidir; gerçek kullanıcı verileri için izin ve veri koruma politikaları gerekir.
- Testlerin amacı belirli bir sonucu zorlamak değil, beklenenin aksine daha maliyetli veya daha düşük kaliteli sonuçları da görünür kılmaktır.

---

### Öğrenme çıktısı

LAB10 sonunda öğrenci şunu açıklayabilmelidir:

**Conversation Summarization bir context compression tekniğidir. Gerçek faydası, yalnızca küçülen input token miktarıyla değil; summary üretiminin maliyeti, sonraki çağrıların toplam maliyeti ve korunan bilgi kalitesi birlikte ölçülerek belirlenir.**
