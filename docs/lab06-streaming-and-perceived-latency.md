# LAB06 — Streaming & Perceived Latency

> **Katılımcı rehberi** | Python · Groq API · Docker Compose\
> **Kapsam:** Non-Streaming ve Streaming karşılaştırması, First Visible Content, Perceived Latency, Total Latency, Throughput ve Answer Completeness.\
> **Önerilen ana deney:** `lab06 all --profile default` — varsayılan olarak **10 mantıksal Groq API çağrısı**.

---

## 1. Bu laboratuvarın amacı

LLM uygulamasında bir kullanıcı soru sorduğunda, modelin bütün yanıtı üretmesini beklemek yerine üretilen içeriği parça parça göstermek mümkündür. Bu yöntem **Streaming** olarak adlandırılır.

Streaming'in hedefi her durumda bütün cevabı daha erken tamamlamak değildir. Temel amaç, kullanıcının yanıtın ilk anlamlı bölümünü daha erken görmesini ve uygulamanın daha duyarlı hissedilmesini sağlamaktır.

Bu ayrım kritik önemdedir:

> **Perceived Latency (algılanan bekleme)** ile **Total Latency (yanıtın tamamlanma süresi)** aynı şey değildir.

LAB06 aynı görevi, aynı model ve benzer üretim ayarlarıyla iki farklı aktarım biçiminde çalıştırır:

- **Non-Streaming:** Model yanıtı tamamlanınca tek seferde teslim edilir.
- **Streaming:** Model yanıtı geldikçe parçalar (chunks) halinde teslim edilir.

Öğrenciler, gerçek Groq API yanıtlarında zamanlamayı, token tüketimini, yanıt bütünlüğünü ve tahmini maliyeti karşılaştıracaktır.

### 1.1 Deney sonunda neler öğrenmelisiniz?

1. `stream=False` ve `stream=True` farkını açıklamak.
2. **First Stream Event** ile **First Visible Content** arasındaki farkı göstermek.
3. TTFT ölçümü ile first-event gözlemini birbirine karıştırmamak.
4. Perceived Latency Gain ve Total Latency Difference formüllerini uygulamak.
5. Streaming event timeline'ındaki reasoning, visible content ve usage olaylarını ayırmak.
6. Completion tokens/s, visible characters/s ve chunk throughput arasındaki farkı anlamak.
7. Streaming token usage bilgisi bulunmadığında maliyet uydurmamak.
8. Response streaming'in cevap kalitesi ve toplam latency açısından trade-off oluşturabileceğini değerlendirmek.

---

## 2. Deney tasarımı: Ne karşılaştırılıyor?

| Özellik | Non-Streaming | Streaming |
|---|---|---|
| CLI condition | `non_streaming` | `streaming` |
| Response aktarımı | Tamamlanmış cevap | SSE event/chunk akışı |
| Kullanıcı ne zaman içerik görür? | Tam yanıt gelince | İlk anlamlı içerik chunk'ı gelince |
| Ölçülen ana süre | `non_streaming_total_latency_ms` | `first_visible_content_ms` |
| Tamamlanma süresi | Non-streaming total latency | `total_stream_latency_ms` |
| Event timeline | Yok | Var, etkinleştirilmişse |
| Token usage | API döndürürse | Stream usage chunk'ı döndürülürse |
| Kalite kontrolü | 5 bölümün varlığı | Aynı 5 bölümün varlığı |

### 2.1 Kontrol edilen değişkenler

Kaynak kodda iki koşula da aynı **Production API Incident Explanation** görevi gönderilir.

Eşit tutulan parametreler:

- `model`
- Prompt/messages
- `reasoning_effort`
- `max_completion_tokens`
- Görevin çıktı formatı
- Deterministic completeness evaluator

Asıl deney değişkeni response delivery mode'dur: streaming veya non-streaming.

Provider yükü, network koşulları, farklı completion uzunlukları ve doğal sampling varyasyonu tümüyle kontrol edilemez. Bu nedenle sonuçlar tek bir çağrı üzerinden evrenselleştirilmemelidir.

---

## 3. LAB06'da kullanılan gerçek senaryo

Her koşulda aynı sentetik incident açıklanır:

```text
At 14:32 UTC, the checkout-service API began returning HTTP 504
Gateway Timeout errors for approximately 8 minutes.

Root cause: connection pool exhaustion in the payment-gateway client.
The issue followed a deployment that removed the client's connection
timeout configuration.

Roughly 3,200 checkout requests failed.

Restarting the payment-gateway client pods restored normal operation
within 90 seconds of the restart.
```

Yukarıdaki metin, kaynağın öğretim amaçlı kısaltılmış özetidir. Tam fixture:

`src/labs/lab06_streaming/experiments.py`

Modelden **beş başlık** altında açıklama istenir:

1. `Incident Summary`
2. `Possible Causes`
3. `User Impact`
4. `Immediate Mitigation`
5. `Prevention Measures`

Her başlığın ardından boş olmayan bir açıklama bulunmalıdır.

### 3.1 Neden bu görev seçildi?

- Birden fazla bölüm içeren görünür çıktı ürettirir.
- Streaming sırasında içeriklerin parça parça gelişini gözlemlemeyi kolaylaştırır.
- İki aktarım biçiminde de aynı görev kullanılır.
- Yanıtın eksiksizliği **deterministic** olarak kontrol edilebilir.

### 3.2 Önemli değerlendirme sınırı

Evaluator, beş başlığın altında boş olmayan metin bulunup bulunmadığını kontrol eder.

**Bu kontrol olay açıklamasının teknik olarak doğru olduğunu kanıtlamaz.**

Örneğin `Prevention Measures` başlığı ve altında yanlış bir öneri varsa, structural completeness testi bunu tek başına yakalayamayabilir.

---

## 4. Streaming nasıl çalışıyor?

### 4.1 Non-Streaming

```text
User Request
    |
    v
Groq API
    |
    | Model response is generated
    |
    v
Complete Response Arrives
    |
    v
Display the full answer
```

Bu durumda yanıtın tamamı gelene kadar uygulama içerik göstermeyebilir.

### 4.2 Streaming

```text
User Request
    |
    v
Groq API
    |
    +--> Event 1 (metadata / role / reasoning)
    |
    +--> Event 2 (first visible content)
    |
    +--> Event 3 (additional visible content)
    |
    +--> Event 4 (additional visible content)
    |
    +--> Final event / usage metadata
    |
    v
Stream Complete
```

**SSE (Server-Sent Events)** yaklaşımında sunucu, yanıt parçalarını bağlantı üzerinden sırayla gönderir. Her chunk'ın mutlaka kullanıcıya gösterilecek metin taşıması gerekmez.

LAB06'nin ölçtüğü temel zamanlama farkları bundan kaynaklanır.

---

## 5. Kritik metrikler: First Event, First Visible ve TTFT

### 5.1 First Stream Event

**Alan:** `first_stream_event_ms`

İstek başlangıcından istemcinin ilk streaming event'ini aldığı ana kadar geçen süredir.

İlk event:

- Role metadata olabilir.
- Reasoning delta olabilir.
- Başka bir kontrol/metadata event'i olabilir.

Bu nedenle **ilk event mutlaka görünür yazı değildir**.

### 5.2 First Visible Content

**Alan:** `first_visible_content_ms`

İstek başlangıcından, istemcinin ilk **boşluk dışı, kullanıcıya gösterilebilir** içerik parçasını aldığı ana kadar geçen süredir.

Bir chunk sadece whitespace içeriyorsa, kaynak kod event'i `VISIBLE_CONTENT` olarak sınıflandırabilse de ilk **anlamlı** görünür içerik zamanı henüz gerçekleşmiş sayılmaz.

**Bu, LAB06'nın perceived latency değerlendirmesinin ana metriğidir.**

### 5.3 TTFT — Time to First Token

TTFT, genel olarak ilk tokenın alınmasına kadar geçen süreyi anlatan bir latency terimidir.

Fakat LAB06 kodunda:

```text
ttft_ms = None
```

olarak kalır. Çünkü mevcut uygulama first SSE event'inden bağımsız, doğrulanmış provider-level per-token TTFT ölçümü elde etmemektedir.

**Şu üç ifadeyi eşitlemeyin:**

```text
first_stream_event_ms
first_visible_content_ms
ttft_ms
```

İlk iki değer istemci tarafından gözlenebilir; üçüncüsü bu LAB'ın mevcut kayıtlarında mevcut değildir.

### 5.4 First Event-to-Visible Gap

**Alan:** `first_event_to_visible_gap_ms`

Formül:

```text
first_event_to_visible_gap_ms =
    first_visible_content_ms - first_stream_event_ms
```

Temsili örnek:

```text
First Stream Event:       170 ms
First Visible Content:    610 ms

First Event-to-Visible Gap = 440 ms
```

Bu boşluk tek başına “model 440 ms reasoning yaptı” anlamına gelmez. Reasoning, transport, kuyruklama, event dağılımı ve diğer etkenler bulunabilir.

---

## 6. Perceived Latency Gain ile Total Latency Difference

LAB06'nin iki ana karşılaştırması farklı sorulara cevap verir.

### 6.1 Perceived Latency Gain

**“Streaming kullanıcıya anlamlı içeriği ne kadar erken gösterdi?”**

Formül:

```text
Perceived Latency Gain (ms) =
    Non-Streaming Total Latency
    - Streaming First Visible Content
```

Yüzde:

```text
Perceived Latency Gain (%) =
    100 × Perceived Latency Gain
    / Non-Streaming Total Latency
```

Örnek — **temsili değerlerdir, gerçek deney sonucu değildir**:

```text
Non-Streaming Total Latency: 2,400 ms
Streaming First Visible:      600 ms

Perceived Latency Gain:     1,800 ms
Perceived Latency Gain:        75%
```

Burada kullanıcı streaming koşulunda ilk anlamlı içeriği, non-streaming yanıtının tamamının gelmesine kıyasla 1.800 ms önce görmüştür.

### 6.2 Total Latency Difference

**“Streaming işlemi bütünüyle daha erken mi tamamlandı?”**

Formül:

```text
Total Latency Difference (ms) =
    Non-Streaming Total Latency
    - Streaming Total Stream Latency
```

Örnek:

```text
Non-Streaming Total Latency: 2,400 ms
Streaming Total Latency:     2,700 ms

Total Latency Difference:     -300 ms
```

Bu durumda:

- Streaming görünür içeriği **daha erken** teslim etmiş olabilir.
- Fakat tüm stream'in tamamlanması **300 ms daha uzun sürmüştür**.

**Bu çelişki değildir.** Perceived Latency ve Total Latency farklı ölçümlerdir.

### 6.3 Negatif değerleri doğru yorumlayın

`perceived_latency_gain_ms` negatifse streaming ilk anlamlı içeriği, karşılaştırılan non-streaming tam yanıttan **daha geç** göstermiştir.

`total_latency_difference_ms` negatifse streaming tamamlama süresi **daha uzundur**.

LAB06 negatif değerleri sıfıra çevirmeden raporlar. Bunları sonuçlardan gizlemeyin.

---

## 7. Docker ortamının kurulması — Windows PowerShell

### 7.1 Gereksinimler

- Docker Desktop
- Docker Compose
- LAB06 içeren proje klasörü
- Canlı deneyler için Groq API anahtarı
- İnternet erişimi

Docker kullanıldığı için bilgisayarınıza ayrıca Python ve `uv` kurmak zorunda değilsiniz.

### 7.2 Proje kök klasörüne geçin

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Kendi bilgisayarınızda proje farklı yerdeyse klasör yolunu değiştirin.

### 7.3 API anahtarınızı tanımlayın

```powershell
Copy-Item .env.example .env
```

`.env` dosyasında:

```dotenv
GROQ_API_KEY=your_api_key_here
```

**API anahtarınızı asla GitHub'a commit etmeyin veya paylaşmayın.**

### 7.4 Live konfigürasyonunu oluşturun

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` dosyasının `[execution]` bölümündeki ayar:

```toml
[execution]
live_api_enabled = true
```

olmalıdır.

Diğer ayarları silmeyin.

Canlı çağrı için iki güvenlik koşulunun birlikte sağlanması gerekir:

1. CLI parametresi `--live`.
2. Config değeri `execution.live_api_enabled = true`.

Varsayılan güvenli `config/default.toml` dosyasını yerinde değiştirmeyin.

### 7.5 Docker image build

```powershell
docker compose build lab-runner
```

Dockerfile `config/` dosyalarını image'a kopyaladığından, `live.toml` oluşturulduktan veya kod güncellendikten sonra rebuild gerekir.

### 7.6 LAB06 yardım ekranı

```powershell
docker compose run --rm lab-runner lab06 --help
```

API çağrısı yapmaz; geçerli experiment ve option değerlerini gösterir.

---

## 8. LAB06 Docker komutları

### A. Default Grand Test — 10 çağrı

```powershell
docker compose run --rm lab-runner lab06 all --profile default --live --config config/live.toml
```

Varsayılan plan:

```text
5 paired repetitions
× 2 conditions (non_streaming + streaming)
= 10 logical API calls
```

**Önerilen ana sınıf uygulaması budur.**

### B. Dry-Run — API çağrısı yapılmaz

```powershell
docker compose run --rm lab-runner lab06 all --profile default --dry-run --config config/default.toml
```

Dry-run:

- Deney planını gösterir.
- Hangi koşulların hangi pair içinde çalışacağını kaydeder.
- Conservative estimated maximum cost hesaplar.
- `experiment_plan.csv` oluşturur.
- **Groq API isteği göndermez.**

Dry-run sonucu gerçek latency veya token ölçümü değildir.

### C. Pilot Grand Test — 4 çağrı

```powershell
docker compose run --rm lab-runner lab06 all --profile pilot --live --config config/live.toml
```

Plan:

```text
2 pairs × 2 conditions = 4 calls
```

İlk canlı kontrol için uygundur.

### D. Extended Grand Test — 16 çağrı

```powershell
docker compose run --rm lab-runner lab06 all --profile extended --live --config config/live.toml
```

Plan:

```text
8 pairs × 2 conditions = 16 calls
```

Daha fazla tekrarın latency dağılımını görmek için kullanılabilir. Bu yine de üretim ortamı benchmark'ı değildir.

### E. Sadece Non-Streaming

```powershell
docker compose run --rm lab-runner lab06 non_streaming --profile default --live --config config/live.toml
```

**5 mantıksal çağrı** planlar. Paired comparison üretmez, çünkü aynı run içinde streaming partneri yoktur.

### F. Sadece Streaming

```powershell
docker compose run --rm lab-runner lab06 streaming --profile default --live --config config/live.toml
```

**5 mantıksal çağrı** planlar. Streaming event timeline'ını ayrı incelemek için uygundur.

### G. Tekrar sayısını değiştirme

```powershell
docker compose run --rm lab-runner lab06 all --profile default --repetitions 3 --live --config config/live.toml
```

Plan:

```text
3 pairs × 2 conditions = 6 calls
```

`--repetitions` parametresi verilirse profile'ın varsayılan pair sayısını override eder. Sonuçta oluşan çağrı sayısı `lab06.max_api_calls` limitini aşmamalıdır.

### H. Özel Run ID

```powershell
docker compose run --rm lab-runner lab06 all --profile default --experiment-run-id class-lab06-group-a --live --config config/live.toml
```

Sonuç dizini:

```text
results/lab06_streaming/class-lab06-group-a/
```

Ölçüm içeren aynı Run ID tekrar kullanılamaz. Yeni denemede başka ID verin.

### I. Extended Dry-Run

```powershell
docker compose run --rm lab-runner lab06 all --profile extended --dry-run --config config/default.toml
```

**16 çağrılık** planı API çağrısı yapmadan görmenizi sağlar.

### Önemli not

Bu sayılar **planlanan mantıksal çağrı** sayılarıdır. Operational retry, error veya bütçe nedeniyle gerçek HTTP denemesi sayısı farklı olabilir.

---

## 9. LAB06 konfigürasyonu

Projede varsayılan LAB06 konfigürasyonu:

```toml
[lab06]
enabled = true
model = "openai/gpt-oss-20b"
default_profile = "default"
paired_repetitions = 5
reasoning_effort = "low"
max_completion_tokens = 768
randomize_condition_order = false
randomization_seed = 42
request_usage_when_supported = true
record_event_timeline = true
max_api_calls = 16
max_estimated_cost_usd = 0.05
```

### 9.1 `paired_repetitions`

Default profil için kaç pair oluşturulacağını belirler.

### 9.2 `reasoning_effort`

Streaming ve non-streaming için **aynı değerdir**. Böylece iki farklı reasoning effort etkisini streaming etkisiyle karıştırmayız.

### 9.3 `max_completion_tokens`

Modelin completion bütçesini sınırlar. Reasoning tokenları da bu bütçeyi kullanabilir.

### 9.4 `request_usage_when_supported`

Streaming kullanımında provider'dan token usage raporu istemek için kullanılır. Bu bilgi gönderilse bile provider'ın ilgili alanları gerçekten döndüreceği garanti edilmez.

### 9.5 `record_event_timeline`

Streaming chunk olaylarının sayısal zaman çizelgesini sonuç dosyalarına kaydeder.

**Tam prompt veya yanıt metni değil, metadata kaydedilir.**

---

## 10. Streaming Event Timeline

LAB06 gelen chunk'ları olay türlerine ayırır:

| Event Type | Ne anlama gelir? |
|---|---|
| `ROLE_ONLY` | Role metadata |
| `REASONING_DELTA` | Ayrı reasoning alanında veri |
| `VISIBLE_CONTENT` | Kullanıcıya gösterilebilir content parçası |
| `EMPTY_CONTENT` | Boş content string |
| `USAGE_ONLY` | Token usage metadata |
| `FINISH_EVENT` | Completion sonlandırma bilgisi |
| `OTHER_METADATA` | Diğer metadata |
| `UNKNOWN` | Açıklanamayan veya boş event |

### 10.1 Event timeline'da neler var?

`stream_events.csv` dosyasında örnek alanlar:

```text
execution_index
event_index
relative_arrival_ms
event_type
visible_content_character_count
cumulative_visible_character_count
has_reasoning_delta
has_finish_reason
has_usage
```

Temsili akış:

| Event | Arrival (ms) | Type | Visible chars |
|---|---:|---|---:|
| 1 | 120 | `ROLE_ONLY` | 0 |
| 2 | 210 | `REASONING_DELTA` | 0 |
| 3 | 430 | `REASONING_DELTA` | 0 |
| 4 | 710 | `VISIBLE_CONTENT` | 16 |
| 5 | 790 | `VISIBLE_CONTENT` | 28 |
| 6 | 1.680 | `USAGE_ONLY` | 0 |

Bu örnekte:

- First Stream Event = **120 ms**
- First Visible Content = **710 ms**
- First Event-to-Visible Gap = **590 ms**

**Bu değerler yalnızca öğretim amaçlı örneklerdir.**

### 10.2 Reasoning content neden görünür response değildir?

Reasoning-capable modeller bazı event'lerde kullanıcıya gösterilen nihai cevap dışında reasoning verileri üretebilir.

LAB06, reasoning delta'larını visible text olarak saymamayı ve reasoning içeriklerini sonuç dosyasına kaydetmemeyi hedefler.

**Görünür karakter sayısı, completion token sayısı değildir.**

---

## 11. Token Usage ve Cost Metrics

LAB06'nin ortak metrikleri:

| Metrik | Açıklama |
|---|---|
| `prompt_tokens` | Request input tokenları |
| `completion_tokens` | Toplam completion tokenları |
| `reasoning_tokens` | Provider raporluyorsa reasoning tokenları |
| `non_reasoning_completion_tokens` | Tutarlı ölçüm varsa completion eksi reasoning |
| `total_tokens` | Toplam token kullanımı |
| `cached_tokens` | Provider raporluyorsa cached input tokenları |
| `usage_status` | Usage bilgisi kullanılabilir mi? |
| `estimated_cost_usd` | Konfigüre fiyatlarla hesaplanan tahmini API maliyeti |

### 11.1 Reasoning tokenlarını iki kere saymayın

Temsili örnek:

```text
Prompt Tokens:                   300
Completion Tokens:               500
Reasoning Tokens:                180
Non-Reasoning Completion Tokens: 320
Total Tokens:                    800
```

Doğru toplam:

```text
300 + 500 = 800
```

Yanlış toplam:

```text
300 + 500 + 180 = 980
```

Reasoning tokenları completion tokenlarının içinde yer alır.

### 11.2 Streaming usage gelmezse?

İlgili response'da token usage yoksa:

```text
usage_status = USAGE_UNAVAILABLE
```

Token veya maliyet alanları `None` / `N/A` olabilir.

**`N/A`, sıfır demek değildir.**

Kaynak kod, streaming için usage istemeyi destekler; fakat provider her çağrıda usage raporlamak zorunda değildir.

### 11.3 Streaming daha az token mı kullanır?

Streaming, tek başına token azaltma tekniği değildir.

Aynı model ve benzer çıktı uzunluğunda, aktarım biçiminin değişmesi nedeniyle token maliyetinin sistematik olarak düşeceğini varsayamayız.

LAB06'nin öncelikli performans hedefi **perceived latency** farkını ölçmektir.

### 11.4 Estimated Cost

Yaklaşık ve anlaşılır formül:

\[
C_{\mathrm{input}} =
\frac{T_{\mathrm{uncached}} \times P_{\mathrm{input}} +
T_{\mathrm{cached}} \times P_{\mathrm{cached}}}{1{,}000{,}000}
\]

\[
C_{\mathrm{output}} =
\frac{T_{\mathrm{completion}}\times P_{\mathrm{output}}}{1{,}000{,}000}
\]

\[
C_{\mathrm{total}}=C_{\mathrm{input}}+C_{\mathrm{output}}
\]

Milyon token başına USD fiyatları config'ten alınır. Fiyatlar güncellenebileceği için gerçek faturayla birebir aynı sonuç garanti edilmez.

---

## 12. Throughput: Hangi hız ölçülüyor?

LAB06'daki throughput ölçümleri de farklı anlamlara sahiptir.

| Alan | Ne ölçer? |
|---|---|
| `effective_completion_tokens_per_second` | Completion token / istemci tarafından ölçülen toplam request süresi |
| `provider_completion_tokens_per_second` | Completion token / provider-reported completion processing time |
| `visible_characters_per_second` | Görünür karakter sayısı / toplam stream süresi |
| `content_chunks_per_second` | Görünür içerik chunk sayısı / toplam stream süresi |

### 12.1 Effective Completion TPS

\[
\mathrm{EffectiveTPS} =
\frac{\mathrm{completion\_tokens}}
{\mathrm{total\_latency\_ms}/1000}
\]

Örnek:

```text
Completion Tokens: 900
Total Latency:     1,500 ms

Effective TPS: 600 tokens/second
```

Bu, tüm request süresini paydada kullandığı için yalnızca saf model decoding hızını ölçmez.

### 12.2 Provider Completion TPS

\[
\mathrm{ProviderTPS} =
\frac{\mathrm{completion\_tokens}}
{\mathrm{provider\_completion\_time\_ms}/1000}
\]

Provider'dan completion time dönmezse bu değer hesaplanmaz.

### 12.3 Chunk sayısı neden token sayısı değildir?

Bir chunk:

- Hiç content taşımayabilir.
- Bir karakter taşıyabilir.
- Birden fazla kelime içerebilir.
- Yalnızca usage veya finish metadata'sı taşıyabilir.

Dolayısıyla:

```text
total_chunks != completion_tokens
```

ve

```text
visible_content_chunks != visible_content_tokens
```

---

## 13. Answer Completeness & Quality

LAB06 yanıtların beş başlık altında üretildiğini kontrol eder.

| Evaluation Status | Yorum |
|---|---|
| `COMPLETE` | Beş gerekli bölüm bulunmuş; yapısal bütünlük var |
| `INCOMPLETE` | Bir veya daha fazla bölüm eksik |
| `TRUNCATED` | Eksik yanıt ve provider `finish_reason=length` |
| `NO_VISIBLE_CONTENT` | Görünür yanıt yok |
| `STREAM_INTERRUPTED` | Streaming sırasında bağlantı/akış kesilmiş |
| `EVALUATION_UNAVAILABLE` | Değerlendirme yapılamıyor |
| `API_ERROR` | API çağrısı başarısız |

Tam yanıt:

```text
Incident Summary: ...
Possible Causes: ...
User Impact: ...
Immediate Mitigation: ...
Prevention Measures: ...
```

Bu beş bölümün her biri boş olmayan açıklamaya sahip olduğunda `COMPLETE` sonucu alınabilir.

Ancak:

**Structural Completeness ≠ Factual Correctness**

Model yanlış teknik yorumlarla beş bölüm oluşturursa, yapısal kontrol bunu tek başına kanıtlayamaz veya çürütemez.

### 13.1 Paired Quality Comparison

LAB06 eşleştirilmiş çağrılar için aşağıdaki statüleri kullanır:

| Sonuç | Anlamı |
|---|---|
| `BOTH_COMPLETE` | İki koşul da eksiksiz |
| `STREAMING_MORE_COMPLETE` | Streaming tamam, non-streaming eksik |
| `NON_STREAMING_MORE_COMPLETE` | Non-streaming tamam, streaming eksik |
| `BOTH_INCOMPLETE` | İkisi de eksik |
| `INSUFFICIENT_DATA` | Sağlıklı karşılaştırma yapılamıyor |

Bu analiz, Streaming'in perceived latency avantajını yorumlarken **yanıtın tamamlanmasını** göz ardı etmemek için önemlidir.

---

## 14. Terminal çıktısını okuma

### 14.1 Streaming ölçümü

Gerçek değerler her çalıştırmada farklı olabilir.

```text
Experiment: streaming
Pair: 1/5

First Stream Event:          ... ms
First Visible Content:       ... ms
First Event-to-Visible Gap:  ... ms
Last Visible Content:        ... ms
Total Stream Latency:        ... ms

Total Chunks:                ...
Visible Content Chunks:      ...
Visible Characters:          ...

Prompt Tokens:               ...
Completion Tokens:           ...
Reasoning Tokens:            ...
Token Usage Status:          AVAILABLE

Finish Reason:               ...
Answer Complete:             ...
Evaluation Status:           ...
Estimated Cost:              $...
```

İnceleme sırası:

1. First Stream Event ve First Visible farklı mı?
2. First Event-to-Visible Gap ne kadar?
3. Kaç chunk var, kaçında görünür içerik var?
4. Stream tamamlandı mı?
5. Usage mevcut mu?
6. Yanıt beş bölüm açısından complete mi?

### 14.2 Non-Streaming ölçümü

```text
Experiment: non_streaming
Pair: 1/5

Total Client Latency:        ... ms
Prompt Tokens:               ...
Completion Tokens:           ...
Reasoning Tokens:            ...
Effective Completion TPS:    ...
Provider Generation TPS:     ...

Finish Reason:               ...
Answer Complete:             ...
Estimated Cost:              $...
```

### 14.3 Pair Comparison

```text
-- Pair 1 Comparison --

Non-Streaming Full Response:  ... ms
Streaming First Visible:      ... ms
Streaming Total Latency:      ... ms

Perceived Latency Gain:       ... ms (...%)
Total Latency Difference:     ... ms

Quality Comparison:           ...
```

### 14.4 Grand Test Summary

```text
-- LAB06 Grand Test Summary --

Completed Pairs:               ...
Successful Calls:              ...

Mean First Stream Event:       ... ms
Mean First Visible Content:    ... ms
Mean Streaming Total Latency:  ... ms
Mean Non-Streaming Latency:    ... ms

Mean Perceived Latency Gain:   ... ms
Median Perceived Latency Gain: ... ms
Mean Total Latency Difference: ... ms

Streaming Usage Available:     .../...
Non-Streaming Usage Available: .../...

Answer Completeness:           .../... COMPLETE
Estimated Total Cost:          $...
```

**Özellikle dikkat:** Mean Perceived Latency Gain pozitif çıkarsa streaming'in ilk görünür içeriği daha erken sağladığı gözlenmiştir; bu, streaming toplam süresinin de daha kısa olduğu anlamına gelmez.

---

## 15. Sonuç dosyaları

Her run için ayrı klasör kullanılır:

```text
results/
└── lab06_streaming/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── stream_measurements.json
        ├── stream_measurements.csv
        ├── stream_events.csv
        ├── paired_comparison.csv
        ├── condition_summary.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Öğrenci neyi incelemeli? |
|---|---|
| `experiment_manifest.json` | Model, config, profil, plan, run durumu |
| `experiment_plan.csv` | Her pair'deki condition sırası |
| `stream_measurements.json/.csv` | Tekil çağrı metrikleri ve timing |
| `stream_events.csv` | Streaming event'lerin arrival time ve char count metadata'sı |
| `paired_comparison.csv` | Perceived gain, total latency ve quality farkları |
| `condition_summary.csv` | Her condition'ın istatistikleri |
| `experiment_summary.json` | Deneyin genel sayısal özeti |
| `analysis_report.md` | Teknik bulgular ve sınırlamalar |
| `charts/` | Grafikler |

**Önerilen inceleme sırası:**

1. `analysis_report.md`
2. `paired_comparison.csv`
3. `stream_measurements.csv`
4. `stream_events.csv`
5. `charts/`

LAB06 ham prompt, final response veya reasoning trace'leri kaydetmez; event timeline sayısal metadata ile sınırlıdır.

---

## 16. Grafikler nasıl yorumlanır?

LAB06 aşağıdaki dokuz grafik dosyasını üretmeyi hedefler:

| Grafik | Odak |
|---|---|
| `first_event_vs_first_visible.png` | İlk stream event'i ile ilk görünür içerik farkı |
| `perceived_latency_comparison.png` | Non-streaming tam yanıtı ile streaming ilk içeriği karşılaştırması |
| `total_latency_comparison.png` | İki condition'ın toplam tamamlanma süreleri |
| `paired_perceived_latency_gain.png` | Her pair'in kazanımı veya kaybı |
| `event_to_visible_gap.png` | Event başlangıcından görünür içeriğe geçen süre |
| `streaming_content_arrival_timeline.png` | Görünür içeriğin zaman içinde birikmesi |
| `effective_token_throughput.png` | Effective token/s karşılaştırması |
| `token_cost_comparison.png` | Token ve estimated cost etkisi |
| `streaming_performance_dashboard.png` | Genel performans paneli |

### 16.1 İlk incelenecek üç grafik

1. **`perceived_latency_comparison.png`** — Kullanıcı cevap görmeye ne zaman başlıyor?
2. **`total_latency_comparison.png`** — Yanıtın tamamı ne zaman bitiyor?
3. **`streaming_content_arrival_timeline.png`** — İçerik bir anda mı yoksa düzenli parçalar halinde mi geliyor?

Gerekli ölçümler yoksa bazı grafiklerin karşılaştırma içeriği sınırlı olabilir. Eksik değerleri sıfır olarak yorumlamayın.

---

## 17. Backfill — API çağrısı olmadan grafikleri yeniden üretme

Bütün mevcut sonuçlar için:

```powershell
docker compose run --rm lab-runner lab06 backfill --config config/default.toml
```

Belirli bir run için:

```powershell
docker compose run --rm lab-runner lab06 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasör adını yazın.

Backfill:

- Groq API çağrısı yapmaz.
- API key gerektirmez.
- Önceden kaydedilmiş ölçümleri okur.
- Grafik PNG'lerini yeniden üretir.
- Mevcut JSON/CSV/Markdown ölçümlerini değiştirmez.

---

## 18. Sınıf içi uygulama görevleri

### Görev 1 — Pilot ile ilk gözlem

Şu komutu çalıştırın:

```powershell
docker compose run --rm lab-runner lab06 all --profile pilot --live --config config/live.toml
```

İki pair için aşağıdaki tabloyu doldurun:

| Pair | First Stream Event (ms) | First Visible (ms) | Streaming Total (ms) | Non-Streaming Total (ms) |
|---|---:|---:|---:|---:|
| 1 | | | | |
| 2 | | | | |

**Soru:** Streaming'in ilk event'i ile kullanıcının gördüğü ilk içerik arasında neden fark olabilir?

### Görev 2 — Default Grand Test

Varsayılan 10 çağrılık testi çalıştırın:

```powershell
docker compose run --rm lab-runner lab06 all --profile default --live --config config/live.toml
```

`paired_comparison.csv` dosyasını inceleyin.

| Pair | Perceived Gain (ms) | Total Latency Difference (ms) | Quality |
|---|---:|---:|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Yorum:** Pozitif perceived gain ile negatif total latency difference aynı pair'de görülebilir mi?

### Görev 3 — Event Timeline

`stream_events.csv` dosyasından tek bir streaming çağrısı seçin.

Şu alanları inceleyin:

```text
event_index
relative_arrival_ms
event_type
visible_content_character_count
cumulative_visible_character_count
```

Aşağıdaki soruları cevaplayın:

1. İlk event hangi türdeydi?
2. İlk görünür içerik hangi event'te geldi?
3. Başta reasoning delta veya metadata event'i var mı?
4. Kaç event sıfır görünür karakter içeriyor?
5. Final usage ayrı bir event olarak raporlandı mı?

### Görev 4 — Matematiksel Perceived Gain

Gerçek bir pair seçip elle hesaplayın:

```text
Gain = Non-Streaming Total - Streaming First Visible
```

Ardından yüzde hesaplayın:

```text
Gain % = 100 × Gain / Non-Streaming Total
```

Hesabınızı `paired_comparison.csv` ile karşılaştırın.

**Önemli:** Paydada `Streaming Total Latency` kullanmayın. Kaynak kod baseline olarak `Non-Streaming Total Latency` kullanır.

### Görev 5 — Throughput ve Chunks

Bir streaming measurement seçin:

- `completion_tokens`
- `total_stream_latency_ms`
- `total_chunks`
- `visible_content_chunks`
- `total_visible_characters`

Veriler uygunsa effective completion TPS'yi hesaplayın.

**Soru:** Bir stream'de 40 chunk bulunması, 40 completion token üretildiğini kanıtlar mı?

### Görev 6 — Quality / Cost Trade-off

`condition_summary.csv` dosyasını kullanarak tabloyu doldurun:

| Metrik | Non-Streaming | Streaming |
|---|---:|---:|
| Başarılı çağrı | | |
| Complete cevap | | |
| Ortalama total latency | | |
| Token usage availability | | |
| Toplam estimated cost | | |

Şu soruları tartışın:

- Streaming görünür cevabı erken gösteriyor, fakat toplam süreyi artırıyorsa kullanılmalı mı?
- Streaming response eksik gelirse kullanıcı deneyimi nasıl etkilenir?
- Streaming ek token tasarrufu sağlamıyorsa yine de faydalı olabilir mi?
- Kullanıcı arayüzünde Streaming'in faydası hangi ürünlerde daha belirgindir?

---

## 19. Deneysel sınırlamalar

Bu laboratuvar kontrollü bir eğitim deneyidir. Aşağıdaki koşullar tam anlamıyla sabitlenemez:

1. **Provider load:** API sunucusundaki anlık yoğunluk değişebilir.
2. **Network variability:** İstemci-sunucu arası gecikme değişebilir.
3. **Sampling variability:** Aynı prompt farklı uzunlukta yanıtlar üretebilir.
4. **Reasoning overhead:** Üretilen reasoning tokenları çağrılar arasında değişebilir.
5. **Usage availability:** Streaming son usage bilgisi her istekte gelmeyebilir.
6. **Small sample size:** Pilot 2 pair, Default 5 pair, Extended 8 pair içerir.
7. **Measurement scope:** `first_visible_content_ms`, sistemin bütün frontend render süresini ölçmez; istemcide content'in alınma zamanıdır.
8. **Quality scope:** Sadece yapısal completeness kontrolü yapılır; bütün teknik ifadeler otomatik olarak doğrulanmaz.

Bu nedenle:

> Tek bir run, bütün Groq modelleri ve bütün kullanıcılar için Streaming'in her zaman hızlı veya ucuz olduğunu kanıtlamaz.

---

## 20. Troubleshooting

### 20.1 `LiveExecutionNotAllowed`

`--live` komutu gönderilmiş ancak `[execution].live_api_enabled` etkinleştirilmemiş olabilir.

Kontrol edin:

- `config/live.toml` var mı?
- `[execution]` içinde `live_api_enabled = true` mu?
- Komutta `--config config/live.toml` var mı?
- Docker image config oluşturulduktan sonra rebuild edildi mi?

### 20.2 `GROQ_API_KEY` eksik

`.env` dosyasını oluşturup anahtarı ekleyin; anahtarı paylaşmayın.

### 20.3 HTTP 429 / Rate limit

Kota veya provider rate limit sınırına ulaşılmış olabilir. Art arda tekrar deneyip fazla çağrı üretmeyin.

- Groq Console limitlerini kontrol edin.
- Pilot profile dönün.
- Canlı API isteğini geçici olarak durdurun.

### 20.4 `USAGE_UNAVAILABLE`

Streaming token usage metadata'sı dönmemiş olabilir. Sonuç sıfır değildir.

- `request_usage_when_supported = true` ayarını kontrol edin.
- Response'un final chunk'ında usage bulunup bulunmadığını inceleyin.
- Maliyet değeri yoksa uydurma rakam kullanmayın.

### 20.5 `first_visible_content_ms = N/A`

Akıştan görünür içerik gelmemiş olabilir veya stream kesilmiş olabilir.

`stream_status` ve `evaluation_status` alanlarını kontrol edin.

### 20.6 `STREAM_INTERRUPTED`

Stream başladıktan sonra bağlantı kesilmiş veya chunk akışı hata vermiş olabilir. Ölçümü tam başarı gibi değerlendirmeyin.

### 20.7 `TRUNCATED`

Cevap tamamlanmamış ve provider `finish_reason=length` döndürmüş olabilir.

`max_completion_tokens`, reasoning token bütçesi ve yanıt formatını gözden geçirin.

### 20.8 `400 Bad Request` / `reasoning_format`

**Uyumluluk notu (2026-10-29):** Repository'deki LAB06 kodu, GPT-OSS 20B çağrılarında `reasoning_format="parsed"` gönderiyor. Groq'un güncel Reasoning dokümantasyonu ise GPT-OSS 20B/120B için bu parametrenin desteklenmediğini belirtiyor.

Bu nedenle model veya API `400` döndürürse problemi yalnızca Docker'a ya da API anahtarına bağlamayın. Eğitmen kaynak kodu ve güncel provider dokümanını karşılaştırmalıdır. Gerekirse ayrı bir kod revizyonuyla desteklenmeyen parametre kaldırılmalıdır; iki condition'ın aynı, desteklenen ayarlarla çalıştığından emin olunmalıdır.

**Bu rehber kodu otomatik olarak değiştirmez.**

### 20.9 Grafikler oluşmadı

Önce `stream_measurements.json` ve `experiment_summary.json` oluşmuş mu kontrol edin. Uygunsa:

```powershell
docker compose run --rm lab-runner lab06 backfill --config config/default.toml
```

komutunu çalıştırın.

### 20.10 Run ID zaten kullanılmış

LAB06 mevcut ölçümleri bulunan aynı run ID'ye yeniden yazmaz. Yeni bir `--experiment-run-id` verin.

---

## 21. Tartışma soruları

1. Streaming ile Non-Streaming arasındaki asıl fark inference işlemi mi, response teslimatı mı?
2. İlk stream event'i neden kullanıcının gördüğü ilk kelime anlamına gelmeyebilir?
3. TTFT ile First Visible Content hangi durumda birbirinden ayrılır?
4. Reasoning tokenları kullanıcıya görünür içerikten önce üretilirse hangi gecikme türü artabilir?
5. Streaming'in total latency'si daha uzun olmasına rağmen neden tercih edilebilir?
6. Streaming neden doğrudan token-cost optimization yöntemi sayılmaz?
7. `USAGE_UNAVAILABLE` ve `0 completion tokens` neden aynı şey değildir?
8. Streaming event'i neden token sayısına eşit değildir?
9. Bir cevap `COMPLETE` olmasına rağmen teknik olarak yanlış olabilir mi?
10. Streaming hangi ürünlerde önemlidir: sohbet arayüzü, batch pipeline, belge işleme, veri sınıflandırma?
11. Network gecikmesi yükseldiğinde hangi latency metriği daha faydalı olur?
12. Kullanıcı algısı açısından ilk içeriğin görünmesi mi, tüm yanıtın bitmesi mi daha önemlidir?

---

## 22. Genel sonuç

LAB06'nın ana öğrenme çıktısı:

**Streaming, çoğunlukla yanıtın tamamını daha hızlı üretmek için değil, içeriği tamamlanmadan kullanıcıya göstermeye başlamak için kullanılır.**

İyi bir uygulama değerlendirmesinde şu metrikler birlikte ele alınmalıdır:

- First Stream Event
- First Visible Content
- Total Latency
- Perceived Latency Gain
- Token Usage Availability
- Effective Throughput
- Estimated Cost
- Answer Completeness

**Yalnızca hız metriklerini değil, kalite ve ölçüm güvenilirliğini de dikkate alın.**

---

## 23. Kaynak dosyalar

| Repository dosyası | İçeriği |
|---|---|
| `src/labs/lab06_streaming/experiments.py` | Incident fixture ve beş başlıklı prompt |
| `src/labs/lab06_streaming/matrix.py` | Pilot/Default/Extended pair planı |
| `src/labs/lab06_streaming/runner.py` | Streaming ve non-streaming çağrıları |
| `src/labs/lab06_streaming/stream_events.py` | Chunk/event sınıflandırması |
| `src/labs/lab06_streaming/models.py` | Ölçüm alanları |
| `src/labs/lab06_streaming/evaluation.py` | Answer completeness evaluator |
| `src/labs/lab06_streaming/stats.py` | Pair comparison ve formüller |
| `src/labs/lab06_streaming/persistence.py` | Sonuç dosyaları |
| `src/labs/lab06_streaming/charts.py` | Otomatik grafikler |
| `src/labs/lab06_streaming/report.py` | Analiz raporu |
| `config/default.toml` | Varsayılan LAB06 konfigürasyonu |

### Resmî Groq dokümantasyonu

- [Text Generation & Streaming](https://console.groq.com/docs/text-chat)
- [Reasoning](https://console.groq.com/docs/reasoning)
- [API Reference](https://console.groq.com/docs/api-reference)
- [Understanding and Optimizing Latency](https://console.groq.com/docs/production-readiness/optimizing-latency)

---

## 24. Repository'de önerilen konum

```text
docs/participant/lab06-streaming-and-perceived-latency.md
```

Ana `docs/participant/README.md` dosyasına eklenebilecek bağlantı:

```markdown
## LAB06 — Streaming & Perceived Latency

[Open LAB06 Participant Guide](lab06-streaming-and-perceived-latency.md)
```
