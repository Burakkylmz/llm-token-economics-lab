# LAB09 — Batch Processing & Asynchronous Workflows
## Öğrenci Laboratuvar Rehberi · Educational Batch Lab 2.0

> **Teknolojiler:** Python · Groq API · Docker Compose · `asyncio` · JSONL · pytest · Matplotlib\
> **Hedef:** Sequential, Async Concurrent ve Groq Batch API stratejilerini aynı Golden Dataset üzerinde karşılaştırmak.\
> **Kaynak sürüm:** `Burakkylmz/llm-token-economics-lab`, `master` dalındaki LAB09 2.0 kodları (29 Ekim 2026).

---

## 1. Bu laboratuvarda ne öğreneceksiniz?

Bir LLM uygulamasında sekiz ayrı metin işlenecek olsun. Aynı görevleri üç farklı yöntemle yürütebiliriz:

1. **Sequential:** İstekler art arda gönderilir; bir önceki tamamlanmadan sonraki başlamaz.
2. **Async Concurrent:** Bağımsız istekler `asyncio` ve `asyncio.Semaphore` yardımıyla eşzamanlı yürütülür.
3. **Provider-Managed Batch API:** İstekler bir JSONL dosyasıyla Groq'a teslim edilir; sağlayıcı job'ı kendi zamanlamasında işler.

Bu stratejiler yalnızca hızları bakımından farklı değildir. Kullanıcı beklentisi, job yaşam döngüsü, hata yönetimi ve fiyatlandırma da değişir.

**Temel araştırma sorusu:**

> Aynı görevleri farklı yürütme stratejileriyle çalıştırdığımızda workflow latency, request latency, throughput, cevap doğruluğu ve estimated cost nasıl değişir?

### Öğrenme çıktıları

- Sequential ve synchronous kavramlarının neden aynı olmadığını açıklayabilmek.
- Python-level concurrency ile provider-managed Batch API'yi ayırt edebilmek.
- `queue_wait_ms`, `request_service_latency_ms` ve `batch_wall_clock_ms` değerlerini yorumlamak.
- `input_file_id`, `batch_id`, `output_file_id` ve `custom_id` görevlerini anlamak.
- Batch job status'u ile request-level correctness'i ayırmak.
- Standard ve Batch fiyatlandırmasını hesaplamak.
- `BATCH_PENDING`, `SUBMISSION_UNCERTAIN`, `MISSING`, `FAILED` gibi durumları yorumlamak.
- Gerçek API olmadan synthetic fixture'larla Batch reconciliation öğrenmek.

## 2. Hipotezler

| ID | Hipotez |
|---|---|
| **H1** | Concurrent processing, Sequential'a kıyasla aynı iş yükünün wall-clock süresini düşürebilir. |
| **H2** | Concurrency artışı, her tekil request'in service latency'sini otomatik olarak azaltmaz. |
| **H3** | Batch API aynı workload için sağlayıcının belirlediği daha düşük fiyatı sunabilir. |
| **H4** | Batch, düşük real-time latency hedefini maliyet etkin deferred execution hedefiyle değiştirir. |
| **H5** | Daha ucuz veya hızlı sonuç, golden answer correctness azaldığında otomatik başarı değildir. |

**Kural:** Ölçülmüş veriyi, hesaplanmış tahmini ve provider policy bilgisini ayrı değerlendirin.

## 3. Deney mimarisi

```text
                      Golden Dataset v1
                         16 tasks
                             |
                    Profile Selection
                     pilot/default/extended
                             |
              +--------------+----------------+
              |              |                |
              v              v                v
          Sequential      Concurrent      Batch JSONL
          async, C=1      async, C=5      file generation
              |              |                |
              v              v                v
        Real-time API   Real-time API   Explicit submission
              |              |                |
              |              |          Provider Batch job
              |              |                |
              |              |          Status check
              |              |                |
              |              |          Output/Error files
              +--------------+----------------+
                             |
              Reconciliation / Evaluation
                             |
              Summary / Markdown / Charts
```

`lab09 all` iki real-time koşulu çalıştırır ve Batch input dosyası üretir. **Batch job'ı otomatik olarak göndermez.** Gerçek gönderim için `submit_batch` komutunu ayrıca çalıştırmanız gerekir.

### 3.1 Sequential koşulu

LAB09, `AsyncGroq` kullanır fakat semaphore kapasitesini `1` yapar. Dolayısıyla **Sequential, synchronous SDK kullanımı değildir**.

### 3.2 Concurrent koşulu

Aynı engine `max_concurrent_requests=5` varsayılanıyla kullanılır. Beş sayısı bir **üst sınırdır**; gözlenen gerçek peak ayrıca ölçülür.

### 3.3 Batch koşulu

Sağlayıcı, JSONL dosyasındaki istekleri kendi planına göre işler. Batch işinin completion window'u real-time request latency ölçümü değildir.

## 4. Golden Dataset

Kaynak:

```text
data/lab09/golden_batch_dataset_v1.json
```

16 sentetik iş görevi içerir. Bunlar eski sürümdeki sekiz tekrarlı aritmetik prompt'un yerini almıştır.

| Task | Kategori | Beklenen cevap |
|---|---|---|
| `task_001` | Sentiment classification | `POSITIVE` |
| `task_002` | Sentiment classification | `NEGATIVE` |
| `task_003` | Order ID extraction | `ORD-55213` |
| `task_004` | Ticket ID extraction | `TKT-9043` |
| `task_005` | Invoice arithmetic | `23.75` |
| `task_006` | Discount calculation | `60` |
| `task_007` | Gift-card request detection | `YES` |
| `task_008` | Gift-card request detection | `NO` |
| `task_009` | Email extraction | `maria.santos@example.com` |
| `task_010` | Delivery-date extraction | `2026-11-12` |
| `task_011` | Currency formatting | `1250.00` |
| `task_012` | HTTP status classification | `CLIENT_ERROR` |
| `task_013` | Priority classification | `HIGH` |
| `task_014` | Priority classification | `LOW` |
| `task_015` | Tax calculation | `216` |
| `task_016` | Phone extraction | `1-800-555-0199` |

Her task'ta `task_id`, `category`, `prompt`, `expected_answer`, `evaluation_type`, `metadata` alanları bulunur. Golden cevaplar modele gönderilmez; evaluator'da kullanılır.

### 4.1 Evaluator

- `exact_match`: Yanıtın bütünü, whitespace ve case normalizasyonu sonrası expected answer ile eşleşmeli.
- `numeric_match`: Bütün yanıt tek sayıya dönüştürülebilmeli ve küçük bir toleransla beklenen değere eşit olmalı. Baştaki `$` ve binlik ayırıcılar temizlenebilir.

**Substring matching yoktur.** `"The result is POSITIVE"` çıktısı `POSITIVE` için exact match değildir.

## 5. Pilot, Default ve Extended

| Profile | Task | Sequential calls | Concurrent calls | Real-time toplam | Batch input requests |
|---|---:|---:|---:|---:|---:|
| `pilot` | 4 | 4 | 4 | **8** | 4 |
| `default` | 8 | 8 | 8 | **16** | 8 |
| `extended` | 16 | 16 | 16 | **32** | 16 |

\[
N_{\mathrm{realtime}} = 2\times N_{\mathrm{task}}
\]

**Batch dosyasındaki 8 request, 8 ek anlık chat completion çağrısı değildir.** Provider-managed bir Batch job'ın içindeki kayıt sayısıdır.

## 6. Kontrol değişkenleri ve deneysel dikkat

Karşılaştırma boyunca `model=openai/gpt-oss-20b`, Golden Dataset v1 ve aynı prompt/expected answer seti kullanılır. Config'te `reasoning_effort=low`, `max_completion_tokens=128` bulunur.

**Kaynak kod sınırlaması:** Real-time çağrılar `reasoning_effort` ve `max_completion_tokens` parametrelerini açıkça gönderir; `batch.py` ile üretilen JSONL ise `body` altında yalnızca `model` ve `messages` içerir. Bu nedenle Batch generation parametreleri real-time ile **tam olarak eşitlenmiş sayılmamalıdır**. Kalite/maliyet sonuçlarında bu farkı belirtin.

Modelin rastlantısallığı, provider load ve network değişkenliği tamamen kontrol edilemez. Küçük örneklem bir production benchmark'ı değildir.

---

## 7. Kurulum — Windows PowerShell + Docker

### 7.1 Gereksinimler

Docker Desktop, Docker Compose, güncel repository ve canlı işlemler için Groq API anahtarı.

Proje köküne geçin:

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Kendi klasörünüz farklıysa yolu değiştirin.

### 7.2 `.env` dosyası

```powershell
Copy-Item .env.example .env
```

`.env` içine:

```dotenv
GROQ_API_KEY=your_api_key_here
```

yazın. **Anahtarınızı GitHub'a commit etmeyin veya paylaşmayın.**

### 7.3 Güvenli canlı konfigürasyon

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` içindeki mevcut bölümde:

```toml
[execution]
live_api_enabled = true
```

değerini etkinleştirin. Diğer TOML bölümlerini silmeyin.

Canlı API için **hem `--live` hem de `execution.live_api_enabled=true`** gereklidir.

### 7.4 Docker build

```powershell
docker compose build lab-runner
```

Dockerfile `config/` klasörünü image'a kopyaladığından `live.toml` oluşturduktan sonra build edin.

### 7.5 Help

```powershell
docker compose run --rm lab-runner lab09 --help
```

API isteği yapılmaz.

---

## 8. Dry-Run — API çağrısı yok

**Default experiment plan:**

```powershell
docker compose run --rm lab-runner lab09 all --profile default --dry-run --config config/default.toml
```

Bu işlem:

- Golden Dataset yükler ve doğrular.
- İki real-time condition için deney planı üretir.
- `experiment_plan.csv` ve `golden_workload_snapshot.json` kaydeder.
- Planlanan real-time call sayısını ve Batch input request sayısını gösterir.
- Groq client oluşturmadan sonlanır.

Pilot:

```powershell
docker compose run --rm lab-runner lab09 all --profile pilot --dry-run --config config/default.toml
```

Extended:

```powershell
docker compose run --rm lab-runner lab09 all --profile extended --dry-run --config config/default.toml
```

İsimlendirilmiş run:

```powershell
docker compose run --rm lab-runner lab09 all --profile default --dry-run --experiment-run-id lab09-class-demo --config config/default.toml
```

Çıktılar `results/lab09_batch_processing/lab09-class-demo/` altında oluşur.

**Preflight cost uyarısı:** CLI'de `Conservative Preflight Cost Ceiling` olarak yazdırılan hesap, `planned_calls × max_completion_tokens × output_price` bileşenini kullanır. Input maliyetini içermediği için gerçek maliyetin garantili üst sınırı değildir.

---

## 9. Grand Test — Sequential + Concurrent

**Önerilen Default Live:**

```powershell
docker compose run --rm lab-runner lab09 all --profile default --experiment-run-id lab09-class-demo --live --config config/live.toml
```

Planlanan: **16 real-time Groq çağrısı**, ayrıca **8 input request içeren Batch JSONL**.

Çalışma sırası:

1. Sequential sekiz task.
2. Concurrent aynı sekiz task.
3. Per-call token, cost, latency, status ve correctness ölçümleri.
4. `request_timeline.csv`.
5. `batch_input.jsonl` (otomatik gönderim **yok**).
6. `batch_job.json` → `NOT_SUBMITTED`.
7. Summary, `analysis_report.md` ve uygun PNG grafikler.

Pilot:

```powershell
docker compose run --rm lab-runner lab09 all --profile pilot --experiment-run-id lab09-pilot-demo --live --config config/live.toml
```

Extended:

```powershell
docker compose run --rm lab-runner lab09 all --profile extended --experiment-run-id lab09-extended-demo --live --config config/live.toml
```

### Tek koşul çalıştırma

Yalnız Sequential:

```powershell
docker compose run --rm lab-runner lab09 sequential --profile default --experiment-run-id lab09-seq-only --live --config config/live.toml
```

Yalnız Concurrent:

```powershell
docker compose run --rm lab-runner lab09 concurrent --profile default --experiment-run-id lab09-conc-only --live --config config/live.toml
```

Concurrency seviyesini değiştirme:

```powershell
docker compose run --rm lab-runner lab09 concurrent --profile default --max-concurrent 3 --experiment-run-id lab09-conc-3 --live --config config/live.toml
```

**Önemli:** Tek koşullu çalıştırmalar `all` gibi `batch_input.jsonl` oluşturmaz. Tam Batch lifecycle için `all` koşusundaki Run ID'yi kullanın.


---

## 10. Batch Lifecycle — Öğrenciler için adım adım

Batch job'ın tamamlanması gerçek zamanlı koşullardan ayrıdır. Bütün işlemlerde **aynı Run ID** kullanılacak:

```text
lab09-class-demo
```

Bu ID'yi kendi run adınızla değiştiriyorsanız aşağıdaki bütün komutlarda da değiştirin.

### 10.1 JSONL'yi inceleyin

`lab09 all --live` sonrası dosya:

```text
results/lab09_batch_processing/lab09-class-demo/batch_input.jsonl
```

İlk iki satır:

```powershell
Get-Content results/lab09_batch_processing/lab09-class-demo/batch_input.jsonl -TotalCount 2
```

Satır sayısını doğrulayın:

```powershell
(Get-Content results/lab09_batch_processing/lab09-class-demo/batch_input.jsonl | Measure-Object -Line).Lines
```

Default profilde **8 satır** bulunmalıdır.

Bir satırın yapısı:

```json
{"custom_id":"task_001","method":"POST","url":"/v1/chat/completions","body":{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"..."}]}}
```

`...` burada temsilî kısaltmadır; gerçek JSONL'de Golden Dataset prompt'u yer alır.

| Alan | Anlamı |
|---|---|
| `custom_id` | Golden `task_id`, örn. `task_001` |
| `method` | `POST` |
| `url` | `/v1/chat/completions` |
| `body.model` | Groq model ID |
| `body.messages` | Request'in mesajları |

**Önemli:** Provider Batch sonuçlarının sıralaması input sırasından farklı olabilir. Eşleştirmede `custom_id` kullanılır.

### 10.2 Batch job'ı gönderin

**Bu komut gerçek Groq Files ve Batch API işlemi gerçekleştirir:**

```powershell
docker compose run --rm lab-runner lab09 submit_batch --experiment-run-id lab09-class-demo --live --config config/live.toml
```

Yapılan işler:

1. JSONL dosyasını yükler.
2. `input_file_id` alır.
3. Batch job oluşturur.
4. `batch_id` alır.
5. Yerel `batch_job.json` dosyasına bunları kaydeder.
6. Job'ın tamamlanmasını **beklemeden** sonlanır.

Terminalin örnek biçimi:

```text
Submitted Batch job: <BATCH_ID> (input_file_id=<INPUT_FILE_ID>)
This does NOT wait for completion...
```

Bu kimliklerin gerçek değerlerini Groq sağlar.

### 10.3 Durumu sorgulayın

```powershell
docker compose run --rm lab-runner lab09 check_batch_status --experiment-run-id lab09-class-demo --live --config config/live.toml
```

Bu komut kaydedilmiş Batch ID'yi kullanır; tekrar elle girmeniz gerekmez. **Tek bir status sorgusu** yapar; sürekli polling yoktur.

| Provider status | Anlamı |
|---|---|
| `validating` | Batch input doğrulanıyor |
| `in_progress` | Sağlayıcı job üzerinde çalışıyor |
| `finalizing` | Sonuçlar hazırlanıyor |
| `completed` | Batch job tamamlandı |
| `failed` | Job başarısız |
| `expired` | İşleme süresi sınırı aşıldı |
| `cancelling` | İptal devam ediyor |
| `cancelled` | Job iptal edildi |

Terminalde provider status yanında request sayıları ve mevcutsa `output_file_id` ile `error_file_id` de raporlanır.

### 10.4 Sonuçları indirin

Job'ın output veya error dosyası hazır olduğunda:

```powershell
docker compose run --rm lab-runner lab09 fetch_batch_results --experiment-run-id lab09-class-demo --live --config config/live.toml
```

Komut:

- Output/error file ID'lerini sorgular.
- Varsa ilgili dosyaları indirir.
- `batch_output.jsonl` ve `batch_errors.jsonl` saklar.
- `custom_id` üzerinden reconciliation yapar.
- Correctness evaluator çalıştırır.
- Batch ve standard cost değerlerini hesaplar.
- `analysis_report.md` ve ilgili sonuç dosyalarını yeniler.
- Uygun grafiklerin üretimini dener.

Output/error dosyası henüz yoksa:

```text
BATCH_PENDING -- nothing to fetch.
```

görmek normaldir.

### 10.5 Batch job hangi ID'leri kullanır?

| ID | Ne işe yarar? |
|---|---|
| `custom_id` | Batch içindeki tek isteği Golden Task ile eşleştirir |
| `input_file_id` | Groq'a yüklenen JSONL dosyasının kimliği |
| `batch_id` | Provider-managed Batch job'ın kimliği |
| `output_file_id` | Job output dosyasının kimliği |
| `error_file_id` | Başarısız isteklerin bulunduğu dosyanın kimliği, varsa |

Bu kimlikler `batch_job.json` ile kalıcı tutulur (tekil `custom_id` değerleri ise JSONL'de bulunur).

### 10.6 Local lifecycle state

Provider status dışında LAB09'un kendi durumları da vardır:

```text
NOT_SUBMITTED
SUBMISSION_IN_PROGRESS
SUBMISSION_UNCERTAIN
SUBMITTED
RESULTS_AVAILABLE
RESULTS_IMPORTED
EVALUATED
```

Bunlar Groq tarafından dönen literal status'lar değildir; LAB09'un local metadata durumlarıdır.

**`completed` provider job status'u, bütün cevapların doğru olduğu anlamına gelmez.**

### 10.7 `SUBMISSION_UNCERTAIN`

Upload/job creation sırasında bağlantı hatası oluşursa isteğin provider'a ulaşıp ulaşmadığı belirsiz kalabilir.

Bu durumda LAB09 `SUBMISSION_UNCERTAIN` kaydeder.

**Tekrar göndermeden önce Groq Console üzerinden kontrol edin.** Mevcut CLI bu duruma uyarı vererek yeniden submission'a izin verebilir; otomatik olarak idempotent/exactly-once davranışı garanti edilmez.

---

## 11. Batch Reconciliation — Her sonuç nereye ait?

Output JSONL farklı sırada dönebilir:

```text
Input order:
task_001
task_002
task_003

Output order:
task_003
task_001
task_002
```

Bu nedenle **sıra değil `custom_id`** esas alınır.

### 11.1 Reconciliation status'ları

| Status | Anlamı |
|---|---|
| `COMPLETED` | Beklenen task için error olmayan response kaydı var |
| `FAILED` | Provider request düzeyinde hata bildirdi |
| `MISSING` | Planlanan task için output/error kaydı yok |
| `UNKNOWN_ID` | Plan dışı bir `custom_id` görüldü |
| `MALFORMED` | Parse edilemeyen veya ID'siz satır var |
| `DUPLICATE` | Aynı `custom_id` birden fazla kez dönmüş olabilir |

**Teknik ayrıntı:** Duplicate ID'ler `duplicate_result_ids` sayacıyla izlenir; bu sayaç her zaman ayrı bir `DUPLICATE` measurement satırına karşılık gelmez.

### 11.2 Request outcome ve answer quality ayrı

Temsili örnek:

```text
Planned requests:   8
Completed rows:     5
Failed requests:    1
Missing requests:   2

Correct answers:    4
Incorrect answers:  1
```

Beş tamamlanmış model yanıtının dördü doğru olabilir. Ancak toplam sekiz planlanmış görevin sadece dördü doğru tamamlanmıştır.

Bu yüzden bir sonuçta iki farklı payda olabilir:

\[
\mathrm{EvaluatedAccuracy}=\frac{\mathrm{Correct}}{\mathrm{CompletedResponses}}
\]

\[
\mathrm{EndToEndCorrectness}=\frac{\mathrm{Correct}}{\mathrm{PlannedRequests}}
\]

Bu iki metriği raporda aynı isimle kullanmayın.

### 11.3 Evaluator status'ları

| Evaluation | Anlamı |
|---|---|
| `CORRECT` | Tam değer eşleşiyor |
| `INCORRECT` | Cevap mevcut ama yanlış |
| `INVALID_FORMAT` | Çıktı beklenen numeric formatta değil |
| `TRUNCATED` | Bazı geçersiz çıktı durumlarında finish reason `length` |
| `MISSING_RESPONSE` | Görünür cevap yok |
| `EVALUATION_UNAVAILABLE` | Değerlendirme yapılamıyor |

Örnekler:

```text
Expected: POSITIVE
Response: positive
=> CORRECT
```

```text
Expected: POSITIVE
Response: The sentiment is POSITIVE
=> INCORRECT (whole-value mismatch)
```

```text
Expected: 1250.00
Response: $1,250.00
=> CORRECT (numeric_match)
```

**API success ≠ answer correctness.** Bir request başarıyla dönse bile format veya içerik yanlış olabilir.

---

## 12. Token Economics — Ne ölçülüyor?

Real-time başarılı çağrılarda:

| Alan | Anlamı |
|---|---|
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Toplam completion tokens |
| `reasoning_tokens` | Raporlandıysa reasoning tokens |
| `total_tokens` | Provider total tokens |
| `cached_tokens` | Raporlandıysa cached input |
| `estimated_standard_cost_usd` | Konfigüre model fiyatlarına göre estimated cost |

**Reasoning tokens completion içine dahildir; iki kere toplamayın.**

Örnek:

```text
Prompt:      100
Completion:   80
Reasoning:    60
Total:       180
```

Burada toplam `100 + 80 = 180` tokendır.

Batch parser tarafında `prompt_tokens`, `completion_tokens`, `total_tokens` çıkarılır; tüm real-time usage detail alanları Batch measurement'a aynen aktarılmaz.

Eksik değer = `N/A` veya `None`, **sıfır değil**.

## 13. Batch API maliyeti ve %50 fiyat indirimi

Güncel LAB09 fiyatlandırma implementasyonu, 29 Ekim 2026'da doğrulanmış Groq Batch API politikasını esas alıyor.

Kaynak: [Groq Batch API](https://console.groq.com/docs/batch)

### 13.1 Standard Pricing

Fiyatlar USD / milyon token olarak veriliyorsa:

\[
C_{\mathrm{standard}}=
\frac{T_{\mathrm{uncached}}P_{\mathrm{input}}+
T_{\mathrm{cached}}P_{\mathrm{cached}}+
T_{\mathrm{completion}}P_{\mathrm{output}}}{1{,}000{,}000}
\]

Real-time request cached usage raporladığında caching fiyatı kullanılabilir.

### 13.2 Batch Pricing

LAB09, Batch input'una ayrı bir cache indirimi uygulamadan temel model fiyatının yarısını kullanır:

\[
C_{\mathrm{batch}}=
\frac{0.5\left(T_{\mathrm{prompt}}P_{\mathrm{input}}+
T_{\mathrm{completion}}P_{\mathrm{output}}\right)}{1{,}000{,}000}
\]

Batch indirimi ve prompt caching indirimi **üst üste uygulanmaz**.

### 13.3 Savings

\[
\mathrm{Savings\%}=
\frac{C_{\mathrm{standard}}-C_{\mathrm{batch}}}
{C_{\mathrm{standard}}}\times 100
\]

Aşağıdaki rakamlar **temsili**:

```text
Standard-price estimate: $0.0040
Batch-price estimate:    $0.0020
Estimated savings:       50%
```

Bu fiyatlandırma bilgisi **gerçek fatura garantisi değildir**. Eksik usage verisi, güncel provider fiyatları ve hesap politikası ayrıca değerlendirilmelidir.

**Önemli:** `cost_comparison.csv` Batch yanıtlarının usage verisinden hesaplanan standard vs Batch policy karşılaştırmasıdır. Real-time Sequential condition'ın gerçekleşen toplam maliyetiyle birebir aynı token tabanı olmak zorunda değildir.

### 13.4 `cost_comparison.csv`

```text
estimated_standard_cost_usd
estimated_batch_cost_usd
estimated_batch_discount_usd
estimated_savings_percent
pricing_source
pricing_verified_date
pricing_policy_version
```

Batch sonuçları gelmeden bu alanlarda ölçülmüş tasarruf olduğunu iddia etmeyin.

---

## 14. Latency ve Throughput

### 14.1 Real-time metrikleri

| Metrik | Açıklama |
|---|---|
| `queue_wait_ms` | Task'ın semaphore izni beklediği süre |
| `request_service_latency_ms` | İstemci tarafında ölçülmüş API çağrı süresi |
| `task_end_to_end_latency_ms` | Queue'dan request tamamlanmasına kadar süre |
| `batch_wall_clock_ms` | Condition'ın bütün görevlerini tamamlama süresi |
| `successful_requests_per_second` | Successful requests / condition süresi |
| `observed_peak_in_flight_requests` | Gerçekte aynı anda yürüyen request sayısının en yüksek değeri |

Sequential condition yine async engine ile çalıştığı için queue wait **sıfıra sabitlenmez**.

### 14.2 Successful RPS

\[
\mathrm{RPS}_{\mathrm{successful}}=
\frac{\mathrm{SuccessfulRequests}}{\mathrm{BatchWallClockMs}/1000}
\]

Temsili:

```text
8 requests completed successfully
4,000 ms wall-clock

RPS = 8 / 4 = 2.0
```

### 14.3 Throughput Speedup

\[
\mathrm{Speedup}=
\frac{\mathrm{ConcurrentSuccessfulRPS}}
{\mathrm{SequentialSuccessfulRPS}}
\]

Temsili:

```text
Sequential: 1.2 requests/s
Concurrent: 3.0 requests/s

Speedup = 2.5x
```

### 14.4 Batch Time Reduction

\[
\mathrm{Reduction\%}=
\frac{T_{\mathrm{sequential}}-T_{\mathrm{concurrent}}}
{T_{\mathrm{sequential}}}\times100
\]

### 14.5 Service Latency ile Workflow Latency farkı

Concurrent koşulda **tekil request daha uzun sürmesine rağmen** tüm workflow daha erken bitebilir.

İki sonucu ayrı inceleyin:

- `batch_wall_clock_ms`: Tüm iş yükü ne kadar sürdü?
- Median `request_service_latency_ms`: Tekil API isteğinin tipik servis süresi ne?

### 14.6 Batch turnaround

Provider-managed Batch için `created_at`, `in_progress_at`, `completed_at` gibi timestamp alanları kaydedilebilir.

**Batch job turnaround, real-time request latency değildir.** Provider timestamp'leri eksikse uydurma süre oluşturmayın.

---

## 15. Config ve safety guards

`config/default.toml` içindeki LAB09 ayarları:

| Config key | Default |
|---|---|
| `lab09.enabled` | `true` |
| `lab09.model` | `openai/gpt-oss-20b` |
| `lab09.default_profile` | `default` |
| `lab09.dataset_version` | `v1` |
| `lab09.reasoning_effort` | `low` |
| `lab09.max_completion_tokens` | `128` |
| `lab09.max_concurrent_requests` | `5` |
| `lab09.sdk_max_retries` | `0` |
| `lab09.batch_completion_window` | `24h` |
| `lab09.max_api_calls` | `32` |
| `lab09.max_estimated_cost_usd` | `0.05` |
| `lab09.max_batch_input_requests` | `16` |

**`sdk_max_retries=0`:** Lab09, kendi Groq client'ında SDK automatic retry'ı kapatır. Böylece gizli retry denemelerinin latency ölçümünü etkilemesi azaltılır.

**İki farklı request limiti:** `max_api_calls=32` gerçek zamanlı planlanan çağrı sayısını; `max_batch_input_requests=16` Batch dosyasının request sayısını kontrol eder.

**Cost guard uyarısı:** Config'de `$0.05` maliyet sınırı tanımlı olsa da güncel `_run_realtime` kodunda bu değerin kesin, her çağrı öncesi bir spending gate olarak uygulandığı görünmüyor. Gerçek API maliyeti için yalnızca bu sınırın harcamayı garanti ettiğini varsaymayın.


---

## 16. LAB09 sonuç dosyaları

Her çalıştırma için ayrı bir dizin kullanılır:

```text
results/
└── lab09_batch_processing/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── golden_workload_snapshot.json
        ├── batch_input.jsonl
        ├── batch_job.json
        ├── sequential_measurements.json
        ├── sequential_measurements.csv
        ├── concurrent_measurements.json
        ├── concurrent_measurements.csv
        ├── request_timeline.csv
        ├── batch_output.jsonl
        ├── batch_errors.jsonl
        ├── batch_measurements.json
        ├── batch_measurements.csv
        ├── condition_summary.csv
        ├── quality_comparison.csv
        ├── cost_comparison.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

**Dosyaların tamamı hemen oluşmaz.** Örneğin Batch henüz gönderilmediyse `batch_output.jsonl` ve `batch_measurements.csv` bulunmaması beklenir.

| Dosya | Öğrenci neyi incelemeli? |
|---|---|
| `experiment_manifest.json` | Model, profil, experiment configuration ve live ölçüm süreleri |
| `experiment_plan.csv` | Real-time task/condition planı |
| `golden_workload_snapshot.json` | O run'da kullanılan görevler |
| `batch_input.jsonl` | Gönderilecek Batch request sözleşmesi |
| `batch_job.json` | Provider kimlikleri ve job durumu |
| `sequential_measurements.*` | Sequential request metrikleri |
| `concurrent_measurements.*` | Concurrent request metrikleri |
| `request_timeline.csv` | Queue, permit, start ve complete olayları |
| `batch_output.jsonl` | Ham provider output kayıtları |
| `batch_errors.jsonl` | Ham provider error kayıtları |
| `batch_measurements.*` | Batch reconciliation ve correctness |
| `condition_summary.csv` | Real-time condition özetleri |
| `quality_comparison.csv` | Evaluated answer accuracy karşılaştırması |
| `cost_comparison.csv` | Standard vs Batch estimated fiyat politikası |
| `experiment_summary.json` | Genel sonuçlar; eksik koşullar dahil |
| `analysis_report.md` | Hipotezler, ölçümler, sınırlamalar, teknik yorumlar |
| `charts/` | Otomatik PNG grafikler |

**Okuma sırası:** `analysis_report.md` → `experiment_summary.json` → `condition_summary.csv` → `quality_comparison.csv` → varsa Batch measurements ve cost comparison.

`batch_output.jsonl` ham model yanıtlarını içerebilir. Gerçek iş verileriyle kullanımda gizlilik/retention politikasını ayrıca değerlendirin.

---

## 17. Sekiz otomatik grafik

LAB09 2.0 şu grafik dosyalarını üretir:

| Grafik | Sorusu |
|---|---|
| `sequential_vs_concurrent_batch_time.png` | Hangi real-time koşul workflow'u daha erken bitirdi? |
| `throughput_comparison.png` | Successful RPS nasıl değişti? |
| `request_latency_distribution.png` | Request latency dağılımı nasıl? |
| `token_consumption_breakdown.png` | Prompt ve completion tokenları nasıl dağıldı? |
| `standard_vs_batch_cost.png` | Estimated standard ve Batch cost farkı ne? |
| `batch_request_status.png` | Completed, failed ve missing dağılımı ne? |
| `quality_comparison.png` | Hangi koşulda kaç cevap doğru? |
| `batch_processing_dashboard.png` | Performans, kalite ve maliyet birlikte ne söylüyor? |

Eksik Batch sonuçlarını zero cost/zero latency gibi yorumlamayın. Grafiklerin mevcut olması, Batch'in kesin tamamlandığı anlamına gelmez.

---

## 18. Offline Evaluate — Kaydedilmiş verileri yeniden değerlendirme

```powershell
docker compose run --rm lab-runner lab09 evaluate --experiment-run-id lab09-class-demo --config config/default.toml
```

Bu komut:

- Kaydedilmiş ölçümleri okur.
- Quality ve cost karşılaştırmalarını tekrar hesaplar.
- `analysis_report.md` ve özetleri günceller.
- Uygun grafikleri yeniden üretir.
- **Groq API çağrısı yapmaz.**

Batch job henüz sonuçlanmamışsa `BATCH_PENDING` beklenen bir değerlendirmedir.

**Önemli:** `evaluate`, provider'dan yeni dosya indirmez. Batch sonucu ilk kez almak için `fetch_batch_results --live` gerekir.

## 19. Backfill — Yeni API çağrısı olmadan PNG üretme

Tek bir run için:

```powershell
docker compose run --rm lab-runner lab09 backfill --run-id lab09-class-demo --config config/default.toml
```

Tüm LAB09 run klasörleri için:

```powershell
docker compose run --rm lab-runner lab09 backfill --config config/default.toml
```

Bu komutlar yalnızca kayıtlı summary verilerinden grafik üretir; API key gerektirmez ve raw JSON/CSV/Markdown ölçümlerini yeniden yazmaz.

**CLI farkı:** `evaluate --experiment-run-id` kullanır; `backfill --run-id` kullanır.

---

## 20. Tamamen Offline Eğitim Modu — API key gerekmez

Groq Batch API erişimi her hesapta bulunmayabilir. Ayrıca gerçek job bir ders oturumunda tamamlanmayabilir. Bu nedenle sentetik fixture'lar vardır:

```text
tests/fixtures/lab09/
├── sample_batch_output.jsonl
├── sample_batch_errors.jsonl
├── partial_batch_output.jsonl
├── malformed_batch_output.jsonl
└── sample_batch_job.json
```

Bu dosyalar **SYNTHETIC FIXTURE — NOT LIVE GROQ TELEMETRY** olarak değerlendirilmelidir.

### 20.1 Fixture senaryoları

| Fixture | Öğrenme amacı |
|---|---|
| `sample_batch_output.jsonl` | Pilot dört task için örnek completed response |
| `sample_batch_errors.jsonl` | Batch error file parsing |
| `partial_batch_output.jsonl` | 4 planlı task'ın yalnızca 2'si döndüğünde 2 `MISSING` |
| `malformed_batch_output.jsonl` | Duplicate ID, unknown ID ve invalid JSON |
| `sample_batch_job.json` | Sentetik tamamlanmış job metadata |

### 20.2 Docker ile fixture testini çalıştırma

```powershell
docker compose run --rm --entrypoint uv lab-runner run pytest -q tests/labs/test_lab09_offline_fixtures.py
```

Bu test:

1. Fixture output/error JSONL dosyalarını okur.
2. Task ID eşleştirmesi yapar.
3. Eksik/duplicate/unknown/malformed kayıtları tespit eder.
4. Deterministic accuracy uygular.
5. Batch fiyatlandırmasını hesaplar.
6. Summary ve analysis report üretim akışını dener.

**Önemli teknik ayrıntı:** Sadece sentetik Batch ölçümü olup Sequential/Concurrent ölçümü bulunmayan fixture testinde grafik üreticisi boş liste döndürmeyi bekler. Yani “offline fixture testinden sekiz grafik çıkmalı” beklentisi doğru değildir.

### 20.3 Daha geniş pytest kontrolü

```powershell
docker compose run --rm --entrypoint uv lab-runner run pytest -q tests/labs
```

Bu komut LAB09 ile birlikte diğer laboratuvarların offline testlerini de çalıştırabilir.

### 20.4 Offline araçlar arasındaki fark

| Araç | Gerçek API? | Ne yapıyor? |
|---|---|---|
| `all --dry-run` | Hayır | Plan ve Golden Workload snapshot |
| `test_lab09_offline_fixtures.py` | Hayır | Sentetik Batch reconciliation/evaluation |
| `evaluate` | Hayır | Kaydedilmiş run ölçümlerini tekrar yorumlar |
| `backfill` | Hayır | Kaydedilmiş summary üzerinden grafikleri üretir |
| `submit_batch --live` | Evet | Gerçek Batch job gönderir |
| `check_batch_status --live` | Evet | Provider durumunu sorgular |
| `fetch_batch_results --live` | Evet | Gerçek output/error dosyalarını indirir |

Mevcut LAB09'da sentetik fixture'ları doğrudan canlı job yerine kullanan bağımsız bir `lab09 demo` CLI komutu yoktur. Offline deneyin resmi yolu `pytest` fixture testleridir.

---

## 21. Öğrencinin dolduracağı ölçüm tabloları

### 21.1 Real-time performance

| Metrik | Sequential | Concurrent |
|---|---:|---:|
| Successful Requests | | |
| Failed Requests | | |
| Observed Peak In-Flight | | |
| Batch Wall-Clock (ms) | | |
| Successful Requests/sec | | |
| Median Queue Wait (ms) | | |
| Median Service Latency (ms) | | |
| Median End-to-End Latency (ms) | | |
| Estimated Standard Cost (USD) | | |

### 21.2 Quality

| Metrik | Sequential | Concurrent | Batch |
|---|---:|---:|---:|
| Evaluated Responses | | | |
| Correct Responses | | | |
| Accuracy | | | |

Batch sonuçları yoksa ilgili hücreye **BATCH_PENDING** yazın; `0` yazmayın.

### 21.3 Batch reconciliation

| Metrik | Değer |
|---|---:|
| Planned Batch Requests | |
| Completed Requests | |
| Failed Requests | |
| Missing Requests | |
| Duplicate Result IDs | |
| Unexpected Result IDs | |
| Correct Answers | |
| Incorrect Answers | |

### 21.4 Batch cost

| Metrik | Değer |
|---|---:|
| Estimated Standard Cost | |
| Estimated Batch Cost | |
| Estimated Discount | |
| Estimated Savings (%) | |
| Pricing Policy Version | |

---

## 22. Sınıf içi uygulamalar

### Görev 1 — Experiment Matrix

Default dry-run yapın:

```powershell
docker compose run --rm lab-runner lab09 all --profile default --dry-run --config config/default.toml
```

`experiment_plan.csv` dosyasını açın. Neden 16 real-time call ve 8 Batch input request planlandığını açıklayın.

### Görev 2 — Pilot Live

```powershell
docker compose run --rm lab-runner lab09 all --profile pilot --experiment-run-id lab09-pilot-study --live --config config/live.toml
```

`condition_summary.csv` dosyasından batch wall-clock, successful RPS ve median service latency değerlerini karşılaştırın. **H1 ve H2 için gözleme dayalı yorum** yazın.

### Görev 3 — Request Timeline

`request_timeline.csv` içindeki olaylar:

```text
QUEUED
PERMIT_ACQUIRED
REQUEST_STARTED
REQUEST_COMPLETED
```

Bir task seçip:

\[
T_{\mathrm{queue}}=T_{\mathrm{permit}}-T_{\mathrm{queued}}
\]

\[
T_{\mathrm{e2e}}=T_{\mathrm{completed}}-T_{\mathrm{queued}}
\]

formüllerini elle uygulayın.

### Görev 4 — Answer Correctness

`quality_comparison.csv` dosyasında `correct_count`, `total_evaluated` ve `accuracy` alanlarını bulun.

**Soru:** API çağrılarının tamamı başarılı olsa da evaluator neden yanlış cevap gösterebilir?

### Görev 5 — JSONL Request Contract

`batch_input.jsonl` açın. Her satırın:

```text
custom_id
method
url
body.model
body.messages
```

alanlarını doğrulayın.

**Soru:** Batch result row sırası değişirse nasıl doğru task ile eşleştirilir?

### Görev 6 — Batch Lifecycle

Hesabınızın Batch API erişimi varsa:

```powershell
docker compose run --rm lab-runner lab09 submit_batch --experiment-run-id lab09-pilot-study --live --config config/live.toml
```

ve daha sonra:

```powershell
docker compose run --rm lab-runner lab09 check_batch_status --experiment-run-id lab09-pilot-study --live --config config/live.toml
```

kullanın.

**Soru:** `in_progress` durumundaki Batch'in neden henüz final output dosyası olmayabilir?

Batch API erişimi olmayan öğrenciler offline fixture uygulamasına geçebilir.

### Görev 7 — Partial Results

```powershell
docker compose run --rm --entrypoint uv lab-runner run pytest -q tests/labs/test_lab09_offline_fixtures.py
```

`partial_batch_output.jsonl` dosyasını okuyun ve 4 görevden neden 2'sinin `MISSING` olduğunu açıklayın.

### Görev 8 — Malformed Output

`malformed_batch_output.jsonl` içinde Duplicate, Unknown ve malformed kayıtları bulun.

**Soru:** Bunların analizden sessizce atılması deney sonucunu nasıl bozar?

### Görev 9 — Cost Policy

Batch sonuçları varsa `cost_comparison.csv` açın.

\[
\mathrm{Savings\%}=
100\times\frac{C_{\mathrm{standard}}-C_{\mathrm{batch}}}
{C_{\mathrm{standard}}}
\]

hesabını elle tekrar yapın. Bu rakamı **ölçülmüş provider faturası** olarak sunmayın.

### Görev 10 — Mimari Karar

| Use Case | Tercih edilen yaklaşım | Teknik gerekçe |
|---|---|---|
| Etkileşimli chatbot | | |
| Aynı anda gelen bağımsız gerçek zamanlı istekler | | |
| Gece işlenen 50.000 müşteri yorumu | | |
| Anlık ödeme riski değerlendirmesi | | |
| Haftalık offline belge sınıflandırma | | |
| Büyük ölçekli deferred data enrichment | | |

Seçiminizi SLA, maliyet, veri gizliliği, rate limit ve hata işleme açısından açıklayın.

---

## 23. Tartışma soruları

1. Sequential ve synchronous aynı şey mi?
2. `asyncio.Semaphore` neyi sınırlar?
3. Provider Batch API ile Python `asyncio.gather` arasındaki fark nedir?
4. Batch input request sayısı neden real-time call sayısına eklenmemelidir?
5. Bir job'ın `completed` olması, bütün task'ların `CORRECT` olduğu anlamına gelir mi?
6. `custom_id` neden önemlidir?
7. `SUBMISSION_UNCERTAIN` durumunda otomatik yeniden gönderme neden tehlikeli olabilir?
8. Eksik token usage neden sıfır olarak kabul edilmemelidir?
9. Batch indirimini cached input indirimiyle üst üste uygulamak doğru mu?
10. Batch job turnaround neden tekil request latency değildir?
11. Real-time latency artarken toplam workflow süresi kısalabilir mi?
12. `BATCH_PENDING` bir hata mı, yoksa raporun eksik kalmasının dürüst açıklaması mı?
13. Batch JSONL generation parametreleri explicit değilse kalite karşılaştırması nasıl etkilenebilir?
14. Test fixture'ından gelen maliyet veya zaman rakamları gerçek Groq ölçümü müdür?

---

## 24. Troubleshooting

### `LiveExecutionNotAllowed`

`--live`, `config/live.toml` ve `execution.live_api_enabled=true` koşullarını kontrol edin. Docker image'ı config değişikliklerinden sonra tekrar build edin.

### `GROQ_API_KEY` bulunamadı

`.env` dosyanızı ve Compose ortamını kontrol edin. Dry-run ve fixture testleri API key gerektirmez.

### HTTP 429 / `RATE_LIMITED`

Groq hesabınızın limitleri dolmuş olabilir. Pilot profile veya daha düşük `--max-concurrent` değerine dönün.

### `403 Permission Denied: Not available for your plan`

Batch API hesabınız için etkin değil. Offline fixture testini kullanın; aynı job'ı tekrar tekrar göndermeyin.

### `No batch input file`

Önce **aynı Run ID ile `lab09 all --live`** çalıştırılmış olmalıdır. `sequential`/`concurrent` tek başına Batch input dosyası üretmez.

### `--experiment-run-id is required`

`submit_batch`, `check_batch_status`, `fetch_batch_results` ve `evaluate` için mevcut Run ID gereklidir.

### `No submitted Batch job found`

İlgili run'da kaydedilmiş `batch_id` yoktur. `batch_job.json` dosyasını inceleyin.

### `BATCH_PENDING -- nothing to fetch`

Henüz output veya error dosyası yok. `check_batch_status` ile durumu sorgulayın; tamamlanmış gibi raporlamayın.

### Run ID yeniden kullanılamıyor

Real-time ölçümleri içeren aynı run üzerinde yeni `all` çalıştırması reddedilir. Yeni experiment için yeni Run ID seçin. Batch lifecycle'ı sürdürürken ise **eski Run ID** kullanılır.

### `SUBMISSION_UNCERTAIN`

İlk gönderimin provider'da gerçekleşip gerçekleşmediği belirsizdir. Groq Console'da manuel reconciliation yapmadan ikinci kez submit etmeyin.

### `lab09 prepare_batch_input` çalışmıyor

Bu eski LAB09 komutudur. Yeni sürümde `batch_input.jsonl`, `lab09 all --live` çalıştırılınca oluşturulur.

### `lab09 async` çalışmıyor

Yeni CLI adı **`concurrent`**.

### `lab09 backfill --experiment-run-id` çalışmıyor

Backfill için `--run-id`, evaluate için `--experiment-run-id` kullanılır.

### `cost_comparison.csv` yok

Batch results henüz import edilmemiş olabilir. Cost comparison, Batch measurement'ları kullanılabilir olunca oluşur.

### Grafikleri göremiyorum

`experiment_summary.json` dosyasını kontrol edin. Gerekirse `lab09 backfill --run-id ...` kullanın. Bazı koşullarda grafik üretilecek yeterli data bulunmayabilir.

### `model_not_found`

Groq hesabınızın erişebildiği modeli doğrulayın. Model değiştiriyorsanız aynı experiment'in Sequential/Concurrent/Batch koşullarında karşılaştırılabilirliği koruyun.

---

## 25. Deneyin bilimsel sınırları

- Pilot 4, Default 8, Extended 16 görev kullanır.
- Her koşulda çok sayıda bağımsız repetition yoktur.
- Provider yükü ve network gecikmesi değişebilir.
- Concurrency nedeniyle 429 ve timeouts oluşabilir.
- Evaluator sadece belirli görev sözleşmelerini doğrular.
- Batch job'ın completion window'u gerçek time-to-first-result garantisi değildir.
- Tahmini maliyet, provider invoice değildir.
- Eksik provider usage veya output, maliyet/kalite karşılaştırmalarını sınırlayabilir.
- Batch JSONL'deki generation ayarları real-time request'lerle tam eşit değildir.
- Synthetic fixture, live telemetry değildir.
- `batch_job.json` Batch lifecycle'ı devam ettirmek içindir; real-time condition'larda checkpoint/resume yoktur.
- Batch ham JSONL çıktısı model cevaplarını içerebilir; gerçek verilerle kullanımda gizlilik politikasını değerlendirin.

---

## 26. LAB08 ile LAB09'un farkı

**LAB08:** Farklı concurrency seviyelerinin throughput, queue wait, P95 ve latency üzerindeki etkisi.

**LAB09:** Aynı iş yükünün real-time Sequential, real-time Concurrent ve provider-managed Batch olarak işlenmesinin **ürün gereksinimi ve ekonomi** açısından karşılaştırılması.

Birincisi concurrency tuning, ikincisi execution architecture seçimidir.

---

## 27. Öğrencinin deney sonunda hazırlayacağı kısa sonuç

Aşağıdaki beş hipotez için ölçümlerinizi yazın:

**H1:** Concurrent condition'ın wall-clock süresi Sequential'dan kısa mı?

**H2:** Concurrent tekil median service latency'yi azaltmış mı, artırmış mı?

**H3:** Batch maliyet avantajı hangi fiyat politikası üzerinden hesaplanıyor?

**H4:** Real-time kullanıcı için Batch neden uygun olmayabilir?

**H5:** Bütün condition'ların Golden Dataset correctness sonuçları korunmuş mu?

Cümle örneği:

> Bu çalıştırmada Concurrent condition, Sequential'a göre [ölçülen sayı] oranında [workflow/throughput] farkı gösterdi. Ancak [quality, API limits veya ölçüm kısıtı] nedeniyle sonuç yalnızca bu küçük workload için geçerlidir.

**Hiçbir stratejiyi önceden kazanan ilan etmeyin.**

---

## 28. Kaynak kodlar

| Repository dosyası | Konusu |
|---|---|
| `src/labs/lab09_batch_processing/cli.py` | CLI ve yaşam döngüsü |
| `src/labs/lab09_batch_processing/datasets.py` | Golden Dataset ve profiller |
| `data/lab09/golden_batch_dataset_v1.json` | Sentetik görevler |
| `src/labs/lab09_batch_processing/matrix.py` | Real-time experiment plan |
| `src/labs/lab09_batch_processing/runner.py` | Semaphore ve request ölçümleri |
| `src/labs/lab09_batch_processing/batch.py` | Batch Files, submit, status ve download |
| `src/labs/lab09_batch_processing/reconciliation.py` | Batch output eşleştirmesi |
| `src/labs/lab09_batch_processing/evaluation.py` | Exact/numeric evaluator |
| `src/labs/lab09_batch_processing/pricing.py` | Standard vs Batch pricing |
| `src/labs/lab09_batch_processing/stats.py` | Throughput/quality özeti |
| `src/labs/lab09_batch_processing/analysis.py` | Analiz pipeline |
| `src/labs/lab09_batch_processing/report.py` | Markdown rapor |
| `src/labs/lab09_batch_processing/charts.py` | Sekiz grafik |
| `src/labs/lab09_batch_processing/charts_cli.py` | Backfill |
| `src/labs/lab09_batch_processing/persistence.py` | Per-run persistence |
| `tests/labs/test_lab09_offline_fixtures.py` | Offline fixture uygulaması |
| `tests/fixtures/lab09/` | Synthetic provider sample verileri |
| `config/default.toml` | LAB09 config |

Resmî / proje kaynakları:

- [GitHub — llm-token-economics-lab](https://github.com/Burakkylmz/llm-token-economics-lab)
- [Groq Batch API](https://console.groq.com/docs/batch)

---

## 29. Repository'de önerilen konum

Önceki LAB01–LAB08 öğrenci rehberleriyle aynı önerilen yer:

```text
docs/participant/lab09-batch-processing-and-asynchronous-workflows.md
```

**Güncel repository notu:** `master` üzerinde ayrıca `docs/lab09-batch-processing-and-asynchronous-workflows.md` adlı mevcut rehber bulunmaktadır. İki ayrı dosya kullanılacaksa `docs/README.md` içindeki bağlantı güncel katılımcı sürümüne yönlendirilmeli; hangisinin ana rehber olduğu açıkça belirlenmelidir.

Katılımcı indeks dosyasına önerilen bağlantı:

```markdown
## LAB09 — Batch Processing & Asynchronous Workflows

[Open LAB09 Participant Guide](lab09-batch-processing-and-asynchronous-workflows.md)
```

---

**Ana çıkarım:** İyi bir LLM processing mimarisi sadece daha hızlı veya daha ucuz olanı seçmez. Kullanıcı beklentisi, job lifecycle, throughput, answer correctness, provider kapasitesi, hata yönetimi ve maliyet birlikte değerlendirilir.
