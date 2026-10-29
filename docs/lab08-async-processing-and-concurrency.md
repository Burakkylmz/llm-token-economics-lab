# LAB08 — Async Processing, Concurrency & Throughput

> **Katılımcı rehberi** · Python · Groq API · Docker Compose · `asyncio`\
> **Kapsam:** Sequential vs Concurrent execution, semaphore, queue waiting, request service latency, task end-to-end latency, throughput, P50/P95, reliability ve token economics.\
> **Önerilen ana deney:** `lab08 all --profile default` — **16 planlanan mantıksal Groq API çağrısı**.

---

## 1. Laboratuvarın amacı

Bir LLM uygulaması tek bir kullanıcıya yanıt üretirken **latency** kritik olabilir. Ancak uygulama aynı anda çok sayıda bağımsız görev işlediğinde başka bir soru ortaya çıkar:

> Birden fazla LLM API isteğini sırayla göndermek yerine eşzamanlı olarak çalıştırmak, toplam iş yükünü ne kadar hızlı tamamlar? Bunun queue waiting, request latency, throughput, hata oranı ve maliyete etkisi nedir?

Bu laboratuvarda aynı sekiz görev, aynı model ve aynı üretim parametreleriyle farklı **concurrency** seviyelerinde çalıştırılır.

**LAB08'in hedefi:** Eşzamanlılığın bir *workflow optimization* tekniği olduğunu, fakat concurrency artırmanın tek tek çağrıları otomatik olarak hızlandırmadığını ölçülebilir şekilde göstermek.

### Öğrenme hedefleri

Deney sonunda şunları ayırt edebilmelisiniz:

1. **Sequential**, **asynchronous** ve **concurrent** execution farkları.
2. `asyncio.Semaphore` ile aynı anda çalışan istek sayısının sınırlandırılması.
3. Configured concurrency ile **observed peak in-flight concurrency** farkı.
4. `queue_wait_ms`, `request_service_latency_ms` ve `task_end_to_end_latency_ms`.
5. `batch_wall_clock_ms` ve `successful_requests_per_second` ilişkisi.
6. P50 (median) ve P95 latency'nin ne anlattığı.
7. Throughput speedup ile batch time reduction arasındaki fark.
8. Daha yüksek concurrency'nin rate limit ve hata oranı açısından riski.
9. Aynı iş yükünün tahmini toplam token maliyetinin nasıl karşılaştırıldığı.
10. Hız, maliyet, doğruluk ve güvenilirliği birlikte değerlendirme.

**Temel ilke:** Daha fazla concurrency, her zaman daha iyi production performansı anlamına gelmez.

---

## 2. Çok önemli ayrım: Async, Sequential ve Concurrent

### 2.1 Asynchronous Programming

`async/await`, uygulamanın I/O bekleyen bir işi sürdürürken başka işlerin ilerlemesine izin verir.

Özellikle network çağrılarında faydalıdır.

Ancak **async kod yazılması tek başına eşzamanlı birden fazla API isteği gönderildiğini göstermez**.

### 2.2 LAB08 Sequential Condition

LAB08'in sequential koşulu:

```text
max_concurrent = 1
execution_mode = async_sequential
```

Bu koşulda kod yine **AsyncGroq** ve **asyncio** kullanır; fakat semaphore aynı anda sadece bir isteğin başlamasına izin verir.

```text
Queue: Task1 Task2 Task3 Task4 ...

Task1: [API Request] -------- Done
Task2:                       [API Request] -------- Done
Task3:                                             [API Request] ------ Done
```

**Önemli:** LAB08'de sequential = synchronous client değildir.

### 2.3 LAB08 Concurrent Condition

Örneğin:

```text
max_concurrent = 5
execution_mode = async_concurrent
```

Birden fazla bağımsız API isteği aynı anda devam edebilir:

```text
Task1: [API Request] ----------------- Done
Task2: [API Request] ---------- Done
Task3: [API Request] -------------------- Done
Task4: [API Request] -------------- Done
Task5: [API Request] ----------- Done
Task6:                             [API Request] -------- Done
```

Burada her isteğin işlenme süresi birbirinden farklı olabilir.

**Concurrency = 5**, bütün 8 isteğin tam olarak beşli gruplar halinde tamamlanacağını garanti etmez. Bir permit serbest kaldığında bekleyen başka görev başlayabilir.

### 2.4 Bu deney bir Batch API deneyi midir?

Hayır.

LAB08, bağımsız API çağrılarının **eşzamanlı yürütülmesini** inceler.

Çok sayıda isteğin provider'a özel bir Batch API ile daha sonra işlenmesi farklı bir yaklaşımdır. Bu laboratuvar concurrency ile provider-side batch scheduling'i karıştırmaz.

---

## 3. Deney mimarisi

Kaynak dosyalar:

```text
src/labs/lab08_concurrency/
  cli.py
  experiments.py
  matrix.py
  models.py
  runner.py
  evaluation.py
  stats.py
  persistence.py
  report.py
  charts.py
  charts_cli.py
```

Deneyin temel yaklaşımı:

```text
                      ┌──────────────────┐
                      │  Golden Workload │
                      │    8 Tasks       │
                      └─────────┬────────┘
                                │
                      ┌─────────▼─────────┐
                      │ Experiment Matrix │
                      └───────┬─────┬─────┘
                              │     │
                ┌─────────────▼┐   ┌▼────────────────┐
                │ Concurrency 1 │   │ Concurrency 5   │
                │ Async/Serial  │   │ Async/Concurrent│
                └─────────────┬─┘   └┬────────────────┘
                              │      │
                              └──┬───┘
                                 ▼
                       Groq API / GPT-OSS 20B
                                 │
                                 ▼
                     Measurements + Evaluation
                                 │
                  ┌──────────────┼────────────────┐
                  ▼              ▼                ▼
            Token/Cost       Latency/RPS      Accuracy/Errors
                  └──────────────┼────────────────┘
                                 ▼
                         Reports & Charts
```

### 3.1 Sabit tutulan değişkenler

Aynı profil içinde concurrency seviyeleri arasında:

- Görev metinleri
- Görev sayısı
- Model
- `reasoning_effort`
- `max_completion_tokens`
- Numeric evaluator
- Async client yaklaşımı

aynıdır.

Asıl karşılaştırılan değişken, **aynı anda izin verilen API isteği sayısıdır**.

### 3.2 Deneysel sınırlama

API provider'ın anlık yoğunluğu, network değişkenliği ve model response süreleri tam sabitlenemez.

Kod, çoklu repetition durumunda concurrency level sıralamasını dönüşümlü yaparak zaman etkisini azaltmayı hedefler. Ancak bu, tüm dışsal etkileri ortadan kaldırmaz.

---

## 4. Kullanılan model ve golden workload

LAB08 varsayılan model:

```text
openai/gpt-oss-20b
```

Default `reasoning_effort`:

```text
low
```

Default completion budget:

```text
max_completion_tokens = 256
```

### 4.1 Sekiz bağımsız aritmetik görev

`experiments.py` içinde bulunan görevler:

| Task ID | Prompt içeriği | Expected Answer |
|---|---|---:|
| `task_001` | 1 + 1 | 2 |
| `task_002` | 2 + 2 | 4 |
| `task_003` | 3 + 4 | 7 |
| `task_004` | 5 + 6 | 11 |
| `task_005` | 7 + 8 | 15 |
| `task_006` | 10 + 11 | 21 |
| `task_007` | 13 + 14 | 27 |
| `task_008` | 20 + 21 | 41 |

Gerçek prompt formatı:

```text
What is 13 plus 14? Answer with only the number.
```

**Neden basit matematik?** Bu laboratuvarın asıl değişkeni task complexity veya reasoning kalitesi değildir. Basit, ayrı, deterministik cevaplı görevler kullanmak; concurrency ve latency farklarını incelemeyi kolaylaştırır.

Her görev farklı input içerir ve beklenen cevap nettir.

### 4.2 Quality evaluator

Evaluator yalnızca **tek bir sayısal çıktı** kabul eder.

| Model response | Beklenen cevap `27` ise |
|---|---|
| `27` | Correct |
| `27.0` | Correct |
| `28` | Incorrect |
| `The answer is 27` | Invalid format |
| `27 or 28` | Invalid format |
| Boş response | Missing response |

Bir model çağrısının HTTP açısından başarılı olması, cevap formatı ve doğruluğunun da başarılı olduğu anlamına gelmez.

---

## 5. Pilot, Default ve Extended profilleri

| Profil | Görev sayısı | Concurrency levels | Repetitions | Planlanan mantıksal API çağrısı |
|---|---:|---|---:|---:|
| `pilot` | 4 | `1, 3` | 1 | **8** |
| `default` | 8 | `1, 5` | 1 | **16** |
| `extended` | 8 | `1, 2, 3, 5` | 1 | **32** |

Formül:

\[
N_{\mathrm{calls}}=N_{\mathrm{tasks}}\times N_{\mathrm{levels}}\times N_{\mathrm{repetitions}}
\]

Default:

```text
8 tasks × 2 levels × 1 repetition = 16 calls
```

Extended:

```text
8 tasks × 4 levels × 1 repetition = 32 calls
```

Bunlar planlanan **mantıksal** API çağrılarıdır. Provider hataları nedeniyle tamamlanan başarılı çağrı sayısı daha düşük olabilir.

**Önemli:** LAB08, gerçek API çağrıları kullandığı için Extended deneyi varsayılan olarak en yüksek call-budget kapsamıdır.

---

## 6. LAB08'in temel ölçümleri

### 6.1 Queue Wait

**Alan:** `queue_wait_ms`

Bir görev kuyruğa alındıktan sonra semaphore permit'i kazanana kadar geçen süredir.

Formül:

\[
T_{\mathrm{queue}}=T_{\mathrm{permit\ acquired}}-T_{\mathrm{queued}}
\]

`concurrency=1` olduğunda diğer görevlerin beklemesi doğaldır. Kodda sequential koşulun queue wait değeri **sıfıra sabitlenmez**, gerçekten ölçülür.

### 6.2 Request Service Latency

**Alan:** `request_service_latency_ms`

İsteğin API çağrısı sırasında gözlemlenen client-side işlem süresidir.

LAB08 bunu paylaşılan `call_async` altyapısının `client_elapsed_ms` değerinden alır.

Bu ölçüm, semaphore kuyruğunda beklenen süreyi dışarıda bırakmayı hedefler.

Ancak network ve provider tarafındaki bekleme gibi etkenleri içerebilir. **Saf inference veya sadece decoding time değildir.**

### 6.3 Task End-to-End Latency

**Alan:** `task_end_to_end_latency_ms`

Görev kuyruğa alındığı andan request tamamlandığı ana kadar geçen süredir.

\[
T_{\mathrm{e2e}}=T_{\mathrm{completed}}-T_{\mathrm{queued}}
\]

Yaklaşık ilişki:

\[
T_{\mathrm{e2e}}\approx T_{\mathrm{queue}}+T_{\mathrm{service}}
\]

Küçük ölçüm sınırı farkları ve Python scheduling overhead'i olabilir.

### 6.4 Batch Wall-Clock Time

**Alan:** `batch_wall_clock_ms`

Sekiz görevlik bir concurrency batch'inin başından bütün görevlerin tamamlanmasına kadar geçen gerçek istemci süresidir.

**Sıralı sekiz request'in sürelerini toplamak** ile **eşzamanlı batch'in tamamlanma süresi** aynı şey değildir.

### 6.5 Successful Requests per Second

**Alan:** `successful_requests_per_second`

\[
\mathrm{SuccessfulRPS}
=
\frac{\mathrm{SuccessfulRequests}}
{\mathrm{BatchWallClockMs}/1000}
\]

Örnek (temsili):

```text
Successful requests: 8
Batch wall-clock:    4,000 ms

Successful RPS = 8 / 4 = 2.0
```

**Throughput**, tamamlanabilen iş miktarının zamana oranıdır. Tekil request latency ile aynı şey değildir.

### 6.6 Completed Requests per Second

**Alan:** `completed_requests_per_second`

Başarılı ve başarısız dahil tamamlanan görev sayısının batch süresine bölünmesidir.

\[
\mathrm{CompletedRPS}
=
\frac{\mathrm{TotalCompletedRequests}}
{\mathrm{BatchWallClockMs}/1000}
\]

Bir sistemin çok sayıda request'i hızlıca **hata ile tamamlaması** yüksek completed RPS üretebilir.

Bu nedenle sağlıklı performans karşılaştırmasında **Successful RPS** önemlidir.

---

## 7. P50, P95 ve latency dağılımı

LAB08, tek bir ortalama süreyle yetinmez. `stats.py` aracılığıyla medyan ve yüzdebirlik (percentile) değerleri raporlanır.

### 7.1 P50

**P50 = Median** (medyan).

Yaklaşık olarak request'lerin yarısı bu sürenin altında veya eşitinde tamamlanır.

### 7.2 P95

**P95**, seçilen örneklemde latency değerlerinin yaklaşık %95'lik kısmının altında veya eşitinde kaldığı percentile eşiğidir.

P95, tipik örneklerden daha yavaş olan **tail latency** davranışını anlamaya yardımcı olur.

Örnek (temsili):

```text
Median End-to-End Latency = 1,200 ms
P95 End-to-End Latency    = 4,800 ms
```

Bu dağılımda çoğu görev makul sürede tamamlansa da yavaş kuyruk/servis olayları yaşanmış olabilir.

**Önemli:** LAB08'nin 4 veya 8 görevli küçük örneklemlerindeki P95 değeri, production SLA garantisi değildir.

### 7.3 Hangi latency'yi incelemeliyiz?

| Metrik | Cevapladığı soru |
|---|---|
| Mean/Median Queue Wait | Görevler admission için ne kadar bekliyor? |
| P50 Service Latency | API isteğinin tipik servis süresi ne? |
| P95 Service Latency | Daha yavaş API istekleri nasıl davranıyor? |
| Median End-to-End | Görevlerin tipik toplam beklemesi ne? |
| P95 End-to-End | Daha yavaş görevlerin uçtan uca süresi ne? |
| Batch Wall-Clock | Bütün iş yükü ne kadar sürede bitiyor? |

Bir concurrency değişikliği **batch süresini azaltabilir**, ama bazı görevlerin service latency değerini artırabilir. İki sonucu ayrı değerlendirin.

---

## 8. Concurrency sınırı ve Semaphore

LAB08'nin asenkron execution engine'i `asyncio.Semaphore` kullanır.

Kavramsal örnek:

```python
semaphore = asyncio.Semaphore(max_concurrent)

async def process_task(task):
    async with semaphore:
        return await call_api(task)
```

Burada `max_concurrent` bir **üst sınırdır**.

```text
Configured concurrency:     5
Observed peak concurrency:  3
```

gibi bir sonuç mümkün olabilir.

Örneğin bazı istekler erken bitmişse veya scheduling yüzünden aynı anda beş istek aktif olmamışsa gözlenen peak daha düşük olabilir.

### 8.1 Observed Peak In-Flight

**Alan:** `observed_peak_in_flight_requests`

Ölçüm, aynı anda gerçekten kaç API request'inin aktif bulunduğunun en yüksek değeridir.

Bu değer `max_concurrent` seviyesini aşmamalıdır.

### 8.2 Concurrency Utilization

**Alan:** `concurrency_utilization_ratio`

LAB08 bunu sadece `peak / configured` ile hesaplamaz. Request timeline'ından **time-weighted average in-flight** bulur.

Formül:

\[
\mathrm{Utilization}
=
\frac{\mathrm{TimeWeightedAverageInFlight}}
{\mathrm{ConfiguredConcurrency}}
\]

Örnek (temsili):

```text
Configured concurrency:       5
Time-weighted mean in-flight:  3.5

Utilization = 3.5 / 5 = 0.70 = 70%
```

Bir anlık peak=5 görülmesi, batch'in tüm süresince beş isteğin aktif kaldığı anlamına gelmez.

---

## 9. Docker hazırlığı — Windows PowerShell

### 9.1 Ön koşullar

- Docker Desktop
- Docker Compose
- LAB08 kodlarını içeren repository
- Canlı çağrılar için Groq API key
- İnternet erişimi

Python ve `uv`, Docker image içinde sağlanır.

### 9.2 Repository kök klasörüne geçin

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Farklı bir konum kullanıyorsanız klasör yolunu değiştirin.

### 9.3 API key dosyasını hazırlayın

```powershell
Copy-Item .env.example .env
```

`.env` dosyasında:

```dotenv
GROQ_API_KEY=your_api_key_here
```

Anahtarınızı commit etmeyin, paylaşmayın ve ekran görüntülerinde göstermeyin.

### 9.4 Canlı test için konfigürasyon

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` dosyasındaki mevcut `[execution]` bölümünde şu ayarı etkinleştirin:

```toml
[execution]
live_api_enabled = true
```

Diğer config bölümlerini silmeyin.

Gerçek API isteği için şu ikisi birlikte gereklidir:

- Komutta `--live`.
- Config'te `execution.live_api_enabled = true`.

### 9.5 Docker image build

```powershell
docker compose build lab-runner
```

Projede `config/` klasörü build sırasında image'a kopyalandığı için `live.toml` oluşturduktan sonra image'ı yeniden build edin.

### 9.6 Yardım ekranı

```powershell
docker compose run --rm lab-runner lab08 --help
```

Bu komut gerçek Groq API çağrısı yapmaz.

---

## 10. Docker komutları — Uygulama

### A. Dry-Run — İlk yapılması gereken işlem

```powershell
docker compose run --rm lab-runner lab08 all --profile default --dry-run --config config/default.toml
```

Bu mod:

- Deney planını gösterir.
- Görevlerin concurrency koşullarına dağıtımını belirler.
- `experiment_plan.csv` oluşturur.
- **API çağrısı yapmaz**.

Dry-run, gerçek request latency veya token kullanımı üretmez.

### B. Pilot Grand Test — 8 API çağrısı

```powershell
docker compose run --rm lab-runner lab08 all --profile pilot --live --config config/live.toml
```

Plan:

```text
4 tasks × 2 levels (1, 3) = 8 logical calls
```

İlk canlı deneme için daha küçük kapsam.

### C. Default Grand Test — 16 API çağrısı

```powershell
docker compose run --rm lab-runner lab08 all --profile default --live --config config/live.toml
```

Plan:

```text
8 tasks at concurrency=1 → 8 calls
8 tasks at concurrency=5 → 8 calls

Total → 16 calls
```

**Ana ders için önerilen deney.**

### D. Extended Grand Test — 32 API çağrısı

```powershell
docker compose run --rm lab-runner lab08 all --profile extended --live --config config/live.toml
```

Plan:

```text
8 tasks × levels [1, 2, 3, 5] = 32 calls
```

Concurrency yükseldikçe throughput'un **doğrusal artıp artmadığını** ölçmek için.

### E. Yalnızca Sequential — 8 API çağrısı

```powershell
docker compose run --rm lab-runner lab08 sequential --profile default --live --config config/live.toml
```

`max_concurrent=1` kullanır.

**Not:** Async client ile çalışır; synchronous Groq client testi değildir.

### F. Yalnızca Concurrent — 8 API çağrısı

```powershell
docker compose run --rm lab-runner lab08 concurrent --profile default --live --config config/live.toml
```

Default profil için en yüksek varsayılan concurrency seviyesi kullanılır: `5`.

### G. Concurrent seviyesini değiştirin

```powershell
docker compose run --rm lab-runner lab08 concurrent --profile default --max-concurrent 3 --live --config config/live.toml
```

**8 API çağrısı** concurrency=3 sınırıyla çalıştırılır.

Kendi run'ında yalnızca tek concurrency seviyesi bulunduğundan, aynı run içinde `concurrency=1` karşılaştırması otomatik oluşturulmaz.

### H. Üç tekrar ile deney planı

```powershell
docker compose run --rm lab-runner lab08 all --profile pilot --repetitions 3 --dry-run --config config/default.toml
```

Bu komut yalnızca **24 planlanan çağrıyı** gösterir ve gerçek istek göndermez:

```text
4 tasks × 2 levels × 3 repetitions = 24
```

Tekrar sayısını artırırken `lab08.max_api_calls` sınırını dikkate alın.

### I. Özel Run ID

```powershell
docker compose run --rm lab-runner lab08 all --profile default --experiment-run-id class-lab08-team-a --live --config config/live.toml
```

Sonuç klasörü:

```text
results/lab08_concurrency/class-lab08-team-a/
```

Aynı run ID altında ölçüm varsa LAB08 yeniden yazmayı reddeder. Sonraki çalıştırmada yeni ID seçin.

### J. Extended Dry-Run

```powershell
docker compose run --rm lab-runner lab08 all --profile extended --dry-run --config config/default.toml
```

32 çağrılık planı API maliyeti oluşturmadan görebilirsiniz.

---

## 11. LAB08 konfigürasyonunda önemli ayarlar

`config/default.toml` içindeki LAB08 varsayılan değerleri:

```toml
[lab08]
enabled = true
model = "openai/gpt-oss-20b"
default_profile = "default"
concurrency_levels = [1, 5]
repetitions = 1

reasoning_effort = "low"
max_completion_tokens = 256

randomize_condition_order = false
randomization_seed = 42

max_allowed_concurrency = 10
sdk_max_retries = 0
max_api_calls = 32
max_estimated_cost_usd = 0.05
```

### 11.1 Neden `sdk_max_retries = 0`?

Groq SDK'nın kendi otomatik retry mekanizması açık olsaydı, bir çağrı birden fazla denemeye dönüşebilir ve gözlemlenen latency'yi etkileyebilirdi.

LAB08, kendi **AsyncGroq** istemcisini bu laboratuvara özel `max_retries=0` ile oluşturur.

Bu ayar diğer laboratuvarların shared client davranışını değiştirmez.

**Önemli:** Retry kapalıyken oluşan `429`, otomatik olarak gizlenmez; ölçümde ayrı error status görülebilir.

### 11.2 Neden `max_allowed_concurrency = 10`?

Yanlışlıkla aşırı yüksek eşzamanlı API isteği gönderilmesini önlemek için tek concurrency seviyesi üzerinde bir üst sınır tanımlanmıştır.

Bu değer provider'ın resmi RPM/TPM kotası değildir; projedeki **safety guard**'dır.

### 11.3 `max_api_calls = 32`

Extended planın en yüksek çağrı kapsamıyla uyumludur.

Toplam plan bu sınırı geçerse LAB08, gerçek API çağrılarını başlatmadan çalışmayı reddeder.

### 11.4 `max_estimated_cost_usd`

Konfigürasyonda tahmini maliyet limiti bulunur. Ancak **CLI'nin mevcut LAB08 uygulamasında** diğer laboratuvarlardaki gibi her çağrı öncesi maliyet budget gate'i uygulanmıyor; `max_api_calls` ve `max_allowed_concurrency` kontrolleri ise açıkça uygulanıyor.

Bu nedenle **`max_estimated_cost_usd=0.05` değerini, LAB08 için tek başına kesin yürütme maliyeti güvencesi saymayın**. Canlı çalıştırmadan önce istek sayısını, model fiyatını ve kullanılabilir Groq kotasını kontrol edin.

### 11.5 Kodda başka LAB'ların ayarlarını değiştirmeyin

LAB08'nin model, retries ve concurrency ayarlarını kontrol ederken LAB01–LAB07 davranışlarını değiştirmeyin. Her laboratuvarın deney değişkenleri ayrı tutulmalıdır.

---

## 12. Request lifecycle ve timeline kayıtları

Her görev için LAB08 dört event kaydeder:

| Event | Anlamı |
|---|---|
| `QUEUED` | Görev kuyruğa girdi |
| `PERMIT_ACQUIRED` | Semaphore izni alındı |
| `REQUEST_STARTED` | API isteğinin yürütülmesi başladı |
| `REQUEST_COMPLETED` | İstek yürütülmesi tamamlandı |

`request_timeline.csv` alanları:

```text
execution_index
repetition_index
max_concurrent
task_id
event_type
relative_ms
```

Örnek (temsili):

```text
T=0 ms      Task 1 QUEUED
T=2 ms      Task 1 PERMIT_ACQUIRED
T=3 ms      Task 1 REQUEST_STARTED
T=6 ms      Task 2 QUEUED
T=1005 ms   Task 1 REQUEST_COMPLETED
T=1006 ms   Task 2 PERMIT_ACQUIRED
T=1007 ms   Task 2 REQUEST_STARTED
```

Bu örnekte Task 2 için:

```text
Queue wait ≈ 1006 - 6 = 1000 ms
```

`concurrency=1` için bu tür kuyruk birikimi beklenebilir.

Gantt ve in-flight grafiklerinde bu olaylar görselleştirilir.

---

## 13. API hata durumları

LAB08 request status değerleri:

| Status | Açıklama |
|---|---|
| `SUCCESS` | API çağrısı başarıyla tamamlandı |
| `RATE_LIMITED` | Rate limit yanıtı |
| `TIMEOUT` | İstek zaman aşımı |
| `CONNECTION_ERROR` | Ağ bağlantısı problemi |
| `PROVIDER_ERROR` | Diğer provider kaynaklı hata |
| `CANCELLED` | Görev iptal edildi |
| `UNKNOWN_ERROR` | Beklenmeyen hata |

Bunlar **API yürütme status'larıdır**.

Ayrıca cevap için:

- `CORRECT`
- `INCORRECT`
- `INVALID_FORMAT`
- `TRUNCATED`
- `MISSING_RESPONSE`
- `EVALUATION_UNAVAILABLE`

gibi evaluation status değerleri tutulur.

Bir `SUCCESS` çağrısının yanıtı yanlış olabilir. `SUCCESS` ile `CORRECT` aynı anlama gelmez.

### 13.1 Concurrency arttığında ne olabilir?

- Batch toplam süresi düşebilir.
- Successful RPS artabilir.
- Queue wait azalabilir.
- Provider kaynaklı service latency artabilir.
- Rate-limit hataları artabilir.
- Hata nedeniyle throughput beklenenden düşük olabilir.

Bunların hepsi **olası sonuçlardır**, garanti değildir. Deneyin amacı bunları gerçek ölçümlerle incelemektir.

---

## 14. Token Economics ve tahmini maliyet

LAB08 her başarılı Groq isteği için mevcutsa şunları kaydeder:

| Alan | Anlamı |
|---|---|
| `prompt_tokens` | Input tokenlar |
| `completion_tokens` | Output/completion tokenlar |
| `reasoning_tokens` | Provider tarafından raporlanan reasoning tokenlar |
| `non_reasoning_completion_tokens` | Uygunsa completion eksi reasoning |
| `total_tokens` | Total token sayısı |
| `cached_tokens` | Cached input tokenlar (varsa) |
| `estimated_cost_usd` | Config fiyatlarına göre hesaplanan maliyet |

**Reasoning tokenlar completion tokenlarının içindedir.** İki kez toplamaya eklemeyin.

### 14.1 Cost hesabı

Basitleştirilmiş:

\[
C_{\mathrm{total}}
=
C_{\mathrm{input}}
+
C_{\mathrm{output}}
\]

Fiyatlar bir milyon token üzerinden tanımlanmışsa:

\[
C_{\mathrm{input}} =
\frac{T_{\mathrm{uncached}}P_{\mathrm{input}}+
T_{\mathrm{cached}}P_{\mathrm{cached}}}{10^6}
\]

\[
C_{\mathrm{output}} =
\frac{T_{\mathrm{completion}}P_{\mathrm{output}}}{10^6}
\]

**Concurrency seviyesini artırmak otomatik olarak her tokenın fiyatını düşürmez.**

Bununla birlikte model çıktıları ve provider caching davranışı farklılaşabildiği için iki koşulun toplam tahmini maliyetleri birebir eşit olmayabilir.

### 14.2 Cost per Successful Request

`cost_per_successful_request_usd` yalnızca başarılı çağrıların raporlanabilir maliyetinden hesaplanır.

Bu metriği, başarısız çağrıların provider tarafından faturalandırılmadığının kesin kanıtı olarak kullanmayın. Kaydedilemeyen usage/maliyet alanları sıfır sayılmamalıdır.

---

## 15. Strateji karşılaştırma formülleri

### 15.1 Throughput Speedup

Baseline = `concurrency=1`

\[
\mathrm{ThroughputSpeedup}
=
\frac{\mathrm{ComparedSuccessfulRPS}}
{\mathrm{BaselineSuccessfulRPS}}
\]

Örnek (temsili):

```text
Sequential RPS: 1.5
Concurrent RPS: 3.0

Throughput speedup = 3.0 / 1.5 = 2.0x
```

Bu **%100 artış**, ancak **2.0x hızlanma** olarak da ifade edilebilir.

### 15.2 Batch Time Reduction

\[
\mathrm{BatchTimeReduction\%}
=
100\times
\frac{T_{\mathrm{baseline}}-T_{\mathrm{compared}}}
{T_{\mathrm{baseline}}}
\]

Örnek:

```text
Sequential batch = 8000 ms
Concurrent batch = 3500 ms

Reduction = 56.25%
```

### 15.3 Service Latency Change

\[
\mathrm{ServiceLatencyDelta\%}=
100\times
\frac{L_{\mathrm{compared}}-L_{\mathrm{baseline}}}
{L_{\mathrm{baseline}}}
\]

Pozitif değer **service latency yükseldi** anlamına gelir.

Negatif değer **service latency düştü** anlamına gelir.

### 15.4 End-to-End Latency Change

Aynı yüzde değişim formülü, **median task end-to-end latency** için uygulanır.

Bu iki latency değişimini birbirinin yerine kullanmayın.

### 15.5 Success Rate Delta

\[
\mathrm{SuccessRateDelta}
=
\mathrm{ComparedSuccessRate}-\mathrm{BaselineSuccessRate}
\]

Kod bu farkı **0–1 oran ölçeğinde** saklar. Örneğin:

```text
Baseline Success Rate: 1.00
Concurrent Success Rate: 0.75

Success Rate Delta: -0.25
```

Bu değer **25 yüzde puanı düşüş** demektir; yüzde değişim formülünün sonucu değildir.

### 15.6 Cost Delta

\[
\mathrm{CostDelta\%}
=
100\times
\frac{C_{\mathrm{compared}}-C_{\mathrm{baseline}}}
{C_{\mathrm{baseline}}}
\]

Pozitif değer: Compared condition daha pahalı.

Negatif değer: Compared condition daha ucuz.

Gerçek maliyet ölçümleri eksikse sonuç `N/A` olabilir.

---

## 16. Terminal çıktısını anlama

### 16.1 Tekil request çıktısı

Örnek terminal yapısı:

```text
[1] task_001 concurrency=1
status=SUCCESS
queue_wait=...ms
service=...ms
e2e=...ms
cost=$...
```

**İncelenecekler:**

- `queue_wait` neden sıralı koşulda artabilir?
- `service` süresi request başına benzer mi?
- `e2e` hangi koşulda düşüyor?
- Hata status'u var mı?

### 16.2 Concurrency koşulu özeti

```text
-- concurrency=5 summary --

Completed Requests:           .../...
Successful Requests:          .../...

Configured Concurrency:       5
Observed Peak Concurrency:    ...

Batch Wall-Clock Time:        ... ms
Successful Requests/sec:      ...

Mean Queue Wait:              ... ms
Median Service Latency:       ... ms
Median End-to-End Latency:    ... ms
P95 End-to-End Latency:       ... ms

Accuracy:                     .../...
Total Estimated Cost:         $...
Rate-Limited Requests:        ...
```

### 16.3 Grand Test Summary

```text
-- LAB08 Grand Test Summary --

Concurrency=1 (async_sequential):
  Batch Wall-Clock:          ... ms
  Successful RPS:            ...
  P50 Service Latency:       ... ms
  P95 Service Latency:       ... ms
  Success Rate:              ...

Concurrency=5 (async_concurrent):
  Batch Wall-Clock:          ... ms
  Successful RPS:            ...
  P50 Service Latency:       ... ms
  P95 Service Latency:       ... ms
  Success Rate:              ...

-- Comparison: concurrency=1 -> concurrency=5 --
  Throughput Speedup:        ...x
  Batch Time Reduction:      ...%
  Service Latency Change:    ...%
  End-to-End Latency Change: ...%
  Success Rate Delta:        ...
  Cost Delta:                ...%
```

Bu üç soruyu birlikte cevaplayın:

1. Toplam iş yükü daha kısa sürede tamamlandı mı?
2. Tek tek görevlerin latency'si değişti mi?
3. Cevap doğruluğu ve başarı oranı korundu mu?

---

## 17. Sonuç dosyaları

Her run için:

```text
results/
└── lab08_concurrency/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── raw_measurements.json
        ├── raw_measurements.csv
        ├── request_timeline.csv
        ├── condition_summary.csv
        ├── concurrency_comparison.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Ne için kullanılır? |
|---|---|
| `experiment_manifest.json` | Run metadata, config, profil |
| `experiment_plan.csv` | Task, concurrency, repetition planı |
| `raw_measurements.json/.csv` | Her request için latency, token, cost, status, correctness |
| `request_timeline.csv` | Queue/permit/start/complete olayları |
| `condition_summary.csv` | Her concurrency seviyesinin özet performansı |
| `concurrency_comparison.csv` | Baseline karşılaştırmaları ve delta metrikleri |
| `experiment_summary.json` | Deney genel sonuçları |
| `analysis_report.md` | Açıklamalar ve deneysel sınırlamalar |
| `charts/` | Otomatik PNG grafikleri |

**Önerilen okuma sırası:**

1. `analysis_report.md`
2. `concurrency_comparison.csv`
3. `condition_summary.csv`
4. `raw_measurements.csv`
5. `request_timeline.csv`
6. `charts/`

Ham model response içeriği bu laboratuvarın sonuç dosyalarına varsayılan olarak kaydedilmez; ölçümler ve deterministic değerlendirme çıktıları tutulur.

---

## 18. Grafikleri nasıl okuyacağız?

LAB08 dokuz grafik üretmeyi hedefler:

| Grafik | İncelenecek ilişki |
|---|---|
| `throughput_vs_concurrency.png` | Concurrency seviyesi yükseldikçe Successful RPS nasıl değişiyor? |
| `batch_completion_time.png` | Tüm iş yükü daha hızlı bitiyor mu? |
| `service_latency_distribution.png` | Service latency dağılımı ve yavaş istekler |
| `queue_waiting_time.png` | Semaphore kuyruğunda bekleme süresi |
| `task_end_to_end_latency.png` | Task'ın toplam bekleme süresi |
| `observed_in_flight_timeline.png` | Gerçek eşzamanlı aktif istek sayısı |
| `request_execution_gantt.png` | Task'ların queue/start/end zaman çizelgesi |
| `token_cost_comparison.png` | Token tüketimi ve estimated cost |
| `performance_reliability_dashboard.png` | Performans ve güvenilirlik birlikte |

### 18.1 Grafik yorumlama yöntemi

Önce:

- `batch_completion_time.png`
- `throughput_vs_concurrency.png`

grafiklerine bakın.

Ardından:

- `queue_waiting_time.png`
- `service_latency_distribution.png`
- `task_end_to_end_latency.png`

üzerinden **neden** hızlanma veya yavaşlama olduğunu araştırın.

Son olarak:

- `performance_reliability_dashboard.png`
- `token_cost_comparison.png`

ile kalite ve maliyeti değerlendirin.

**Tek bir grafikten “en iyi concurrency = 5” sonucu çıkarmayın.** İş yükü, quota ve provider davranışları değişebilir.

---

## 19. Backfill — API olmadan grafikleri yeniden üretme

Önceden kaydedilmiş sonuçlar için:

```powershell
docker compose run --rm lab-runner lab08 backfill --config config/default.toml
```

Belirli bir run için:

```powershell
docker compose run --rm lab-runner lab08 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasör adını yazın.

Backfill:

- Gerçek API çağrısı yapmaz.
- API key gerektirmez.
- Kayıtlı ölçümlerden PNG grafiklerini yeniden üretir.
- JSON/CSV/Markdown ölçümlerini değiştirmez.

---

## 20. Sınıf içi uygulama görevleri

### Görev 1 — Dry-Run ile experiment matrix

```powershell
docker compose run --rm lab-runner lab08 all --profile default --dry-run --config config/default.toml
```

`experiment_plan.csv` dosyasını açın.

Şunları doğrulayın:

- Kaç farklı task var?
- Kaç concurrency level var?
- Kaç repetition var?
- Toplam kaç çağrı planlandı?

**Soru:** Neden concurrency=5 için de 8 ayrı görev bulunuyor?

### Görev 2 — Pilot Grand Test

```powershell
docker compose run --rm lab-runner lab08 all --profile pilot --live --config config/live.toml
```

Tabloyu gerçek değerlerle doldurun:

| Metrik | Concurrency=1 | Concurrency=3 |
|---|---:|---:|
| Successful Requests | | |
| Batch Wall-Clock (ms) | | |
| Successful RPS | | |
| Mean Queue Wait (ms) | | |
| P95 End-to-End (ms) | | |
| Correct Answers | | |

**Soru:** Queue wait azalırken service latency artabilir mi?

### Görev 3 — Default Grand Test

```powershell
docker compose run --rm lab-runner lab08 all --profile default --live --config config/live.toml
```

`concurrency_comparison.csv` dosyasından hesaplanan sonuçları kullanın:

| Metrik | Ölçülen değer |
|---|---|
| Throughput Speedup | |
| Batch Time Reduction | |
| Median Service Latency Delta | |
| Median End-to-End Delta | |
| Success Rate Delta | |
| Cost Delta | |

**Soru:** Throughput artarken tekil isteğin service latency'si de artarsa nasıl yorumlarız?

### Görev 4 — Queue Timeline ve Gantt

`request_timeline.csv` dosyasını inceleyin.

Aynı task için:

- `QUEUED`
- `PERMIT_ACQUIRED`
- `REQUEST_STARTED`
- `REQUEST_COMPLETED`

olaylarını bulun.

Queue wait ve total end-to-end süresini elle hesaplayın.

**Soru:** Concurrency=1 için queue wait neden çoğu görevde sıfıra yakın değildir?

### Görev 5 — Rate Limit Riski

Default ve Extended sonuçlarını karşılaştırın:

```powershell
docker compose run --rm lab-runner lab08 all --profile extended --live --config config/live.toml
```

Şu alanları inceleyin:

- `rate_limited_requests`
- `successful_requests`
- `success_rate`
- `successful_requests_per_second`
- `observed_peak_in_flight_requests`

**Soru:** Concurrency yükseltildiğinde throughput'un artmaması hangi koşullarda beklenir?

### Görev 6 — Latency Percentiles

`condition_summary.csv` dosyasından P50 ve P95 değerlerini inceleyin.

Bir concurrency seviyesi için şunları yanıtlayın:

1. P95, P50'den belirgin biçimde yüksek mi?
2. Yavaş görevlerde queue wait mi, service latency mi baskın?
3. Bu küçük örneklemden production SLA çıkarabilir miyiz?

### Görev 7 — Quality–Cost–Performance Trade-off

Aşağıdaki tabloyu gerçek ölçümlerle doldurun:

| Concurrency | Successful RPS | Success Rate | Correct Answers | Estimated Cost |
|---|---:|---:|---:|---:|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 5 | | | | |

Bu dört seviyeyi birlikte karşılaştırmak için Extended profil gerekir.

**Soru:** Sizin sisteminizde maliyet, hız ve güvenilirlikten hangisi en kritik? Aynı concurrency seviyesini bütün uygulamalarda kullanmak doğru olur mu?

---

## 21. Tartışma soruları

1. Async programming ile concurrency aynı şey midir?
2. `concurrency=1` kullanırken neden hâlâ AsyncGroq client kullanılıyor?
3. Semaphore neyi kontrol eder?
4. `max_concurrent=5` her an beş request'in aktif olduğu anlamına gelir mi?
5. Queue wait ile service latency arasındaki sınır nerede başlar?
6. Batch completion time ile tekil request latency neden farklıdır?
7. Throughput artırmak P95 latency'yi kötüleştirebilir mi?
8. 429 hataları arttığında yüksek concurrency'yi başarı sayabilir miyiz?
9. Neden completed RPS yerine successful RPS'ye bakmalıyız?
10. Configured concurrency ile provider'ın RPM/TPM limitleri arasındaki fark nedir?
11. SDK automatic retry'nin kapatılması deney güvenilirliğini nasıl etkiler?
12. Concurrency artışı neden token başına fiyatı otomatik düşürmez?
13. Kaç çağrıyla P95 hakkında güvenilir bir sonuca varılabilir?
14. Bu sonuçlar üretim kapasitesi planlamasında nasıl ve hangi sınırlamalarla kullanılmalı?

---

## 22. Troubleshooting

### 22.1 `LiveExecutionNotAllowed`

Muhtemel nedenler:

- `--live` eksik.
- `config/live.toml` içinde `live_api_enabled = false`.
- Docker image eski konfigürasyonu içeriyor.

Çözüm: Live config'i doğrulayıp image'ı yeniden build edin.

### 22.2 `GROQ_API_KEY` yok

`.env` dosyasında geçerli anahtar olduğundan emin olun. Anahtarınızı paylaşmayın.

### 22.3 HTTP 429 / `RATE_LIMITED`

Provider'ın rate limit veya kota sınırı aşılmış olabilir.

- Aynı deneyi peş peşe çalıştırmayın.
- Groq Console üzerinden kota durumunu kontrol edin.
- Pilot profile veya daha düşük concurrency seviyesine dönün.
- SDK retry'nin burada kasıtlı olarak kapalı olduğunu unutmayın.

### 22.4 `planned calls exceed lab08.max_api_calls`

Toplam iş yükü 32 çağrı limitini aşmıştır.

Planlanan çağrı sayısı:

```text
Task count × Concurrency level count × Repetitions
```

Önce daha küçük profile dönün.

### 22.5 `max_allowed_concurrency`

`--max-concurrent` değeri LAB08 güvenlik tavanını aşıyorsa işlem reddedilir.

Varsayılan tavan `10`dur. Bu, Groq hesabınızın resmi eşzamanlılık limiti değildir.

### 22.6 `estimated_cost_usd = N/A`

Model fiyat konfigürasyonu veya provider usage bilgileri bulunmuyor olabilir.

Eksik ölçümü sıfır olarak yorumlamayın.

### 22.7 `MISSING_RESPONSE`, `TRUNCATED` veya `INVALID_FORMAT`

API isteği başarılı olsa bile modelin numeric output contract'ı sağlanmamış olabilir.

### 22.8 Concurrency=5 beklenen hızlanmayı sağlamadı

Olası nedenler:

- Provider yükü
- Request service latency artışı
- Rate limit
- Network değişkenliği
- Düşük observed peak concurrency
- Küçük görev sayısı
- Yanıt uzunluğu / reasoning token varyasyonu

Doğrudan “async çalışmıyor” sonucuna varmayın.

### 22.9 Grafikler oluşmadı

Önce JSON ve CSV sonuçlarının varlığını doğrulayın; sonra `lab08 backfill` kullanın.

### 22.10 Run ID yeniden kullanılamıyor

Önceki run ölçümleri korunur; yeni `--experiment-run-id` verin.

---

## 23. LAB08'in diğer laboratuvarlarla ilişkisi

- **LAB04:** Gereksiz model invocation ve output budget optimizasyonu.
- **LAB06:** Streaming'in perceived latency etkisi.
- **LAB07:** Rate limiting ve retry stratejilerinin davranışı.
- **LAB08:** Çok sayıda bağımsız API isteğinin concurrency ile yürütülmesi.

Özellikle LAB07 ile LAB08 bağlantısı önemlidir:

**Concurrency yükseltmek throughput'u iyileştirebilir; fakat provider rate limit'i karşısında 429 riskini artırabilir.**

Production'da concurrency, yalnızca `asyncio` mekanizmasıyla değil, **rate limiting, admission control, retries ve maliyet politikalarıyla birlikte** düşünülmelidir.

---

## 24. Sonuç

LAB08 tamamlandığında aşağıdaki çıkarımı savunabilmelisiniz:

> **Concurrency optimizasyonunun hedefi, her API isteğini ayrı ayrı hızlandırmak değil, bağımsız işleri daha verimli biçimde tamamlamaktır. Başarılı optimizasyon; throughput, batch süresi, queue wait, service latency, tail latency, doğruluk, hata oranı ve maliyeti birlikte değerlendirmeyi gerektirir.**

Tek bir concurrency seviyesi her iş yükü ve her Groq hesabı için en iyi seçenek değildir.

---

## 25. Kaynak kodlar

| Dosya | İncelenecek bölüm |
|---|---|
| `src/labs/lab08_concurrency/cli.py` | CLI ve experiment orchestration |
| `src/labs/lab08_concurrency/experiments.py` | Golden arithmetic workload |
| `src/labs/lab08_concurrency/matrix.py` | Profil ve concurrency seviyeleri |
| `src/labs/lab08_concurrency/runner.py` | Async execution, semaphore, timeline |
| `src/labs/lab08_concurrency/evaluation.py` | Numeric correctness |
| `src/labs/lab08_concurrency/stats.py` | RPS, P50/P95, utilization, comparisons |
| `src/labs/lab08_concurrency/models.py` | Measurement schemas ve request statuses |
| `src/labs/lab08_concurrency/persistence.py` | Per-run sonuçlar |
| `src/labs/lab08_concurrency/charts.py` | Dokuz PNG grafik |
| `src/labs/lab08_concurrency/report.py` | Analysis report |
| `config/default.toml` | LAB08 varsayılan ayarları |

---

## 26. Repository'de önerilen konum

Öğrenci rehberini şu dizine yerleştirin:

```text
docs/participant/lab08-async-processing-and-concurrency.md
```

`docs/participant/README.md` dosyasına ekleyin:

```markdown
## LAB08 — Async Processing & Concurrency

[Open LAB08 Participant Guide](lab08-async-processing-and-concurrency.md)
```
