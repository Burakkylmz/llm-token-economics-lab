# LAB04 — Output Control & Invocation Optimization

> **Katılımcı rehberi · Türkçe · Groq API · Python · Docker Compose**\
> **LAB04'ün amacı:** LLM çıktısını sınırlandırmanın token tüketimi, cevap bütünlüğü, latency ve maliyet üzerindeki etkisini; ayrıca gereksiz API çağrılarının workflow ekonomisini nasıl değiştirdiğini kontrollü deneylerle incelemek.

## 1. Öğrenme hedefleri

Bu laboratuvarı bitirdiğinizde aşağıdaki sorulara ölçümler üzerinden cevap verebilmelisiniz:

1. `max_completion_tokens` tam olarak neyi sınırlar? Neden her zaman üretilen token sayısını göstermez?
2. Reasoning destekleyen bir modelde küçük completion budget neden beklenmedik biçimde eksik cevap oluşturabilir?
3. `finish_reason="length"` ile doğruluk/bütünlük testleri birlikte nasıl yorumlanır?
4. Tek bir API çağrısında birden fazla soru cevaplatmak, üç ayrı çağrıya göre daha ekonomik olabilir mi?
5. Toplam token tüketimi, tahmini maliyet ve **workflow wall-clock latency** nasıl karşılaştırılır?
6. Ucuz görünen fakat hatalı/eksik cevap üreten yaklaşım gerçekten bir optimizasyon mudur?

**Laboratuvarın iki bağımsız deneyi vardır:** `output_limit_sweep` ve `invocation_comparison`. `lab04 all` ikisini sırayla çalıştırır.

## 2. Deney mimarisi ve koşullar

| Deney | Kontrol edilen değişken | Varsayılan koşullar | Planlanan çağrı |
|---|---|---|---:|
| `output_limit_sweep` | `max_completion_tokens` | 64 / 256 / 1024, her biri 2 tekrar | 6 |
| `invocation_comparison` | Aynı sorular için API invocation sayısı | 1 birleşik çağrı / 3 ayrı çağrı | 4 |
| `all` | Her iki deney | İki deneyin tüm koşulları | **10** |

- **Provider:** Groq API
- **Model:** `openai/gpt-oss-20b`
- **Reasoning effort:** `low` (iki deneyde de sabit)
- **Invocation çalışma biçimi:** `sequential` (concurrency bu laboratuvarın konusu değildir)
- **Konfigürasyon:** `config/default.toml` içindeki `[lab04]` bölümü

### 2.1 Deney A — Output Limit Sweep

Bu deney, **her çağrıda aynı input mesajlarını** kullanır. Modelden kısa bir API Retry Policy belgesinden beş alan çıkarması istenir.

Kaynak metinde şu bilgiler bulunur:

```text
Initial request counts as attempt 1.
Maximum retry attempts: 4.
Maximum total attempts: 5.
Base retry delay: 1 second.
Nominal exponential backoff delays: 1, 2, 4, 8 seconds.
Retryable HTTP status for this exercise: 429.
```

Beklenen output formatı:

```text
max_retry_attempts: 4
max_total_attempts: 5
base_delay_seconds: 1
last_nominal_delay_seconds: 8
retryable_http_status: 429
```

Bu, gerçek servisiniz için önerilen evrensel bir retry policy değil, **denetimli bir eğitim verisidir**.

| Condition | `max_completion_tokens` | Deneyde beklenen gözlem |
|---|---:|---|
| `restrictive` | 64 | Reasoning overhead yüzünden eksik/truncated cevap riski |
| `moderate` | 256 | Daha fazla generation bütçesi; yine de kalite garanti değil |
| `generous` | 1024 | Daha geniş üst sınır; maliyet ve kalite karşılaştırması |

**Hipotez:** Çok dar completion budget, daha az output token tüketebilir ancak gerekli alanların tamamının üretilmesini engelleyebilir. Büyük bütçe ise modelin mutlaka o kadar token kullanacağı anlamına gelmez.

**Kontroller:** Prompt, model, reasoning effort ve beklenen cevap değişmez. Bu bölümde farklılaştırılan esas değişken `max_completion_tokens` değeridir.

### 2.2 Deney B — Invocation Comparison

Bu kez aynı üç genel bilgi sorusu iki farklı stratejiyle çözülür:

| Soru | Beklenen bilgi |
|---|---|
| Japonya'nın başkenti nedir? | Tokyo |
| Deniz seviyesinde su kaç °C'de kaynar? | 100 |
| *Hamlet* oyununun yazarı kimdir? | William Shakespeare |

**Single Invocation:** Üç soru tek request'te gönderilir; modelden üç satırlık alan/değer formatı istenir. **1 API çağrısı** yapılır.

**Multiple Invocations:** Her soru ayrı request ile gönderilir. **3 API çağrısı** yapılır.

```text
SINGLE:
  [Question 1 + Question 2 + Question 3] -> API Call -> 3 answers

MULTIPLE (sequential):
  [Question 1] -> API Call -> Answer 1
  [Question 2] -> API Call -> Answer 2
  [Question 3] -> API Call -> Answer 3
```

Her iki stratejinin de iş açısından gerekli üç cevabı üretmesi beklenir. Birleştirme, tüm üretim durumlarında otomatik olarak daha ucuz veya daha doğru değildir; **ölçümler belirleyicidir**.

**Önemli değerlendirme ayrıntısı:** Single stratejide üç alanı birlikte döndüren cevap yapılandırılmış olarak kontrol edilir. Multiple stratejide tekil, serbest metin cevapların doğruluğu daha esnek bir kelime eşleştirme *heuristic* yöntemiyle kontrol edilir. Bu iki evaluator aynı katılıkta değildir; karşılaştırmanın sınırlamalarından biridir.

---

## 3. Ön hazırlık — Windows PowerShell

### 3.1 Gereksinimler

- Docker Desktop + Docker Compose
- LAB repository'sinin yerel kopyası
- **Live** deneyler için geçerli Groq API key
- İnternet erişimi ve Groq hesabında kullanılabilir API kotası

Komutları repository kökünde çalıştırın:

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Kendi repository konumunuz farklıysa bu yolu değiştirin.

### 3.2 API anahtarı

Henüz yoksa `.env` dosyasını oluşturun:

```powershell
Copy-Item .env.example .env
```

`.env` içinde:

```dotenv
GROQ_API_KEY=your_api_key_here
```

**Güvenlik:** API key'i GitHub'a commit etmeyin, ekran paylaşımında açık bırakmayın ve başkasının key'ini kullanmayın.

### 3.3 Canlı çalışma konfigürasyonu

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` içindeki mevcut `[execution]` bölümünü bulun ve aşağıdaki alanı değiştirin:

```toml
live_api_enabled = true
```

`[execution]` bölümünün diğer alanlarını ve dosyadaki diğer bölümleri silmeyin. `config/default.toml` güvenli varsayılan olarak `false` kalmalıdır.

Canlı API çağrısının başlayabilmesi için **iki koşul da** gereklidir:

- CLI'da `--live`
- Live konfigürasyonunda `execution.live_api_enabled = true`

### 3.4 Image oluşturma

```powershell
docker compose build lab-runner
```

Projenin Dockerfile'ı kodu ve `config` klasörünü image içine kopyaladığı için, `config/live.toml` oluşturduktan veya kodu güncelledikten sonra yeniden build edin.

### 3.5 CLI doğrulama

```powershell
docker compose run --rm lab-runner lab04 --help
```

Bu komut gerçek API çağrısı yapmaz.

---

## 4. LAB04 Docker komutları

### A — Default Grand Test (önerilen sınıf çalışması)

```powershell
docker compose run --rm lab-runner lab04 all --live --config config/live.toml
```

Varsayılan kapsam:

```text
Output Limit Sweep:
  3 conditions × 2 repetitions = 6 logical calls

Invocation Comparison:
  1 single call + 3 multiple calls = 4 logical calls

TOTAL = 10 planned logical API calls
```

Gerçek HTTP deneme sayısı retry mekanizmaları nedeniyle farklı olabilir. API kotası veya maliyet sınırı nedeniyle koşulların tamamı çalışmayabilir.

### B — Dry Run (önce bunu çalıştırın)

```powershell
docker compose run --rm lab-runner lab04 all --dry-run --config config/default.toml
```

Bu işlem:

- Deney koşullarını ve planlanan çağrıları hazırlar.
- `experiment_plan.csv` oluşturur.
- **Groq API'ye çağrı göndermez.**
- Gerçek kullanım, latency veya maliyet ölçümleri üretmez.

### C — Pilot Grand Test (daha düşük kota)

```powershell
docker compose run --rm lab-runner lab04 all --repetitions 1 --live --config config/live.toml
```

- Output limit sweep: `3 × 1 = 3` çağrı
- Invocation comparison: `1 + 3 = 4` çağrı
- **Toplam: 7 mantıksal API çağrısı**

`--repetitions` yalnızca output limit sweep'in tekrar sayısını etkiler; invocation comparison her zaman dört planlanmış çağrı içerir.

### D — Yalnızca Output Limit Sweep

```powershell
docker compose run --rm lab-runner lab04 output_limit_sweep --live --config config/live.toml
```

**6 mantıksal çağrı** planlanır.

Daha küçük pilot:

```powershell
docker compose run --rm lab-runner lab04 output_limit_sweep --repetitions 1 --live --config config/live.toml
```

**3 mantıksal çağrı** planlanır.

### E — Yalnızca Invocation Comparison

```powershell
docker compose run --rm lab-runner lab04 invocation_comparison --live --config config/live.toml
```

**4 mantıksal çağrı** planlanır (1 birleşik + 3 ayrı).

### F — Özel run ID

```powershell
docker compose run --rm lab-runner lab04 all --experiment-run-id lab04-class-group-a --live --config config/live.toml
```

Sonuçlar:

```text
results/lab04_output_control/lab04-class-group-a/
```

Aynı run ID altında ölçüm bulunan bir deneyi yeniden başlatmak reddedilir. Yeniden çalıştırmak için yeni run ID kullanın.

### G — Planlanan çağrı sınırını aşan örnek (çalıştırmayın)

```powershell
# ÖRNEK: 3 × 3 + 4 = 13 çağrı, varsayılan max_api_calls = 10 sınırını aşar.
# docker compose run --rm lab-runner lab04 all --repetitions 3 --live --config config/live.toml
```

Bu kapsam normalde preflight kontrolü tarafından reddedilir. Limitleri bilinçsizce yükseltmeyin.

---

## 5. Output Limit Sweep metrikleri

| Metrik | Anlamı |
|---|---|
| `max_completion_tokens` | Provider'a gönderilen completion token **üst sınırı** |
| `prompt_tokens` | Provider'ın raporladığı input tokenları |
| `completion_tokens` | Reasoning dahil toplam completion tokenları |
| `reasoning_tokens` | Reasoning için kullanılan ve completion toplamına dahil olan tokenlar |
| `non_reasoning_completion_tokens` | Geçerli ölçümler varsa `completion_tokens - reasoning_tokens` |
| `total_tokens` | Provider'ın raporladığı toplam kullanım |
| `finish_reason` | Modelin yanıtı hangi nedenle durdurduğu |
| `output_truncated` | `finish_reason == "length"` ise `True` |
| `required_field_count` | Gereken alan sayısı (bu görevde 5) |
| `present_field_count` | Modelin çıktısında bulunan gerekli alanlar |
| `correct_field_count` | Doğru değere sahip gerekli alanlar |
| `field_completeness_ratio` | Bulunan gerekli alanlar / toplam gerekli alanlar |
| `field_correctness_ratio` | Doğru gerekli alanlar / toplam gerekli alanlar |
| `overall_answer_accepted` | Gerekli alanların tümü mevcut ve doğru mu? |
| `total_latency_ms` | Client tarafında request süresi |
| `estimated_cost_usd` | Konfigüre fiyatlarla hesaplanan tahmini maliyet |

### 5.1 `max_completion_tokens` bir hedef değil, sınırdır

Örneğin `max_completion_tokens=1024` tanımlamak, modelin 1024 token kullanacağı anlamına **gelmez**. Model daha erken durabilir.

Ayrıca reasoning destekleyen modellerde completion budget yalnızca kullanıcıya görünen birkaç satırla tüketilmez. Reasoning tokenları da bütçeye dahildir.

Temsili örnek:

```text
max_completion_tokens:          256
completion_tokens:              256
reasoning_tokens:               245
non_reasoning_completion_tokens: 11
finish_reason:                 length
```

Bu değerler **temsili**dir; laboratuvarda ölçülmüş sonuç değildir.

Modelin bütçesinin büyük bölümü reasoning tarafından tüketilmiş olabilir. Cevap görünür olsa bile eksik olabilir veya hiç gerekli alan çıkmayabilir.

### 5.2 Reasoning tokenlarını iki kere saymayın

Basitleştirilmiş ilişki:

\[
T_{completion} = T_{reasoning}+T_{non\text{-}reasoning}
\]

Örnek: 200 completion tokenının 180'i reasoning ise, toplam completion 380 değil **200** tokendır.

### 5.3 `finish_reason` ve Truncation

- `stop`: Provider normal duruş bildirdi. Bu, cevabın **mutlaka doğru** olduğu anlamına gelmez.
- `length`: Provider üretimin uzunluk/bütçe sınırına takıldığını bildirdi. LAB04 bunu `output_truncated=True` olarak kaydeder.
- Eksik `finish_reason`: Truncation bilgisi mevcut değil; sıfır veya `False` diye varsaymayın.

**Önemli:** LAB04, `completion_tokens == max_completion_tokens` eşitliğine bakarak truncation sonucu uydurmaz. Sağlayıcının gerçek `finish_reason` değerini kullanır.

### 5.4 Completeness ile Correctness farklıdır

Bu görevde beş zorunlu alan vardır.

Temsili sonuç:

```text
Required Fields:   5
Present Fields:    3
Correct Fields:    2
```

\[
Completeness = 3/5 = 60\%
\]

\[
Correctness = 2/5 = 40\%
\]

`overall_answer_accepted=False` olur.

Ayrıca:

- Eksik alanı doğru saymayın.
- Yanlış değeri doğru saymayın.
- Tekrarlanan (`duplicate`) alan kabul edilmez.
- Gereksiz `extra` alanlar ayrıca raporlanır.
- Boş/parse edilemeyen çıktı `missing_response` olarak ele alınabilir.

**Ana ders:** Yalnızca çıktıyı kısaltan ancak kritik alanları kaybettiren bir ayar başarılı optimizasyon değildir.

---

## 6. Invocation Comparison metrikleri

| Metrik | Neden önemli? |
|---|---|
| `api_call_count` | Tek çağrı ile üç çağrı arasındaki temel fark |
| `total_prompt_tokens` | İlgili stratejinin toplam input tüketimi |
| `total_completion_tokens` | İlgili stratejinin toplam completion tüketimi |
| `total_estimated_cost_usd` | Üç soruyu cevaplamak için stratejinin toplam maliyeti |
| `workflow_wall_clock_ms` | Tüm stratejinin başlangıçtan bitişe gerçek elapsed süresi |
| `client_elapsed_ms` | Her tekil request'in client-side süresi |
| `all_answers_accepted` | Strateji gereken bütün cevapları başarıyla sağladı mı? |
| `quality_comparison` | Maliyet avantajı kaliteyle birlikte değerlendirildiğinde durum |

### 6.1 Per-request latency vs Workflow wall-clock

Single strateji bir çağrının etrafında ölçülür. Multiple strateji ise üç **ardışık** çağrıdan oluşan bütün döngü boyunca ölçülür.

```text
Multiple strategy:
request_1 -> request_2 -> request_3
|---------------- workflow ----------------|
```

Bu nedenle:

\[
L_{workflow} \neq \sum_i L_{client,i}
\]

Tam eşit olmak zorunda değildir; uygulama içinde çağrılar arasında ek iş olabilir.

**LAB04, concurrency veya parallel processing performans testi yapmaz.** Bunlar LAB08'de incelenir.

### 6.2 Total Cost ve Cost Difference

İki stratejide iş açısından ihtiyaç aynı üç cevaptır.

\[
C_{single}=\sum C_{\text{single strategy calls}}
\]

\[
C_{multiple}=\sum C_{\text{multiple strategy calls}}
\]

Kodun kullandığı fark:

\[
\Delta C=C_{multiple}-C_{single}
\]

- `ΔC > 0`: Multiple daha pahalı gözlendi.
- `ΔC < 0`: Single daha pahalı gözlendi.
- `ΔC = 0`: Tahmini maliyetler eşit gözlendi.

Yüzdesel fark, uygun veri varsa `single` maliyeti referans alınarak hesaplanır. Veriler eksikse sonuç `N/A` olabilir.

### 6.3 Quality-constrained optimization

Single stratejinin `1` çağrı yapması, her zaman daha iyi olduğu anlamına gelmez. İstenilen üç cevaptan biri yanlışsa veya yanlış formatta geldiyse maliyet düşmüş olsa da görev başarısız olabilir.

Örnek yorum:

> Single strateji daha düşük tahmini maliyete sahip çıktı, ancak üç gerekli alanın tamamını doğru üretmedi. Bu nedenle sonuç, kaliteden bağımsız bir maliyet avantajı olarak değerlendirilmeli; kalite açısından başarılı optimizasyon denmemeli.

**Küçük ama önemli fark:** Terminaldeki `Correct Answers` sayısı Single için **birleştirilmiş çağrının kabul durumu** üzerinden, Multiple için **ayrı çağrıların kabul durumları** üzerinden raporlanır. `1/1` ile `3/3` sayılarını “aynı üç alanın doğruluk yüzdesi” diye doğrudan karşılaştırmayın. İlgili alan bazlı metrikleri ve `all_answers_accepted` sonucunu da inceleyin.

Multiple stratejinin serbest metin evaluator'ı daha esnek olduğundan, `accepted` sonucu her iki stratejide birebir aynı değerlendirme katılığında olmayabilir.

---

## 7. Terminal çıktılarını okuma

### 7.1 Output Limit Sweep

Gerçek ölçümlerde aşağıdaki alanlar bulunur:

```text
Experiment: output_limit_sweep
Condition: restrictive
Run: 1/2

Max Completion Tokens:      ...
Prompt Tokens:              ...
Completion Tokens:          ...
Reasoning Tokens:           ...
Non-Reasoning Tokens:       ...
Finish Reason:              ...
Output Truncated:           ...
Required Fields:            5
Present Fields:             ...
Correct Fields:             ...
Answer Completeness:        ...
Answer Correct (all fields):...
Client Latency:             ... ms
Estimated Cost:             $...
```

İnceleme sırası:

1. Limit kaç token?
2. Model ne kadar completion/reasoning token tüketti?
3. `finish_reason=length` var mı?
4. Beş alanın kaçı mevcut ve doğru?
5. Tahmini maliyet ve latency ne oldu?

### 7.2 Invocation Comparison

```text
-- invocation_comparison summary --

Single Invocation:
  API Calls:               1
  Total Prompt Tokens:     ...
  Total Completion Tokens: ...
  Total Estimated Cost:    $...
  Workflow Wall-Clock:     ... ms

Multiple Invocations:
  API Calls:               3
  Total Prompt Tokens:     ...
  Total Completion Tokens: ...
  Total Estimated Cost:    $...
  Workflow Wall-Clock:     ... ms

Cost Difference:            ...
Latency Difference:         ...
Quality Comparison:         ...
```

**İlk kontrol:** İki strateji de gereken bilgileri doğru sağladı mı? **İkinci kontrol:** Aynı kalite sağlanıyorsa hangisinin toplam token, maliyet ve workflow latency değerleri daha avantajlı?

---

## 8. Sonuç dosyaları ve klasörler

Canlı bir çalıştırmanın dosyaları şu dizinde tutulur:

```text
results/
└── lab04_output_control/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── output_limit_measurements.json
        ├── output_limit_measurements.csv
        ├── invocation_measurements.json
        ├── invocation_measurements.csv
        ├── condition_summary.csv
        ├── invocation_comparison.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

Bazı dosyalar yalnızca ilgili deney veya yeterli veri mevcutsa üretilir (örneğin `invocation_comparison.csv`).

| Dosya | İçerik |
|---|---|
| `experiment_manifest.json` | Çalıştırma kimliği, model, konfigürasyon, yürütme durumu |
| `experiment_plan.csv` | Planlanan output limit ve invocation çağrıları |
| `output_limit_measurements.json/.csv` | Her limit/tekrar için token, truncation, completeness, maliyet, latency |
| `invocation_measurements.json/.csv` | Single ve multiple çağrılarının tekil ölçümleri |
| `condition_summary.csv` | Limit koşullarının özet istatistikleri |
| `invocation_comparison.csv` | Stratejilerin birikimli karşılaştırması (mevcutsa) |
| `experiment_summary.json` | Çalıştırmanın toplu sonuçları |
| `analysis_report.md` | Araştırma sonuçları ve metodolojik sınırlamalar |
| `charts/` | PNG grafikler |

**Önemli:** LAB04, modelin ham yanıt metinlerini varsayılan olarak ölçüm dosyalarına kaydetmez; char count, correctness/completeness gibi türetilmiş bilgiler tutulur. Bu, LAB10'un sentetik input/output inspection tasarımından farklıdır.

Önce `analysis_report.md`, sonra `condition_summary.csv`, ardından ham ölçümler incelenmelidir.

---

## 9. Grafikler ve yorumlama

LAB04, kullanılabilir ölçümlere göre aşağıdaki grafik dosyalarını üretir:

| Grafik | Öğrenci neye bakmalı? |
|---|---|
| `completion_tokens_vs_budget.png` | Bütçe büyüdükçe completion tokenlar nasıl değişiyor? |
| `reasoning_vs_non_reasoning_tokens.png` | Completion bütçesi reasoning ve yanıt arasında nasıl paylaşılıyor? |
| `finish_reason_distribution.png` | Hangi koşullarda `length` daha sık görülüyor? |
| `answer_completeness_vs_budget.png` | Çıktı bütünlüğü budget ile değişiyor mu? |
| `output_limit_vs_estimated_cost.png` | Daha geniş sınır gerçekten daha pahalı mı? |
| `invocation_token_breakdown.png` | Single vs Multiple toplam input/completion tüketimi |
| `invocation_cost_comparison.png` | Workflow'un toplam tahmini maliyeti |
| `workflow_wall_clock_latency.png` | Tek birleşik ve üç ardışık çağrının süresi |
| `quality_constrained_optimization_dashboard.png` | Maliyet, kalite ve performansın birlikte yorumu |

Grafiklerin oluşturulması kullanılabilir verilere bağlıdır. Yalnızca bir deney çalıştırırsanız diğer bölümün bazı grafikleri boş veya oluşturulmamış olabilir.

**Grafik okuma ilkesi:** Trendleri yorumlarken tekrar sayısının düşük olduğunu unutmayın; tek çalıştırmadan modelin tüm isteklerde aynı şekilde davranacağı sonucunu çıkarmayın.

---

## 10. Offline Chart Backfill

Var olan sonuçlardan yeniden grafik üretmek için:

```powershell
docker compose run --rm lab-runner lab04 backfill --config config/default.toml
```

Yalnızca seçilen çalıştırmanın grafikleri için:

```powershell
docker compose run --rm lab-runner lab04 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine ilgili sonuç klasörünün adını yazın.

Backfill:

- API çağrısı yapmaz.
- API key gerektirmez.
- Var olan ölçümlerden PNG grafiklerini yeniden oluşturur.
- Kayıtlı JSON/CSV/Markdown ölçüm sonuçlarını değiştirmez.
- `--run-id` verilmezse mevcut per-run sonuçlarıyla beraber varsa eski düz (legacy) dosyaları da değerlendirebilir.

---

## 11. Öğrenci uygulamaları

### Görev 1 — Output budget ve truncation

`output_limit_sweep` çalıştırın. Ölçüm sonuçlarından tabloyu doldurun:

| Condition | Limit | Ortalama Completion Tokens | Ortalama Reasoning Tokens | Truncated / Calls | Accepted / Calls |
|---|---:|---:|---:|---:|---:|
| `restrictive` | 64 | | | | |
| `moderate` | 256 | | | | |
| `generous` | 1024 | | | | |

**Değerlendirin:** En düşük completion sınırında beş gerekli alanın tamamı üretildi mi? `finish_reason` neydi? Limit yükseldikçe gerçekten daha fazla token kullanıldı mı?

### Görev 2 — Completeness vs correctness

`output_limit_measurements.csv` üzerinden `present_field_count` ile `correct_field_count` arasındaki farkları inceleyin.

Şu soruyu yanıtlayın:

> Bütün alanların mevcut olması, cevabın bütününün doğru olduğu anlamına gelir mi?

Örneğin alanların hepsi mevcut fakat birinin değeri yanlışsa `field_completeness_ratio` ile `field_correctness_ratio` nasıl ayrılır?

### Görev 3 — Reasoning overhead

Her limit için şu verileri karşılaştırın:

```text
completion_tokens
reasoning_tokens
non_reasoning_completion_tokens
finish_reason
```

Hangi koşullarda reasoning tüketimi, görünür/nihai çıktı için kalan completion bütçesini daraltmış olabilir? Bu çıkarımın sınırlarını belirtin.

### Görev 4 — Invocation ekonomisi

`invocation_comparison` çalıştırın ve tabloyu doldurun:

| Metrik | Single Invocation | Multiple Invocations |
|---|---:|---:|
| API çağrı sayısı | 1 | 3 |
| Total Prompt Tokens | | |
| Total Completion Tokens | | |
| Total Estimated Cost | | |
| Workflow Wall-Clock (ms) | | |
| All Answers Accepted | | |

**Tartışın:** Aynı üç soruyu üç çağrı yerine bir çağrıda çözmenin olası maliyet avantajı nedir? Hangi kalite veya uygulama gereksinimleri çoklu çağrıları haklı kılabilir?

### Görev 5 — Quality-Constrained Optimization

`analysis_report.md` içindeki kalite karşılaştırmasını inceleyin.

Sonuçlarınızdan birini seçerek iki ayrı karar yazın:

- **Cost-only decision:** Yalnızca en düşük tahmini maliyete bakınca hangi stratejiyi seçerdiniz?
- **Quality-constrained decision:** Yalnızca bütün gerekli cevaplar kabul edilirse hangi stratejiyi seçerdiniz?

İki karar farklıysa nedenini açıklayın.

### Görev 6 — Deneyin sınırlamaları

Aşağıdaki etkenlerin sonuçları neden değiştirebileceğini açıklayın:

- Provider yükü ve network latency
- Run-to-run reasoning token değişkenliği
- Tek/çoklu çağrı stratejilerinin farklı cevap formatları
- Tekrar sayısının düşük olması
- Output limitinin kalitenin tek belirleyicisi olmaması
- Operational retry ve API rate limit

---

## 12. Sınıf içi tartışma soruları

1. `max_completion_tokens=1024` yazmak 1024 output token üretileceği anlamına gelir mi?
2. `completion_tokens == max_completion_tokens` tek başına truncation kanıtı mıdır?
3. `finish_reason="stop"` olsa da cevap neden eksik veya yanlış olabilir?
4. Reasoning modelinde `reasoning_tokens` maliyete nasıl yansır?
5. Completeness ve correctness arasındaki fark nedir?
6. Bir API çağrısının başarılı olması, iş açısından başarılı sonuç üretildiğini kanıtlar mı?
7. Birleşik prompt ile üç ayrı prompt'un içerikleri birebir aynı mı? Bu durum karşılaştırmayı nasıl etkiler?
8. Üç sequential çağrı, aynı işi yapan bir çağrıdan neden daha yüksek workflow latency oluşturabilir? Bu durum kesin midir?
9. Çoklu invocation bazen hangi nedenlerle tercih edilebilir? (Hata izolasyonu, bağımsız retry, farklı modeller, ayrı yetkilendirme vb.)
10. Üretimde maliyet optimizasyonu yaparken neden bir quality gate tanımlamalıyız?

---

## 13. Troubleshooting

### `LiveExecutionNotAllowed`

Gerçek çağrılar kapalıdır. `--live` parametresini ve `config/live.toml` içinde `live_api_enabled=true` değerini kontrol edin. `.env` API key'inizin mevcut olduğundan emin olun.

### `config/live.toml` bulunamadı

```powershell
Copy-Item config/default.toml config/live.toml
docker compose build lab-runner
```

Daha sonra `[execution]` bölümündeki ilgili boolean alanı etkinleştirin ve yeniden build edin.

### `planned calls exceed lab04.max_api_calls`

Default bütçe **10 mantıksal çağrıdır**. `--repetitions 3` ile `all` denemek 13 çağrı planlar. Pilot ya da varsayılan kapsamı kullanın.

### HTTP 429 / Rate Limit

Groq API limitinize ulaşmış olabilirsiniz. Aynı komutu art arda tekrarlamayın; hesap kotasını kontrol edip daha küçük deneye dönün.

### `finish_reason="length"`

Bu, çıktının limit nedeniyle sonlandığını gösterir. Beş gerekli alanın tümü üretilip üretilmediğini ayrıca kontrol edin. Model budget'ının tamamı reasoning'e ayrılmış olabilir.

### Metriklerde `N/A` veya `None`

İlgili usage / fiyat bilgisi raporlanmamış, geçersiz veya eksik olabilir. Böyle bir alanı otomatik olarak sıfır saymayın.

### Grafik üretilemedi

Önce JSON/CSV ölçümlerinin oluştuğundan emin olun. Ardından `backfill` kullanın.

### Aynı `--experiment-run-id` ile tekrar çalışmıyor

LAB04, mevcut ölçümlerin üzerine yazılmasını engeller. Yeni run ID ile tekrar çalıştırın.

### `invocation_comparison.csv` yok

Yalnızca `output_limit_sweep` çalıştırılmış olabilir veya invocation karşılaştırmasını üretecek yeterli ölçüm oluşmamış olabilir. Manifest ve `experiment_summary.json` durumunu kontrol edin.

---

## 14. LAB04 diğer laboratuvarlarla nasıl ilişkili?

| Laboratuvar | LAB04 ile ilişkisi |
|---|---|
| LAB01 — Token Anatomy | Input, completion, reasoning tokenları ve cost temeli |
| LAB02 — Context Growth | Input context maliyeti; LAB04'te output bütçesi odağı |
| LAB03 — Prompt Caching | Tekrarlanan input maliyetini yönetme; LAB04 invocation sayısına odaklanır |
| LAB08 — Concurrency | Sequential ile concurrent workflow farkını ayrıca inceler |
| LAB10 — Summarization | Summary oluşturma overhead ve downstream cevap kalitesinin ölçülmesi |

**LAB04'ün temel mesajı:** Output token sınırını küçültmek veya API invocation sayısını azaltmak, ancak **doğru ve eksiksiz sonucu koruduğunda** gerçek bir optimizasyondur.

## 15. Kaynak koddan öğrenme

İleri inceleme için:

```text
src/labs/lab04_output_control/
    cli.py          # CLI koşulları, dry-run, live guard
    experiments.py  # Deney soruları ve koşullar
    evaluation.py   # Deterministic field completeness / correctness
    matrix.py       # Planlanan çağrı matrisi
    runner.py       # Groq çağrıları ve workflow timing
    models.py       # Kayıt edilen metrikler
    stats.py        # Koşul ve strateji özetleri
    persistence.py  # JSON / CSV / manifest dosyaları
    report.py       # Markdown analysis report
    charts.py       # PNG grafikler
    charts_cli.py   # Offline backfill
```

Varsayılan parametreleri görmek için `config/default.toml` içindeki `[lab04]` bölümüne bakın.

---

## 16. Kapanış — Beklenen öğrenme çıktısı

LAB04 sonunda ölçümlerle destekleyerek şu iki cümleyi açıklayabilmelisiniz:

> **Output control:** En küçük `max_completion_tokens` her zaman en iyi seçenek değildir; reasoning overhead, truncation ve output completeness birlikte değerlendirilmelidir.

> **Invocation optimization:** Daha az API çağrısı, aynı görevin toplam token/maliyet/latency yükünü azaltabilir; ancak sonuç kalitesi sağlanmadan bu yaklaşım başarılı kabul edilemez.

**Not:** Bu rehberdeki formüller ve örnek sayılar öğreticidir. Kendi experiment run'ınızdaki gerçek değerleri `results/lab04_output_control/<experiment_run_id>/` içindeki ölçümlerden alın.
