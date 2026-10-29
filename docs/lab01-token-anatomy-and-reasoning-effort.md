# LAB01 — Token Anatomy, Cost Calculation & Reasoning Effort

> **Katılımcı rehberi** · Python / Groq API / Docker Compose\
> **Kapsam:** Basic Mode (`short_prompt`, `long_prompt`) ve Reasoning Effort Comparison (`effort`).\
> **Amaç:** API'nin raporladığı token türlerini doğru yorumlamak, yaklaşık çağrı maliyetini hesaplamak ve reasoning effort'un token tüketimi, latency ve maliyetle ilişkisini kontrollü deneylerde gözlemlemek.

## 1. Bu laboratuvarda ne öğreneceksiniz?

Bir LLM API çağrısı yalnızca modele gönderdiğiniz metinden ibaret değildir. Model ayrıca bir completion üretir ve desteklenen modellerde bu completion'ın bir kısmı **reasoning tokens** olabilir. Faturalandırma, modelin gerçek token kullanımına ve sağlayıcının fiyatlandırma kurallarına bağlıdır.

LAB01 iki temel soruyu inceler:

1. **Token Anatomy:** Kısa ve uzun görevlerde `prompt_tokens`, `completion_tokens`, `reasoning_tokens` ve maliyet nasıl değişiyor?
2. **Reasoning Effort:** Aynı görev için `low`, `medium`, `high` seviyelerini kullandığımızda reasoning tokenları, toplam completion, latency ve maliyet nasıl değişiyor?

**Önemli:** Daha yüksek reasoning effort'un her istekte daha fazla token kullanacağı veya daha kaliteli cevap üreteceği garanti değildir. Bu laboratuvarın amacı varsayımı tekrar etmek değil, gerçekleşen sonuçları ölçmektir.

## 2. İki deney türü arasındaki fark

| Özellik | Basic Mode | Reasoning Effort Mode |
|---|---|---|
| CLI | `lab01 all` | `lab01 effort` |
| Değişen faktör | Kısa/uzun görev ve prompt | `reasoning_effort` |
| Prompt türleri | `short_prompt`, `long_prompt` | Varsayılan: `short_prompt`; isteğe bağlı `long_prompt` |
| Reasoning effort | API'ye özellikle gönderilmez | `low`, `medium`, `high` |
| Tekrar sayısı | Varsayılan 5 / prompt | Varsayılan 2 / condition |
| Varsayılan mantıksal çağrı | 10 | 6 |
| `--dry-run` desteği | Yok | Var |
| Sonuçlar | Temel JSON/CSV/summary ve grafikler | Run-specific JSON/CSV/rapor ve grafikler |

**Provider:** Groq API\
**Varsayılan model:** `openai/gpt-oss-20b` (`config/default.toml` üzerinden)\
**Not:** `reasoning_effort` değerleri bu laboratuvarda Groq üzerindeki desteklenen GPT-OSS modelleriyle kullanılır; model değiştirirseniz parametre uyumluluğunu kontrol edin.

### 2.1 Basic Mode — Hangi promptlar kullanılıyor?

Kaynak kod: `src/labs/lab01_token_anatomy/experiments.py`

- `short_prompt`: **“What is the capital of France?”**
- `long_prompt`: Token-based LLM API pricing konusunu en az dört paragrafta açıklama görevi.

İki prompt yalnızca uzunluk açısından değil, **istenen cevap kapsamı** açısından da farklıdır. Bu nedenle gözlenen completion-token farkı yalnızca input prompt uzunluğunun nedensel etkisi olarak yorumlanmamalıdır.

### 2.2 Reasoning Effort — Kontrol edilen değişken

Effort modunda aynı prompt, üç farklı seviyede çalıştırılır:

| Condition | Prompt | Reasoning effort |
|---|---|---|
| `short_prompt__low` | Aynı kısa prompt | `low` |
| `short_prompt__medium` | Aynı kısa prompt | `medium` |
| `short_prompt__high` | Aynı kısa prompt | `high` |

`temperature` ve `top_p` bu deneyin bağımsız değişkenleri değildir: İlgili API isteğine **gönderilmezler**; provider defaults geçerlidir. Varsayılan değerlerinin mutlaka sıfır veya bir olduğunu varsaymayın.

## 3. Kurulum — Windows PowerShell ve Docker

Gereksinimler: Docker Desktop + Docker Compose, proje repository'si ve **live** deneyler için geçerli Groq API anahtarı.

Komutları repository kök dizininde çalıştırın:

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Bilgisayarınızda repository başka bir dizindeyse yolu değiştirin.

### 3.1 Groq API anahtarını yapılandırma

Repository'deki `.env.example` dosyasını `.env` olarak kopyalayın:

```powershell
Copy-Item .env.example .env
```

Ardından `.env` dosyasına kendi `GROQ_API_KEY` değerinizi ekleyin. **API anahtarınızı GitHub'a göndermeyin ve sınıf içi ekran görüntülerinde paylaşmayın.**

### 3.2 Live execution için TOML dosyası

Güvenli varsayılan konfigürasyonda gerçek API çağrıları kapalıdır. Kendi live dosyanızı oluşturun:

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` dosyasında `[execution]` altında yalnızca ilgili değeri değiştirin:

```toml
[execution]
live_api_enabled = true
```

Diğer TOML bölümlerini silmeyin. `config/live.toml` dosyasını Git'e commit etmeyin.

**Önemli:** Hem `--live` argümanı hem de `execution.live_api_enabled = true` gereklidir. Biri eksikse gerçek Groq API çağrısı başlamamalıdır.

### 3.3 Docker image'ını oluşturma

```powershell
docker compose build lab-runner
```

Projede `config/` ve kaynak kod Docker image'ına build aşamasında kopyalandığı için konfigürasyon veya LAB01 kodları değiştiğinde image'ı yeniden oluşturun.

## 4. Docker komutları — Basic Mode

### A. LAB01 yardım ekranı (API çağrısı yok)

```powershell
docker compose run --rm lab-runner lab01 --help
```

### B. Kısa prompt deneyini çalıştırma

```powershell
docker compose run --rm lab-runner lab01 short_prompt --live --config config/live.toml
```

Varsayılan konfigürasyonda **5 mantıksal API çağrısı** gerçekleştirir.

### C. Uzun prompt deneyini çalıştırma

```powershell
docker compose run --rm lab-runner lab01 long_prompt --live --config config/live.toml
```

Varsayılan konfigürasyonda **5 mantıksal API çağrısı** gerçekleştirir.

### D. Basic Mode — İki prompt birlikte

```powershell
docker compose run --rm lab-runner lab01 all --live --config config/live.toml
```

Varsayılan konfigürasyonda **5 kısa + 5 uzun = 10 mantıksal API çağrısı** yapar.

### E. Daha küçük Basic Mode çalıştırması

```powershell
docker compose run --rm lab-runner lab01 all --repetitions 2 --live --config config/live.toml
```

Bu komut, her promptu iki kez çalıştırır: **2 × 2 = 4 mantıksal çağrı**.

> **Maliyet uyarısı:** Bu bölümdeki `--live` komutlarının tümü gerçek Groq çağrısı yapar. Basic Mode için ayrı bir `--dry-run` uygulaması bulunmadığından güvenli bir plan önizlemesi istiyorsanız `lab01 --help` ve konfigürasyonu inceleyin; canlı komutu yalnızca deney yapacağınız zaman çalıştırın.

## 5. Docker komutları — Reasoning Effort Comparison

Öğrencilerin token/reasoning farklarını karşılaştırması için **önerilen LAB01 deneyi** bu moddur.

### A. API çağrısı yapmadan planı görüntüleme

```powershell
docker compose run --rm lab-runner lab01 effort --dry-run --config config/default.toml
```

Bu komut bir deney planı oluşturur, **gerçek API çağrısı yapmaz**. Varsayılan plan **6 mantıksal çağrıdır**.

### B. Pilot — 3 çağrı

```powershell
docker compose run --rm lab-runner lab01 effort --repetitions 1 --live --config config/live.toml
```

Hesap: **1 prompt × 3 effort × 1 repetition = 3 çağrı**.

### C. Default Grand Test — 6 çağrı

```powershell
docker compose run --rm lab-runner lab01 effort --live --config config/live.toml
```

Hesap: **1 prompt × 3 effort × 2 repetitions = 6 çağrı**.

### D. Extended — 12 çağrı

```powershell
docker compose run --rm lab-runner lab01 effort --prompt short_prompt long_prompt --live --config config/live.toml
```

Hesap: **2 prompt × 3 effort × 2 repetitions = 12 çağrı**.

Bu deneyin varsayılan `max_api_calls` üst sınırı 12'dir. CLI ve live configuration değerlerinin bu kapsamla uyumlu olduğundan emin olun.

### E. Sadece Low ve Medium karşılaştırması

```powershell
docker compose run --rm lab-runner lab01 effort --reasoning-effort low medium --live --config config/live.toml
```

Varsayılan kısa prompt ve iki tekrar ile **2 effort × 2 repetitions = 4 çağrı**.

### F. Özel deney kimliğiyle çalıştırma

```powershell
docker compose run --rm lab-runner lab01 effort --experiment-run-id class-lab01-group-a --live --config config/live.toml
```

Effort Mode, kaydedilen ölçümler ve konfigürasyon uyumluysa aynı run ID üzerinden kaldığı yerden devam etmeyi destekler. Bu seçeneği **yeni bir deney** oluşturmak için tekrar kullanmayın: tamamlanmış çağrılar yeniden yapılmayabilir. Yeni çalıştırma için yeni run ID belirleyin.

### G. Kullanımdan kaldırılan `factorial` modu

Eski LAB01'deki `temperature × top_p × reasoning_effort` tam faktöriyel deney **emekliye ayrılmıştır**. Eski `lab01 factorial` komutu, güvenlik nedeniyle API çağrısı yapmadan bilgilendirme gösterir.

**Bunu deney olarak çalıştırmayın.** Reasoning karşılaştırması için `lab01 effort` kullanın.

## 6. LAB01 gerçekte neyi ölçüyor?

Bir çağrıda şu akış gerçekleşir:

```text
User Prompt
    ↓
Groq Chat Completions API
    ↓
Model Completion
    ↓
Provider Usage + Client-Side Timing
    ↓
Token Breakdown + Estimated Cost
    ↓
JSON / CSV / Charts
```

### 6.1 Input token ve output token

- **`prompt_tokens`**: Provider'ın input request için raporladığı token miktarıdır. Chat request message formatting gibi etkileri içerebilir.
- **`completion_tokens`**: Modelin tamamladığı yanıt üretimine ait toplam completion token miktarıdır; raporlandığı yerde reasoning tokenları buna dahildir.
- **`total_tokens`**: Provider'ın raporladığı toplam token kullanımıdır. Normal koşullarda prompt + completion toplamıyla uyumlu olmalıdır.
- **`reasoning_tokens`**: Provider tarafından ayrı raporlanıyorsa completion içindeki reasoning için kullanılan token sayısıdır. Son kullanıcıya gösterilen yanıtın tamamı demek değildir.
- **`non_reasoning_completion_tokens`**: Her iki alan biliniyorsa `completion_tokens − reasoning_tokens` biçiminde türetilen metrik. Bu değeri kesin olarak “kullanıcıya görünür output tokens” şeklinde adlandırmayın.
- **`cached_tokens`**: Raporlanmışsa tekrar kullanılan input tokenlarının sayısıdır. Bu alanın eksik olması `0` olduğunu kanıtlamaz.

### 6.2 Reasoning tokens neden ayrıca toplanmıyor?

Örnek (temsili değerler):

```text
prompt_tokens:                     200
completion_tokens:                 300
reasoning_tokens:                  220
non_reasoning_completion_tokens:    80
total_tokens:                      500
```

Doğru hesap:

```text
Total = Prompt + Completion = 200 + 300 = 500
```

Yanlış hesap:

```text
Total = Prompt + Completion + Reasoning = 720   ← double counting
```

Reasoning tokens, zaten completion tokenlarının **içinde** raporlandığı için ikinci kez eklenmez.

### 6.3 Maliyet nasıl hesaplanıyor?

LAB01 `config` dosyasında tanımlanan model fiyatlarından **tahmini maliyet** üretir.

Basitleştirilmiş, cache-read fiyatını da ayıran formül:

\[
C = \frac{T_{uncached}P_{input} + T_{cached}P_{cached} + T_{completion}P_{output}}{1{,}000{,}000}
\]

Burada fiyatlar USD / 1 milyon token cinsindendir. İlgili çağrıda cached token verisi mevcut değilse cache etkisi kesin olarak değerlendirilemez. LAB01 maliyet sonuçlarını **provider'ın nihai faturası** olarak kabul etmeyin; güncel model pricing ve kullanım detaylarına bağlı tahminlerdir.

### 6.4 Latency ve finish reason

- **`client_elapsed_ms` / `total_latency_ms`**: İsteğin istemci tarafından ölçülen süresi. Ağ gecikmesi ve provider processing etkilerini birlikte içerir.
- **`finish_reason=stop`**: Model doğal şekilde tamamlamış olabilir.
- **`finish_reason=length`**: Completion bütçesi veya bir uzunluk sınırı nedeniyle erken sonlandırma ihtimali. Özellikle effort karşılaştırmasında sonuçların geçerliliğini etkileyebilir.
- **`final_answer_char_count`**: Üretilen son yanıtın karakter sayısı. Token sayısının yerine geçmez.

**Dikkat:** Yüksek reasoning effort nedeniyle completion tokenlarının önemli kısmı reasoning'e ayrılırsa görünür cevabın uzunluğu aynı kalırken maliyet değişebilir.

## 7. Öğrenci olarak terminal çıktısını nasıl yorumlamalısınız?

### 7.1 Basic Mode çıktısı

Bu alanları arayın:

```text
Run 1/5
  prompt_tokens:        ...
  completion_tokens:    ...
  reasoning_tokens:     ...
  non_reasoning_tokens: ...
  finish_reason:        ...
  estimated_cost_usd:   ...
```

Her condition özetinde şu başlıklar bulunur:

- Calls
- Prompt Tokens per Call
- Completion Tokens: minimum, maximum, mean, standard deviation, range
- Reasoning Tokens: mean, available measurements
- Finish Reasons

**Yorum:** Uzun prompt koşulu daha yüksek completion token ürettiyse bu, görevde daha uzun cevap istendiği için de olabilir. Tek başına “uzun prompt her zaman daha çok output token üretir” iddiasına dönüşmemelidir.

### 7.2 Effort Mode çıktısı

```text
Effort: low (prompt: short_prompt)
Run: 1/2
Prompt Tokens:        ...
Completion Tokens:    ...
Reasoning Tokens:     ...
Non-Reasoning Tokens: ...
Latency:              ... ms
Estimated Cost:       $...
Finish Reason:        ...
```

Koşul özetinde şunları karşılaştırın:

- Mean Reasoning Tokens
- Mean Completion Tokens
- Mean Latency
- Mean Estimated Cost
- Başarılı ve başarısız çağrı sayıları

Özellikle `high` effort'un `low` effort'a göre gerçekten daha fazla reasoning token kullanıp kullanmadığını **ölçümlere bakarak** belirleyin; peşinen varsaymayın.

### 7.3 Sample size uyarısı

Default Effort Mode'da her condition **yalnızca iki kez** ölçülür. Basic Mode'da varsayılan olarak beş tekrar vardır.

Bu küçük örneklemler eğitim açısından yeterli olabilir; fakat p95, varyans veya model performansı hakkında güçlü istatistiksel genelleme yapılamaz.

Bir çıktı eksikse `None` / `N/A` ile gösterilmelidir; eksik provider ölçümünü sıfır olarak yorumlamayın.

## 8. Sonuç dosyalarını ve grafikleri bulma

### 8.1 Basic Mode

Varsayılan dosyalar:

```text
results/
├── lab01_token_anatomy.json
├── lab01_token_anatomy.csv
├── lab01_token_anatomy_summary.json
└── lab01_token_anatomy/
    └── charts/
```

**Dikkat:** Basic Mode CSV çıktısı mevcut dosyaya kayıt ekleyebilir; farklı çalıştırmaların satırları aynı CSV'de birikebilir. JSON ve summary dosyaları aynı output adı kullanılırsa yenilenebilir. Temiz karşılaştırma için `--output-name` kullanın veya önceki sonuçları koruyun.

Örnek:

```powershell
docker compose run --rm lab-runner lab01 all --repetitions 2 --output-name lab01_class_a --live --config config/live.toml
```

### 8.2 Effort Mode

Sonuçlar her çalıştırma için ayrı klasörde tutulur:

```text
results/
└── lab01_effort/
    └── <run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── raw_measurements.json
        ├── raw_measurements.csv
        ├── condition_summary.csv
        ├── effort_summary.json
        ├── analysis_report.md
        └── charts/
```

Grafikler arasında `reasoning_vs_non_reasoning_tokens.png`, `completion_tokens_by_effort.png`, `latency_by_effort.png`, `estimated_cost_by_effort.png`, `reasoning_ratio_by_effort.png` ve `summary_dashboard.png` bulunabilir.

PowerShell'de en son Effort Mode sonuçlarını açmak için:

```powershell
$latest = Get-ChildItem .\results\lab01_effort -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$latest.FullName
Get-ChildItem $latest.FullName
```

Analiz raporunu okumak için:

```powershell
Get-Content (Join-Path $latest.FullName 'analysis_report.md') -Encoding UTF8
```

### 8.3 Mevcut sonuçlardan grafikleri yeniden oluşturma (offline)

```powershell
docker compose run --rm lab-runner lab01 backfill --effort-only --config config/default.toml
```

Bu komut mevcut Effort Mode sonuçlarından grafikleri tekrar oluşturur; **Groq API çağrısı yapmaz**. Belirli bir run için `--run-id` kullanılabilir. Basic Mode grafiklerinin üretimi, canlı ölçüm tamamlandığında otomatik yapılır; bu `backfill` komutu Basic Mode klasörlerini hedeflemez.

## 9. Sınıf içi uygulama

### Görev A — Token Anatomy

1. `lab01 --help` komutuyla geçerli execution modes'u bulun.
2. Eğitmenin onayıyla kısa ve uzun prompt deneylerini çalıştırın.
3. Her condition için `prompt_tokens`, `completion_tokens` ve `reasoning_tokens` değerlerini karşılaştırın.
4. `completion_tokens` içinde reasoning tokenlarını ikinci kez saymadığınızı formülle gösterin.
5. İki prompt için `estimated_cost_usd` farkının nereden geldiğini açıklayın.

### Görev B — Reasoning Effort

1. `lab01 effort --dry-run` komutuyla üç effort seviyesinin deney planını inceleyin.
2. Default 6-call deneyi çalıştırın.
3. `low`, `medium`, `high` için reasoning tokens, latency ve estimated cost karşılaştırması yapın.
4. `finish_reason=length` varsa bunun sonucu nasıl etkileyebileceğini açıklayın.
5. Hiçbir condition'ın her ölçümde en iyi olmak zorunda olmadığını örnekle tartışın.

### Tartışma soruları

- `completion_tokens` ile `reasoning_tokens` neden toplanmamalı?
- `prompt_tokens` aynı olmasına rağmen `completion_tokens` neden değişebilir?
- Reasoning effort arttığında latency her zaman artar mı?
- Kısa bir soru için `high` effort hangi koşullarda gereksiz olabilir?
- `non_reasoning_completion_tokens`, modelin kullanıcıya gösterdiği metnin kesin token sayısı mıdır?
- Aynı prompt için iki tekrar, güvenilir bir genel sonuç çıkarmaya yeterli midir?
- Maliyet hesabında cached token alanı eksikse hangi belirsizlik oluşur?

## 10. Sık karşılaşılan sorunlar

| Durum | Kontrol edilecek nokta |
|---|---|
| `lab01` tanınmıyor | Repository'nin güncel olduğundan ve Docker image'ının yeniden build edildiğinden emin olun. |
| Live execution not allowed | `--live` kullandığınızı, `config/live.toml` içinde `live_api_enabled = true` olduğunu ve image'ı yeniden build ettiğinizi kontrol edin. |
| `GROQ_API_KEY` bulunamadı | `.env` dosyasını ve Compose environment aktarımını kontrol edin. API anahtarını paylaşmayın. |
| HTTP 429 | Groq kotası/rate limit; arka arkaya kör tekrar çalıştırmayın. Retry politikası ve kotayı kontrol edin. |
| `--dry-run` Basic Mode'da beklenen sonucu vermiyor | Bu seçenek `effort` için uygulanır. Basic Mode'un ayrı dry-run planlayıcısı yoktur. |
| `factorial` çalışmıyor | Bu eski deney bilinçli olarak devre dışı bırakılmıştır; `lab01 effort` kullanın. |
| `reasoning_tokens = N/A` | Provider o çağrı için alanı raporlamamış olabilir; bunu otomatik `0` saymayın. |
| `finish_reason = length` | Model completion limitine takılmış olabilir; deneyi dikkatle yorumlayın. |
| Grafikler görünmüyor | Terminalde chart generation mesajını ve ilgili `results/` dizinini kontrol edin. |
| `max_api_calls` sınırı aşıldı | Effort configuration ve seçilen prompt/effort/repetition kombinasyonlarını kontrol edin. |

## 11. Güvenlik ve maliyet

- `--live` gerçek Groq çağrısı başlatır; token kotası ve API maliyeti doğabilir.
- `--dry-run`, Effort Mode deney planını kontrol etmek için kullanılmalıdır.
- `GROQ_API_KEY`, `.env` ve `config/live.toml` içeriğini GitHub'a eklemeyin.
- Raw model yanıtları LAB01'in mevcut sonuç kayıtlarında genellikle saklanmaz; temel olarak usage/metadata, char count ve özet ölçümleri kaydedilir. LAB10'daki tam sentetik input/output inspection davranışını LAB01'de varsaymayın.
- Model pricing zaman içinde değişebilir. `estimated_cost_usd` değeri, konfigürasyondaki fiyatlar kullanılarak hesaplanan tahmini maliyettir.
- Herhangi bir düzeyin otomatik olarak daha hızlı, daha ucuz veya daha kaliteli olduğuna dair evrensel sonuç çıkarılmamalıdır.

---

### Öğrenme çıktısı

LAB01'i tamamlayan öğrenci, **input, completion, reasoning ve cached tokens** kavramlarını birbirinden ayırabilmeli; reasoning tokenlarını iki kez saymadan maliyet hesabını açıklayabilmeli; aynı görevin farklı reasoning effort seviyelerinde token tüketimi, latency ve maliyet açısından nasıl karşılaştırılacağını gösterebilmelidir.

### Kaynak kod ve ileri okuma

- [`LAB01 Basic experiments`](../../src/labs/lab01_token_anatomy/experiments.py)
- [`LAB01 Reasoning Effort`](../../src/labs/lab01_token_anatomy/effort_cli.py)
- [`Default configuration`](../../config/default.toml)
- [Groq API Documentation](https://console.groq.com/docs/overview)
