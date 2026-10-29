# LAB02 — Context Growth & Context Bloat

> **Katılımcı rehberi** · Python / Groq API / Docker Compose\
> **Kapsam:** `minimal_context`, `expanded_context`, `irrelevant_context` ve `multi_turn_growth`.\
> **Amaç:** Context boyutu ve alaka düzeyinin input token tüketimi, maliyet, latency ve cevap doğruluğuyla ilişkisini ölçmek; çok turlu konuşmalarda cumulative token tüketiminin nasıl büyüdüğünü gözlemlemek.

## 1. Bu laboratuvarda ne öğreneceksiniz?

LLM API'sine gönderilen her istek bir **context** içerir: talimatlar, görevle ilgili bilgiler, kullanıcının sorusu ve gerektiğinde önceki konuşmalar. Context büyüdükçe modele gönderilen input token miktarı genellikle artar; bunun maliyet ve latency üzerinde etkileri olabilir.

LAB02 iki ayrı problemle ilgilenir:

1. **Context Growth & Context Bloat:** Aynı soruya farklı boyut ve alaka düzeyindeki context'lerle cevap verildiğinde neler değişiyor?
2. **Multi-Turn Growth:** Chatbot'un önceki kullanıcı ve asistan mesajlarını her istekte tekrar göndermesi, tekil ve birikimli input token tüketimini nasıl etkiliyor?

**Önemli:** Daha büyük context'in her zaman daha kötü cevap verdiği veya her zaman daha yüksek latency oluşturduğu iddia edilmez. Deney gerçek API kullanımını ölçer; sonuçlar gözlemlenen koşullara göre yorumlanmalıdır.

### 1.1 Araştırma soruları

- Context büyüdükçe `prompt_tokens` nasıl değişiyor?
- İlgili bilgiyi koruyup ilgisiz içerik eklediğimizde ne oluyor?
- Aynı sorunun doğruluğu farklı context'lerde korunuyor mu?
- Context boyutu ve `reasoning_tokens` arasındaki ilişki nedir?
- Çok turlu konuşmada `cumulative_prompt_tokens` neden hızla artabiliyor?
- Token tüketimi, latency ve tahmini maliyet arasında nasıl bir trade-off var?

## 2. LAB02 deney mimarisi

LAB02 iki bölümden oluşur.

| Bölüm | CLI koşulları | Kontrol edilen durum |
|---|---|---|
| Context Comparison | `minimal_context`, `expanded_context`, `irrelevant_context` | Aynı soru, farklı context boyutu/alakası |
| Multi-Turn Growth | `multi_turn_growth` | Önceki konuşmaların tekrar gönderildiği artan history |

**Provider:** Groq API\
**Varsayılan model:** `openai/gpt-oss-20b` (`config/default.toml`)\
**Çalıştırma yaklaşımı:** Gerçek API çağrıları; sonuçlar JSON, CSV, Markdown rapor ve grafikler halinde kaydedilir.

### 2.1 Context Comparison — Üç koşul

Üç koşulda da kaynak kodun sorduğu **aynı soru** kullanılır:

> According to the policy, what is the maximum number of retry attempts, excluding the initial attempt?

Cevabın dayandığı policy:

```text
Maximum total attempts: 5
Initial attempt: 1
Maximum retry attempts: 4
```

**Beklenen çıktı: `4`**

| Condition | API'ye gönderilen context | Eğitim amacı |
|---|---|---|
| `minimal_context` | Kısa ve doğrudan ilgili retry policy | Baseline |
| `expanded_context` | Aynı temel bilgi + ilgili, uzun teknik açıklamalar | Relevant context growth |
| `irrelevant_context` | İlgisiz ofis/facilities açıklamaları + temel retry policy | Context bloat |

Üç koşul için sistem talimatı, kullanıcı sorusu, beklenen cevap ve cevap formatı aynıdır. Modelden yalnızca tam sayı döndürmesi istenir.

**Önemli deney sınırlaması:** Koşullar yalnızca token sayısı bakımından farklı değildir. `irrelevant_context` koşulunda temel bilgi context'in sonunda; `expanded_context` koşulunda başındadır. Bu nedenle sonuçlar tek başına “yalnızca context büyüklüğü gecikmeye neden oldu” şeklinde yorumlanamaz. Bilgi konumu, içerik yapısı, reasoning varyasyonu ve provider yükü de etkili olabilir.

### 2.2 Multi-Turn Growth — Konuşma geçmişinin büyümesi

Bu deneyde başlangıçta kısa bir `system` mesajı bulunur. Her turda yeni kullanıcı sorusu eklenir, API çağrısı yapılır, model cevabı history'ye eklenir ve bir sonraki turda geçmiş yeniden gönderilir.

Örnek akış:

```text
Turn 1 request:
  System
  User 1

Turn 2 request:
  System
  User 1
  Assistant 1
  User 2

Turn 3 request:
  System
  User 1
  Assistant 1
  User 2
  Assistant 2
  User 3
```

Kaynak kodda her turda kullanılan soru şablonu:

```text
Tell me a short, distinct fact about topic #<turn>.
```

Bu bölümün hedefi, cevap kalitesini değerlendirmekten çok **yeniden gönderilen conversation history'nin token ekonomisini** göstermektir.

LAB02, burada history'yi özetlemez veya kırpmaz. Conversation Summarization ayrı olarak LAB10'un konusudur.

## 3. Kurulum — Windows PowerShell ve Docker

Gereksinimler: Docker Desktop, Docker Compose, repository ve canlı deneyler için Groq API anahtarı.

Komutları repository kök klasöründe çalıştırın:

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Repository'nizin konumu farklıysa yolu değiştirin.

### 3.1 Groq API anahtarını ayarlama

Proje kökünde:

```powershell
Copy-Item .env.example .env
```

`.env` dosyasına kendi Groq API anahtarınızı ekleyin:

```dotenv
GROQ_API_KEY=your_api_key_here
```

**API anahtarınızı GitHub'a göndermeyin, ekran görüntülerinde göstermeyin ve sınıf arkadaşlarınızla paylaşmayın.**

### 3.2 Live konfigürasyonu hazırlama

Güvenli varsayılan dosyada gerçek API çağrıları kapalıdır. Canlı deneyler için kopya oluşturun:

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` içerisindeki `[execution]` bölümünde:

```toml
[execution]
live_api_enabled = true
```

ayarını yapın. **Dosyanın diğer bölümlerini silmeyin.**

Gerçek API çağrısının başlaması için hem `--live` parametresi hem de `live_api_enabled = true` gerekir.

### 3.3 Docker image'ını oluşturma

```powershell
docker compose build lab-runner
```

Kaynak kod ve `config/` klasörü image'a build sırasında kopyalandığı için, `config/live.toml` oluşturulduktan veya proje kodları güncellendikten sonra image'ı yeniden build edin.

### 3.4 LAB02 yardım ekranı

```powershell
docker compose run --rm lab-runner lab02 --help
```

Bu komut API çağrısı yapmaz ve desteklenen parametreleri gösterir.

## 4. Docker komutları — Deneyleri çalıştırma

### A. Default Grand Test — Önerilen sınıf uygulaması

```powershell
docker compose run --rm lab-runner lab02 all --live --config config/live.toml
```

Varsayılan konfigürasyonda:

- `minimal_context`: 2 çağrı
- `expanded_context`: 2 çağrı
- `irrelevant_context`: 2 çağrı
- `multi_turn_growth`: 4 çağrı

**Toplam: 10 planlanmış mantıksal API çağrısı.**

Operasyonel retry nedeniyle gerçek HTTP denemelerinin sayısı mantıksal çağrı sayısından farklı olabilir. Başarısızlık veya bütçe sınırı nedeniyle planlanan çağrıların tamamı gerçekleşmeyebilir.

### B. API çağrısı yapmadan deney planını oluşturma

```powershell
docker compose run --rm lab-runner lab02 all --dry-run --config config/default.toml
```

Bu mod yalnızca planı oluşturur ve `experiment_plan.csv` dosyasını kaydeder. **Groq API çağrısı yapılmaz.**

`--dry-run` sonucunda gerçek `prompt_tokens`, latency veya maliyet ölçümü oluşmuş gibi düşünmeyin.

### C. Pilot — Daha küçük deney

```powershell
docker compose run --rm lab-runner lab02 all --repetitions 1 --turns 3 --live --config config/live.toml
```

Planlanan mantıksal çağrılar:

```text
3 context conditions × 1 repetition = 3
3 multi-turn calls                   = 3
Total                                = 6
```

Sınıfta ilk kontrol için kullanabilirsiniz.

### D. Yalnızca minimal context

```powershell
docker compose run --rm lab-runner lab02 minimal_context --live --config config/live.toml
```

Varsayılan: **2 mantıksal çağrı**.

### E. Yalnızca expanded context

```powershell
docker compose run --rm lab-runner lab02 expanded_context --live --config config/live.toml
```

Varsayılan: **2 mantıksal çağrı**.

### F. Yalnızca irrelevant context

```powershell
docker compose run --rm lab-runner lab02 irrelevant_context --live --config config/live.toml
```

Varsayılan: **2 mantıksal çağrı**.

### G. Yalnızca Multi-Turn Growth

```powershell
docker compose run --rm lab-runner lab02 multi_turn_growth --live --config config/live.toml
```

Varsayılan: **4 konuşma turu = 4 mantıksal çağrı**.

### H. Turn sayısını değiştirme

```powershell
docker compose run --rm lab-runner lab02 multi_turn_growth --turns 6 --live --config config/live.toml
```

Bu komut, **6 mantıksal çağrı** planlar. Standalone multi-turn deneyinin varsayılan çağrı bütçesi içine sığar.

**Dikkat:** Aşağıdaki komutta 3 context condition × 2 tekrar + 6 turn = 12 çağrı planlanır:

```powershell
docker compose run --rm lab-runner lab02 all --turns 6 --live --config config/live.toml
```

Varsayılan `lab02.max_api_calls = 10` olduğundan bu kapsam **reddedilmelidir**. Limitleri sırf daha fazla çağrı yapmak için yükseltmeyin; önce öğretim hedefini ve mevcut API kotasını değerlendirin.

### I. Sonuç klasörüne özel isim verme

```powershell
docker compose run --rm lab-runner lab02 all --experiment-run-id class-lab02-group-a --live --config config/live.toml
```

Sonuç klasörü:

```text
results/lab02_context_growth/class-lab02-group-a/
```

LAB02, dolu bir run ID ile aynı ölçümleri tekrar eklemeyi reddeder. Yeni deneyde farklı bir run ID kullanın.

## 5. LAB02 hangi metrikleri ölçüyor?

### 5.1 Context Comparison metrikleri

| Metrik | Anlamı |
|---|---|
| `context_chars` | Fixture context metninin karakter uzunluğu; token değildir |
| `prompt_tokens` | Provider'ın tüm input request için raporladığı token sayısı |
| `completion_tokens` | Toplam completion tokenları; varsa reasoning dahildir |
| `reasoning_tokens` | Provider tarafından raporlanan reasoning tokenları |
| `non_reasoning_completion_tokens` | Uygunsa `completion_tokens - reasoning_tokens` |
| `total_tokens` | Provider'ın toplam token kullanımı |
| `cached_tokens` | Provider raporluyorsa cached input tokenları |
| `total_latency_ms` | İstemci tarafında ölçülen request gecikmesi |
| `estimated_cost_usd` | Konfigüre edilen token fiyatlarına göre tahmini maliyet |
| `finish_reason` | Modelin üretimi neden sonlandırdığı (`stop`, `length` vb.) |
| `answer_correct` | Beklenen cevapla eşleşme durumu |
| `answer_evaluation_status` | `correct`, `incorrect`, `format_invalid`, `missing_response` vb. |
| `attempts_used` | Operational retry dahil deneme sayısı |
| `succeeded` | API çağrısının başarıyla tamamlanması |

**Önemli fark:** `context_chars` yalnızca ilgili fixture'ın karakter sayısıdır. `prompt_tokens` ise system instruction, context, soru ve mesaj biçimlendirme etkileri dahil API'nin raporladığı input token sayısıdır.

API çağrısının başarılı olması, model cevabının doğru olduğu anlamına gelmez.

### 5.2 Reasoning tokens ve completion tokens

Temsili örnek:

```text
prompt_tokens:                    500
completion_tokens:                200
reasoning_tokens:                 180
non_reasoning_completion_tokens:   20
total_tokens:                     700
```

Burada reasoning tokenları, completion tokenlarının **içindedir**. `500 + 200 + 180` toplamayın. Bu örnekte toplam 700 tokendır.

`non_reasoning_completion_tokens`, kullanıcıya görünür cevabın birebir tokenizer karşılığı olarak değerlendirilmemelidir.

Reasoning veya cached alanları raporlanmamışsa bunların `0` olduğu varsayılmaz.

### 5.3 Latency

LAB02'nin `total_latency_ms` ölçümü istemci tarafındadır. Network, provider load, queueing ve model çalışması gibi etkileri içerebilir.

**Bir koşulun latency değeri daha yüksek çıktığında, bunu yalnızca uzun context'e bağlayamayız.** Özellikle iki tekrar üzerinden kesin nedensel veya istatistiksel sonuç çıkarmayın.

### 5.4 Cevap doğruluğu

Context Comparison'ın beklenen cevabı `4` olduğu için evaluator katı bir integer format kontrolü uygular.

Örnek:

| Model çıktısı | Değerlendirme |
|---|---|
| `4` | `correct` |
| `5` | `incorrect` |
| `There are 4 retries.` | `format_invalid` |
| `Not 4` | `format_invalid` |
| Boş yanıt | `missing_response` |

Bu yaklaşım, beklenen değerin bir cümle içinde geçmesini yeterli sayarak yanlış bir başarı sonucu üretmemeyi amaçlar.

## 6. Multi-Turn Growth metriklerini anlama

Her yeni turn'de kullanıcı mesajı eklenir ve önceki konuşma yeniden modele gönderilir.

| Metrik | Anlamı |
|---|---|
| `turn_index` | Kaçıncı konuşma turu |
| `messages_count` | O turda API'ye gönderilen mesaj sayısı |
| `prompt_tokens` | Sadece ilgili çağrının input tokenları |
| `incremental_prompt_tokens` | Bir önceki turun `prompt_tokens` değerine göre fark |
| `cumulative_prompt_tokens` | İlk turdan itibaren tüm çağrıların input token toplamı |
| `cumulative_total_tokens` | İlk turdan itibaren provider-reported total token toplamı |
| `cumulative_estimated_cost_usd` | Konfigüre fiyatlarla birikimli tahmini maliyet |

### 6.1 `prompt_tokens` ve `cumulative_prompt_tokens` farkı

Aşağıdaki değerler **temsili** örnektir, LAB02'de ölçülmüş değildir:

| Turn | Input Tokens | Incremental | Cumulative Input Tokens |
|---|---:|---:|---:|
| 1 | 150 | N/A | 150 |
| 2 | 300 | 150 | 450 |
| 3 | 450 | 150 | 900 |
| 4 | 600 | 150 | 1.500 |

Dördüncü çağrı tek başına 600 input token kullanırken dört çağrının toplamı 1.500 input tokendır.

Formül:

\[
P_{\text{cumulative}}(n)=\sum_{t=1}^{n}P_t
\]

`incremental_prompt_tokens`, iki çağrının provider-reported input token sayıları arasındaki farktır; birebir “yeni eklenen user metninin token sayısı” değildir. Önceki assistant cevabının uzunluğu da bu farkı etkileyebilir.

### 6.2 Neden cumulative tüketim hızla büyür?

Her turda yaklaşık sabit \(k\) tokenlık yeni history eklenirse, tek çağrının input tüketimi yaklaşık doğrusal artabilir:

\[
P_t \approx P_0+kt
\]

Tüm çağrıların toplamı ise:

\[
\sum_{t=1}^{n}P_t \approx nP_0+k\frac{n(n+1)}{2}
\]

Bu **basitleştirilmiş teorik modeldir**. Gerçek sonuçlar assistant yanıtları, tokenization ve diğer request içeriğine bağlıdır.

LAB02'deki Multi-Turn Growth deneyi bu birikimli etkinin gözlemlenmesini sağlar.

### 6.3 Cumulative cost

Tahmini input/output maliyeti konfigüre edilen model fiyatlarına göre hesaplanır.

Basitleştirilmiş gösterim:

\[
C \approx T_{\text{input}}\times P_{\text{input}}
+T_{\text{output}}\times P_{\text{output}}
\]

Burada token fiyatları **token başına** birimle düşünülmelidir. Projedeki fiyatlar milyon token başına USD olduğundan hesaplamada 1.000.000'a bölme yapılır.

Cached input tokenları mevcutsa farklı fiyatlandırma uygulanabilir. Gerçek fiyat, provider'ın güncel politikalarına bağlıdır ve bu laboratuvarda raporlanan maliyet **estimated cost** niteliğindedir.

## 7. Terminal çıktısını okuma

Gerçek sayıların her çalıştırmada farklı olabileceğini unutmayın. Aşağıdaki yapı örnek amaçlıdır.

### 7.1 Context Condition

```text
Condition: minimal_context
Run: 1/2

Prompt Tokens:         ...
Completion Tokens:     ...
Reasoning Tokens:      ...
Non-Reasoning Tokens:  ...
Total Tokens:          ...
Latency:               ... ms
Estimated Cost:        $...
Finish Reason:         ...
Answer Correct:        True (correct)
```

İlk bakılacak noktalar:

1. `prompt_tokens`: Context boyutu arttığında ne değişti?
2. `Answer Correct`: Soru hâlâ doğru cevaplandı mı?
3. `reasoning_tokens`: Kısa veya uzun context'te belirgin fark var mı?
4. `Latency`: Aynı koşuldaki tekrarlar arasında değişkenlik var mı?
5. `Estimated Cost`: Maliyetteki farkın hangi bölümü input/output kaynaklı?

### 7.2 Multi-Turn

```text
Turn: 3/4

Messages Count:            ...
Prompt Tokens:             ...
Incremental Prompt Tokens: ...
Cumulative Prompt Tokens:  ...
Latency:                   ... ms
Cumulative Estimated Cost: $...
```

Bu bölümde tek tur maliyetinden çok, **turn sayısı arttıkça cumulative input ve cumulative cost'un nasıl büyüdüğüne** odaklanın.

### 7.3 Deney tamamlanması

```text
Experiment Completed

Planned Calls:     10
Successful Calls:  ...
Results Directory: results/lab02_context_growth/<run_id>
```

`Experiment Stopped Early` görülürse raporu başarısız olmayan bir Grand Test gibi sunmayın. Hangi çağrıların gerçekleştiğini ve neden durduğunu inceleyin.

## 8. Sonuç dosyalarını inceleme

Her live çalıştırma, ayrı bir sonuç klasörü kullanır:

```text
results/
└── lab02_context_growth/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── context_measurements.json
        ├── context_measurements.csv
        ├── multi_turn_measurements.json
        ├── multi_turn_measurements.csv
        ├── condition_summary.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Ne için kullanılır? |
|---|---|
| `experiment_manifest.json` | Model, deney konfigürasyonu, planlanan çağrılar ve yürütme durumu |
| `experiment_plan.csv` | Planlanan context koşulları ve turn'ler |
| `context_measurements.json/.csv` | Her context-condition çağrısının ham ölçümleri |
| `multi_turn_measurements.json/.csv` | Tur bazında history büyümesi ve cumulative metrikler |
| `condition_summary.csv` | Context koşullarının özet istatistikleri |
| `experiment_summary.json` | Tüm koşuların durum ve maliyet özeti |
| `analysis_report.md` | Yorumlar, ölçümler ve deneysel sınırlamalar |
| `charts/` | Ölçümlerden türetilen PNG grafikler |

**Önce `analysis_report.md` dosyasını okuyun**, sonra grafiklerde görülen farkları ham ölçümlerde doğrulayın.

LAB02, varsayılan olarak modelin ham cevap metnini sonuç dosyalarına kaydetmez; correctness/format bilgileri ve ölçümler saklanır.

## 9. Grafikler — Nasıl yorumlanır?

LAB02, ölçümler mevcut olduğunda aşağıdaki grafik dosyalarını oluşturmayı hedefler:

| Grafik | Öğrencinin bakması gereken |
|---|---|
| `context_size_vs_latency.png` | Context boyutu ve latency birlikte nasıl değişiyor? |
| `token_breakdown.png` | Prompt/completion token bileşenleri |
| `reasoning_tokens_by_context.png` | Reasoning tokenları context koşullarına göre nasıl dağılıyor? |
| `estimated_cost_by_context.png` | Tahmini maliyetlerin karşılaştırması |
| `multi_turn_growth.png` | Turn ilerledikçe input büyümesi |
| `cumulative_token_consumption.png` | Biriken input/total token etkisi |
| `answer_correctness.png` | Koşullar arasında cevap doğruluğu |
| `summary_dashboard.png` | Genel deney görünümü |

Grafik sayısı kullanılabilir ölçümlere bağlıdır. Tek bir condition çalıştırıldığında bazı karşılaştırmalar bulunmayabilir.

### 9.1 Grafik üzerinden örnek yorum

Şu yorum **doğrudur**:

> Bu çalıştırmada irrelevant context koşulunda mean prompt token sayısı minimal context'ten daha yüksek ölçüldü.

Şu yorum ise **yeterli kanıta dayanmaz**:

> İlgisiz context, her zaman modelin reasoning yapmasını zorlaştırır ve latency'yi kesin artırır.

Çünkü ikinci iddia, küçük örneklemden geniş bir nedensel sonuca geçmektedir.

## 10. API çağrısı yapmadan grafik üretme — Backfill

Daha önce tamamlanmış LAB02 sonuçlarından PNG grafiklerini yeniden üretmek için:

```powershell
docker compose run --rm lab-runner lab02 backfill --config config/default.toml
```

Belirli bir run için:

```powershell
docker compose run --rm lab-runner lab02 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasörünün adını yazın.

Backfill:

- Gerçek Groq API çağrısı yapmaz.
- API key gerektirmez.
- Var olan ölçümleri okuyup grafikleri yeniden oluşturur.
- JSON/CSV/Markdown ölçüm dosyalarını yeniden yazmaz.

`--run-id` belirtilmezse mevcut per-run sonuçlarının yanında eski flat-file LAB02 sonuçları için de grafik oluşturulabilir.

## 11. Sınıf içi uygulama görevleri

### Görev 1 — Aynı soruya üç farklı context

Default Grand Test'i veya yalnızca üç condition'ı çalıştırın. Aşağıdaki tabloyu gerçek ölçümlerle doldurun.

| Condition | Mean Prompt Tokens | Mean Latency (ms) | Mean Cost (USD) | Correct / Calls |
|---|---:|---:|---:|---:|
| `minimal_context` | | | | |
| `expanded_context` | | | | |
| `irrelevant_context` | | | | |

Yorumlayın:

- Hangi koşul en çok input token tüketti?
- En fazla token tüketen koşul, en yüksek latency'yi de verdi mi?
- Aynı cevabın doğruluğu korundu mu?
- Sonuçların hangileri doğrudan ölçüldü, hangileri yorum?

### Görev 2 — Token ekonomisi

İki koşulun `prompt_tokens` ve `estimated_cost_usd` farklarını hesaplayın.

Şunları tartışın:

- Maliyet farkı yalnızca input token farkından mı kaynaklanıyor?
- `completion_tokens` ve `reasoning_tokens` sonucu nasıl etkileyebilir?
- Cache raporlanmışsa input maliyeti neden değişebilir?

### Görev 3 — Conversation History Growth

`multi_turn_growth` çalıştırın.

Her turn için:

```text
turn_index
messages_count
prompt_tokens
incremental_prompt_tokens
cumulative_prompt_tokens
cumulative_estimated_cost_usd
```

değerlerini kaydedin.

Dördüncü turdaki `prompt_tokens` ile dört turun `cumulative_prompt_tokens` değerini karşılaştırın.

### Görev 4 — Deneysel sınırlamalar

`analysis_report.md` içerisindeki **Experimental Confounders** bölümünü okuyun.

Şu soruya cevap hazırlayın:

> `irrelevant_context` daha yavaş çıkarsa bunun tek nedeni context'in ilgisiz olması mıdır?

Bilgi konumu, içerik yapısı, output length, reasoning variability ve provider load etkilerini ayrı ayrı değerlendirin.

## 12. Tartışma soruları

1. Context growth ile context bloat arasındaki fark nedir?
2. `context_chars` neden `prompt_tokens` ile aynı değildir?
3. `completion_tokens` ve `reasoning_tokens` neden doğrudan toplanmaz?
4. Bir API çağrısının başarılı olması, cevabın da doğru olduğunu kanıtlar mı?
5. Çok turlu chatbotta konuşma uzadıkça cumulative token tüketimi neden hızlanabilir?
6. Daha büyük context'in daha yüksek latency yarattığını söylemek için hangi ek kontroller gerekir?
7. Aynı soruya doğru cevap veren iki condition arasında nasıl bir cost/performance tercihi yapardınız?
8. LAB02'deki full-history davranışını production ortamında sürdürmenin olası riskleri nelerdir?

## 13. Troubleshooting

### `LiveExecutionNotAllowed`

Olası neden:

- `--live` kullanıldı fakat `config/live.toml` içinde `live_api_enabled = true` değil.
- `GROQ_API_KEY` tanımlı değil.

Çözüm:

1. `.env` dosyasını doğrulayın.
2. `config/live.toml` dosyasını doğrulayın.
3. `docker compose build lab-runner` komutunu tekrar çalıştırın.

### `config/live.toml` bulunamıyor

Önce:

```powershell
Copy-Item config/default.toml config/live.toml
docker compose build lab-runner
```

komutlarını çalıştırın.

### `planned calls exceed lab02.max_api_calls`

Deney kapsamı çağrı bütçesini aşıyor.

`all`, `--repetitions` ve `--turns` seçeneklerinin toplam çağrı sayısını hesaplayın. Gerekirse pilot komuta dönün.

### HTTP 429 / Rate limit

Çok fazla istek yapılmış veya sağlayıcı kotası aşılmış olabilir.

- Aynı komutu sürekli tekrarlamayın.
- Groq Console üzerinden limitleri kontrol edin.
- Daha küçük pilot deney seçin.
- API anahtarı paylaşarak limit sorununu çözmeye çalışmayın.

### Bazı metrikler `N/A`

Provider ilgili alanı raporlamamış olabilir veya çağrı başarısız olmuş olabilir.

**Eksik ölçümü otomatik olarak sıfır kabul etmeyin.**

### Grafik oluşmadı

Önce ölçümlerin gerçekten yazıldığını kontrol edin. Uygun veri varsa `backfill` deneyebilirsiniz.

### Sonuç klasörü aynı isimle zaten var

Yeni `--experiment-run-id` kullanın. LAB02 aynı run ID altında kayıtlı ölçümlerin üzerine tekrar sonuç eklemeyi reddeder.

## 14. LAB02'nin LAB10 ile ilişkisi

LAB02, conversation history yeniden gönderildiğinde **context growth ve cumulative token tüketimi** sorununu gözlemletir.

LAB10 ise **Conversation Summarization** kullanarak bu history'nin daha kısa bir temsilini oluşturmayı ve summary-generation overhead dahil gerçek toplam maliyeti incelemeyi amaçlar.

Bu iki laboratuvar birbirini tamamlar; LAB02'nin içinde summarization optimizasyonu çalıştırılmaz.

## 15. Deney sonunda ne öğrenmiş olmalısınız?

LAB02 tamamlandığında şu ayrımları açıklayabilmelisiniz:

- **Context length** ile **context relevance** aynı şey değildir.
- **Request-level input tokens** ile **cumulative input tokens** farklıdır.
- **API success** ile **answer correctness** farklıdır.
- **Reasoning tokens**, completion tokens içinde yer alabilir.
- **Latency**, yalnızca prompt uzunluğuyla açıklanamaz.
- **Estimated cost**, provider usage ve konfigüre fiyatlara dayanır.
- **Conversation history growth**, çok turlu uygulamalarda maliyet üzerinde önemli bir tasarım kararıdır.

> **Temel çıkarım:** Daha fazla context göndermek her zaman daha iyi cevap elde etmek anlamına gelmez. İyi bir LLM uygulaması, ihtiyacı olan bilgiyi yeterli bağlamla sunarken token tüketimini, maliyeti ve doğruluğu birlikte değerlendirir.

---

**Kaynak kodu incelemek isteyenler için:**

- `src/labs/lab02_context_growth/experiments.py` — Context fixtures ve sabit soru
- `src/labs/lab02_context_growth/runner.py` — Gerçek çağrılar ve multi-turn growth
- `src/labs/lab02_context_growth/evaluation.py` — Deterministic answer evaluation
- `src/labs/lab02_context_growth/stats.py` — Condition summaries
- `src/labs/lab02_context_growth/report.py` — Analiz raporu
- `src/labs/lab02_context_growth/charts.py` — Grafik üretimi
- `config/default.toml` — LAB02 varsayılan konfigürasyonu
