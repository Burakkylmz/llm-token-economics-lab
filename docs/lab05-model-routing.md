# LAB05 — Model Routing: Cost, Quality & Latency

> **Katılımcı rehberi | Python · Groq API · Docker Compose**\
> **Öğrenme alanı:** LLM Model Routing, task complexity, rule-based routing, token economics ve quality-constrained optimization.\
> **Önerilen sınıf deneyimiz:** `lab05 all --profile default` — varsayılan olarak **12 mantıksal API çağrısı** planlar.

---

## 1. Bu laboratuvarın amacı

Bir LLM uygulamasındaki bütün istekleri aynı modele göndermek kolaydır; fakat her görev aynı model kapasitesini gerektirmeyebilir. Örneğin basit bir bilgi sorusunu cevaplamak ile birden fazla iş kuralını birlikte değerlendirmek farklı yetenekler gerektirebilir.

**Model Routing**, her isteği özelliklerine göre uygun modele yönlendirme yaklaşımıdır.

LAB05'in temel araştırma sorusu:

> Bir uygulama, bütün görevlerde küçük modeli kullanmak yerine bazı görevleri daha güçlü bir modele yönlendirerek cevap kalitesini koruyabilir veya iyileştirebilir mi? Bunu yapmanın token, maliyet ve latency bedeli nedir?

Bu laboratuvarda **sadece maliyeti azaltmaya çalışmıyoruz**. Aynı zamanda yanlış cevap verme, formatı bozma ve kritik görevleri yetersiz modele yönlendirme riskini de gözlemliyoruz.

### 1.1 Öğrenme hedefleri

Deney bittiğinde şu konuları açıklayabilmelisiniz:

1. `Always-Small`, `Always-Large` ve `Rule-Based Routing` arasındaki fark.
2. Modelin büyüklüğü ile görevin gerçek zorluğunun neden aynı şey olmadığı.
3. Router'ın hangi sinyallerle model seçtiği.
4. Seçilen modelin input, output ve reasoning token tüketimi.
5. `accuracy`, `estimated_cost_usd` ve `total_latency_ms` metriklerinin birlikte yorumlanması.
6. Task-level `ROUTING_QUALITY_REGRESSION` ve `QUALITY_PRESERVED` sonuçları.
7. Daha düşük maliyetin neden tek başına optimizasyon başarısı sayılmadığı.

**Temel ilke:** Ucuz model seçmek bir amaç değil, **gerekli cevap kalitesini sağlayan en uygun model seçimini** yapmanın araçlarından biridir.

---

## 2. Deneyde kullanılan modeller

Projedeki LAB05 varsayılan konfigürasyonu iki modeli kullanır:

| Rol | Groq model ID | Deneydeki anlamı |
|---|---|---|
| Small | `openai/gpt-oss-20b` | Basit görevlerde kullanılabilecek daha küçük model |
| Large | `openai/gpt-oss-120b` | Daha karmaşık görevler için değerlendirilen büyük model |

Bu adlandırmaların anlamı **LAB05'in model karşılaştırma rolüdür**. `Large` modelin her görevi daha doğru yanıtlayacağı garanti edilmez.

Her iki model de aynı Groq API üzerinden çağrılır. LAB05, OpenAI Platform API kullanan bir multi-provider deneyi değildir.

Konfigürasyonun varsayılan bazı sabitleri:

```toml
[lab05]
small_model = "openai/gpt-oss-20b"
large_model = "openai/gpt-oss-120b"
default_profile = "default"
reasoning_effort = "low"
max_completion_tokens = 1024
max_api_calls = 18
max_estimated_cost_usd = 0.05
```

**Not:** Modellerin güncel erişilebilirliği ve fiyatlandırması değişebilir. Laboratuvar, konfigürasyondaki fiyatlara göre **tahmini** maliyet hesaplar; sonuçlar gerçek provider kullanım alanlarına dayanır.

---

## 3. Karşılaştırılan üç strateji

### 3.1 Strategy A — Always-Small

**ID:** `always_small`

Her görev, ne kadar kolay veya karmaşık olursa olsun, Small modele gönderilir.

```text
Task 1 ─┐
Task 2 ─┼──> GPT-OSS 20B
Task 3 ─┤
Task 4 ─┘
```

Bu koşul LAB05'in **fixed baseline** stratejisidir.

**Sorusu:** Her görevde küçük modeli kullanmanın maliyeti ve kalite riski nedir?

### 3.2 Strategy B — Always-Large

**ID:** `always_large`

Her görev Large modele gönderilir.

```text
Task 1 ─┐
Task 2 ─┼──> GPT-OSS 120B
Task 3 ─┤
Task 4 ─┘
```

**Sorusu:** Her isteği büyük modele göndermek ne kazandırıyor, ne kadar maliyet oluşturuyor?

`all --profile extended` koşulunda karşılaştırmaya dahil edilir. İstenirse ayrı komut olarak da çalıştırılabilir.

### 3.3 Strategy C — Rule-Based Routing

**ID:** `rule_based_routing`

İstek önce **deterministic router** tarafından incelenir; sonra 20B veya 120B modeli seçilir.

```text
                       ┌─ Basit görev ───> GPT-OSS 20B
Incoming task ─> Router│
                       └─ Kural eşleşti ─> GPT-OSS 120B
```

Router'ın kendisi LLM değildir ve **ek Groq API çağrısı yapmaz**.

Gerçek uygulamada bu ayrım önemlidir: Eğer model seçimi başka bir LLM'e yaptırılsaydı, routing kararının da token, latency ve maliyet overhead'i olurdu. **LAB05'te böyle bir overhead bulunmaz.**

---

## 4. Router model seçimini nasıl yapıyor?

Bu projedeki router yalnızca **görevin prompt metnini** inceler.

Kullandığı sinyaller:

| Feature / Rule | Örnek sinyal | Etki |
|---|---|---|
| Multi-step reasoning | `then`, `step by step`, `in total` | Large model seçilebilir |
| Conditional constraints | `if` + constraint ifadesi | Large model seçilebilir |
| Code generation | `function`, `python`, `def` | Large model seçilebilir |
| Structured output | `json object`, `json schema` | Large model seçilebilir |
| Uzun prompt | Character threshold aşılması | **Tek başına Large seçim nedeni değildir** |

Bir Large-model kuralı eşleşmiyorsa Small model seçilir.

Router'ın temel karar politikası:

```text
IF a large-model routing rule matches:
    select GPT-OSS 120B
ELSE:
    select GPT-OSS 20B
```

Bu yalnızca öğrenim amacı taşıyan, **explainable rule-based policy** örneğidir. Karmaşıklığın bütün türlerini anlayan evrensel bir sınıflandırıcı değildir.

### 4.1 Router hangi bilgileri kullanmaz?

Router'a şunlar verilmez:

- Golden expected answer
- `complexity_label`
- Modelin doğru/yanlış sonuçları
- Önceki kalite ölçümleri
- İkinci bir LLM'nin değerlendirmesi

Böylece deneyde beklenen cevabı bilen “hileli” bir router oluşması önlenir.

### 4.2 Routing kararını inceleme

`--dry-run` çıktısında öğrenciler aşağıdaki alanları görebilir:

```text
task=task_001
strategy=rule_based_routing
selected_model=openai/gpt-oss-20b
routing_reason=No large-model routing rule matched
```

Bu örnek bir karar biçimini gösterir; aynı komut gerçek seçimi mevcut policy'ye göre hesaplayacaktır.

Öğrencinin sorusu:

> Router neden bu modeli seçti? Görevin gerçekten ihtiyaç duyduğu kapasite ile seçilen model uyumlu mu?

---

## 5. Golden Dataset ve task profilleri

LAB05, **13 görevden oluşan versioned golden dataset** içerir.

Kaynak dosya:

```text
src/labs/lab05_model_routing/golden_dataset.json
```

Görevler arasında şunlar bulunur:

- Factual retrieval
- Simple arithmetic
- Structured extraction
- Classification
- Multi-step reasoning
- Multi-constraint decision
- Conditional business rule
- Structured output generation
- Code generation

Her görev kaydında `task_id`, `category`, `complexity_label`, `prompt`, `expected_answer` ve `evaluation_type` gibi alanlar bulunur.

### 5.1 Çalıştırma profilleri

| Profil | Değerlendirilen görev | Strateji | Planlanan mantıksal API çağrısı |
|---|---:|---:|---:|
| `pilot` | 4 | Always-Small + Router | **8** |
| `default` | 6 | Always-Small + Router | **12** |
| `extended` | 6 | Always-Small + Always-Large + Router | **18** |

**Önemli:** Extended profil, bütün 13 golden task'ı çalıştırmaz. Default ile **aynı 6 görev** üzerinde üçüncü strateji olarak Always-Large karşılaştırmasını ekler.

Bu, deneylerin kapsamını ve API kotasını kontrol altında tutar.

### 5.2 Pilot profildeki görevler

| Task ID | Kategori | Öğrenme amacı |
|---|---|---|
| `task_001` | Factual retrieval | Basit bilgi sorusu |
| `task_003` | Simple arithmetic | Basit matematik işlemi |
| `task_007` | Classification | Ürün yorumunun duygu sınıflandırması |
| `task_010` | Multi-constraint decision | Birden fazla koşul içeren karar |

### 5.3 Default profildeki görevler

| Task ID | Kategori | Görev tipi |
|---|---|---|
| `task_001` | Factual retrieval | İtalya'nın başkenti |
| `task_003` | Simple arithmetic | `12 + 7` |
| `task_005` | Structured extraction | Invoice alanlarını çıkarma |
| `task_007` | Classification | Sentiment classification |
| `task_009` | Multi-step reasoning | İki güne yayılan cupcake hesabı |
| `task_011` | Conditional business rule | Koşullu shipping ücreti |

Her strateji, kendi profilindeki aynı görevleri yanıtlar; böylece task-level karşılaştırma yapılabilir.

---

## 6. Docker ile hazırlık — Windows PowerShell

Aşağıdaki komutlar, terminalin proje kök klasöründe açıldığını varsayar.

### 6.1 Gerekenler

- Docker Desktop ve Docker Compose
- LAB05 kodlarını içeren `llm-token-economics-lab` repository
- Canlı deney için Groq API key
- İnternet erişimi

Python veya `uv` sisteminize ayrıca kurulmak zorunda değildir; Docker image bunları kullanır.

### 6.2 Proje klasörüne geçin

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Bilgisayarınızda farklı bir klasör kullanıyorsanız yolu değiştirin.

### 6.3 Groq API anahtarını hazırlayın

Projenin örnek environment dosyasını kopyalayın:

```powershell
Copy-Item .env.example .env
```

`.env` dosyasına kendi anahtarınızı yazın:

```dotenv
GROQ_API_KEY=your_api_key_here
```

**API anahtarını GitHub'a commit etmeyin, arkadaşlarınızla paylaşmayın ve ekran görüntüsünde göstermeyin.**

### 6.4 Canlı konfigürasyon oluşturun

Güvenli varsayılan konfigürasyonu kopyalayın:

```powershell
Copy-Item config/default.toml config/live.toml
```

Ardından `config/live.toml` dosyasında `[execution]` bölümündeki değeri değiştirin:

```toml
[execution]
live_api_enabled = true
```

Dosyanın geri kalanını silmeyin.

Gerçek Groq API çağrısı için **iki koşul birlikte** gereklidir:

1. CLI komutunda `--live`.
2. Config içinde `execution.live_api_enabled = true`.

`config/default.toml` dosyasını kalıcı olarak live çalışacak hale getirmeyin.

### 6.5 Docker image'ını build edin

```powershell
docker compose build lab-runner
```

`config/live.toml` dosyasını oluşturduktan sonra image'ı yeniden build etmek önemlidir; repository'nin Dockerfile'ı `config/` klasörünü image içerisine kopyalar.

### 6.6 Yardım ekranı

```powershell
docker compose run --rm lab-runner lab05 --help
```

Bu komut gerçek API isteği göndermez.

---

## 7. Docker komutları — Deneyleri başlatma

### A. Dry-Run — Önerilen ilk adım

```powershell
docker compose run --rm lab-runner lab05 all --profile default --dry-run --config config/default.toml
```

Dry-run:

- 12 çağrılık deney planını çıkarır.
- Seçilen görevleri ve stratejileri gösterir.
- Rule-based router'ın model seçimlerini offline hesaplar.
- Conservative estimated maximum cost görüntüler.
- `experiment_plan.csv` oluşturur.
- **Groq API çağrısı yapmaz.**

Gerçek token, latency veya cevap doğruluğu ölçümlerinin dry-run sonucunda oluşması beklenmez.

### B. Pilot Grand Test — 8 çağrı

```powershell
docker compose run --rm lab-runner lab05 all --profile pilot --live --config config/live.toml
```

Dört görev iki farklı strateji ile çalıştırılır.

**Öneri:** Canlı çağrılara ilk defa geçiyorsanız Pilot deneyle başlayın.

### C. Default Grand Test — 12 çağrı

```powershell
docker compose run --rm lab-runner lab05 all --profile default --live --config config/live.toml
```

Altı görev üzerinde:

- Always-Small: 6 çağrı
- Rule-Based Routing: 6 çağrı

**Toplam: 12 mantıksal API çağrısı.**

Bu, dersin ana karşılaştırmasıdır.

### D. Extended Grand Test — 18 çağrı

```powershell
docker compose run --rm lab-runner lab05 all --profile extended --live --config config/live.toml
```

Aynı altı görev üç stratejiyle karşılaştırılır:

- Always-Small: 6
- Always-Large: 6
- Rule-Based Routing: 6

**Toplam: 18 mantıksal API çağrısı.**

Bu deney sayesinde “her zaman büyük model kullanırsak ne olur?” sorusu da ölçülebilir.

### E. Yalnızca Fixed Small Baseline

```powershell
docker compose run --rm lab-runner lab05 fixed_model_baseline --profile default --live --config config/live.toml
```

**6 mantıksal API çağrısı** planlar.

### F. Yalnızca Rule-Based Routing

```powershell
docker compose run --rm lab-runner lab05 rule_based_routing --profile default --live --config config/live.toml
```

**6 mantıksal API çağrısı** planlar.

### G. Yalnızca Always-Large

```powershell
docker compose run --rm lab-runner lab05 always_large --profile default --live --config config/live.toml
```

**6 mantıksal API çağrısı** planlar.

`always_large` stratejisi tek başına çalıştırılabilir; fakat `all --profile default` içine otomatik olarak dahil edilmez. Üç stratejili birleşik karşılaştırma için `extended` kullanın.

### H. Extended Dry-Run

```powershell
docker compose run --rm lab-runner lab05 all --profile extended --dry-run --config config/default.toml
```

Gerçek API isteği yapmadan 18 çağrının hangi model ve görevlerle eşleştirileceğini görün.

### I. Özel Run ID

```powershell
docker compose run --rm lab-runner lab05 all --profile default --experiment-run-id class-lab05-team-a --live --config config/live.toml
```

Sonuçlar:

```text
results/lab05_model_routing/class-lab05-team-a/
```

LAB05, ölçüm kaydı bulunan aynı run ID'ye yeniden yazmayı reddeder. Yeni bir çalıştırmada yeni run ID kullanın.

**Not:** Bu sayılar planlanan **mantıksal** API çağrılarıdır. Başarısızlık, maliyet sınırı veya operational retry nedeniyle gerçek deneme sayısı farklı olabilir.

---

## 8. LAB05 hangi metrikleri ölçüyor?

### 8.1 Model seçimi

| Metrik | Anlamı |
|---|---|
| `strategy` | Kullanılan model seçim stratejisi |
| `selected_model` | Gerçekte hangi model çağrıldı? |
| `model_tier` | `small` veya `large` |
| `routing_reason` | Seçim gerekçesi |
| `matched_rules` | Router'ın eşleştirdiği kurallar |
| `policy_version` | Router policy sürümü |
| `complexity_label` | Golden dataset'te bulunan offline zorluk etiketi |

`complexity_label`, router'ın kararına girdi değildir. Analiz yapmak için dataset metadata'sıdır.

### 8.2 Token Economics

| Metrik | Anlamı |
|---|---|
| `prompt_tokens` | API'ye gönderilen input tokenları |
| `completion_tokens` | Toplam output/completion tokenları |
| `reasoning_tokens` | Provider tarafından raporlanan reasoning tokenları |
| `non_reasoning_completion_tokens` | Uygunsa completion ile reasoning farkı |
| `total_tokens` | Toplam token tüketimi |
| `cached_tokens` | Provider raporladıysa cached input tokenları |

**Reasoning tokens, completion tokens içine dahildir.** İkisini tekrar toplayarak toplam token hesaplamayın.

### 8.3 Cost & Performance

| Metrik | Anlamı |
|---|---|
| `estimated_input_cost_usd` | Input maliyeti tahmini |
| `estimated_output_cost_usd` | Output maliyeti tahmini |
| `estimated_cost_usd` | İlgili çağrının toplam tahmini maliyeti |
| `total_latency_ms` | İstemci tarafında ölçülen çağrı gecikmesi |
| `finish_reason` | Üretimin neden sonlandığı |
| `attempts_used` | Operational retry dahil deneme sayısı |

Her iki modelin input ve output fiyatları farklı olabilir. Bu nedenle yalnızca token sayısına bakarak “hangi çağrı daha ucuz?” sorusuna kesin cevap veremeyiz.

### 8.4 Quality

| Metrik | Anlamı |
|---|---|
| `answer_correct` | Beklenen cevaba göre doğruluk |
| `answer_complete` | Gerekli alan veya içeriğin bulunması |
| `evaluation_status` | `CORRECT`, `INCORRECT`, `INCOMPLETE`, `INVALID_FORMAT`, `TRUNCATED`, vb. |
| `evaluation_detail` | Evaluator açıklaması |
| `accuracy` | Stratejide `CORRECT` sayısının toplam görev sayısına oranı |

**API success ≠ Answer correctness.**

Başarılı bir HTTP isteği, yanlış veya eksik bir cevap döndürebilir.

### 8.5 Evaluation sınırları

LAB05, görev tipine göre farklı deterministic evaluator kullanır:

- Exact match
- Numeric comparison
- Classification
- Multi-field extraction
- JSON schema/value validation
- Structural code validation

Code generation örneğinde syntax/AST doğrulaması, kodun fonksiyonel olarak doğru çalıştığını kanıtlamaz. Bu tür kayıtların `STRUCTURAL_VALIDATION_ONLY` durumu tam doğrulukla karıştırılmamalıdır.

Özellikle bu laboratuvarın küçük görev kümesindeki `accuracy` sonuçları, modelin genel yetenekleri hakkında kapsamlı benchmark olarak yorumlanmamalıdır.

---

## 9. Deney sonuçlarını nasıl okuyacağız?

### 9.1 Her görev için terminal çıktısı

Gerçek sayılar modele ve çalıştırma koşullarına göre değişir.

Örnek çıktı yapısı:

```text
LAB05 -- Model Routing

Task: task_009
Category: multi_step_reasoning
Strategy: rule_based_routing

Selected Model: ...
Routing Reason: ...

Prompt Tokens: ...
Completion Tokens: ...
Reasoning Tokens: ...
Non-Reasoning Tokens: ...
Total Tokens: ...

Answer Correct: ...
Answer Complete: ...
Evaluation Status: ...
Finish Reason: ...

Client Latency: ... ms
Estimated Cost: $...
```

**İlk bakılacak sorular:**

1. Router hangi modeli seçti?
2. Neden o modeli seçti?
3. Görev doğru tamamlandı mı?
4. Baseline aynı görevde nasıl sonuç verdi?
5. Daha yüksek maliyet bir kalite kazanımı sağladı mı?

### 9.2 Model Routing Comparison

Grand Test sonunda strateji bazında özet göreceksiniz:

```text
-- Model Routing Comparison --

always_small:
  Tasks: ...
  Correct Answers: ...
  Accuracy: ...
  Small Model Calls: ...
  Large Model Calls: ...
  Total Estimated Cost: $...
  Mean Latency: ... ms

rule_based_routing:
  Tasks: ...
  Correct Answers: ...
  Accuracy: ...
  Small Model Calls: ...
  Large Model Calls: ...
  Total Estimated Cost: $...
  Mean Latency: ... ms

Cost Delta (routing - baseline): $...
Overall Assessment: ...
```

Burada asıl karşılaştırma:

**Always-Small vs Rule-Based Routing**

Extended profilde ayrıca **Always-Large** sonucu da görülür.

### 9.3 “Routing Cost Savings” nasıl yorumlanır?

LAB05'in kullandığı hesap:

\[
\mathrm{CostDelta} = C_{\mathrm{routing}} - C_{\mathrm{baseline}}
\]

\[
\mathrm{Savings\%} =
\frac{C_{\mathrm{baseline}} - C_{\mathrm{routing}}}
{C_{\mathrm{baseline}}} \times 100
\]

Örnek değerler **temsili**:

```text
Always-Small Cost: $0.0010
Routing Cost:      $0.0013

Cost Delta:        +$0.0003
Routing Savings:   -30%
```

Burada Router **%30 daha pahalıdır**. Bu sonuç otomatik olarak başarısızlık anlamına gelmez: Router doğru cevap sayısını artırmış olabilir.

Tersi durumda da aynı yaklaşım geçerlidir. Router daha ucuz fakat daha az doğru cevap ürettiyse, bunu koşulsuz başarı olarak sunamayız.

### 9.4 Quality Regression

LAB05 iki stratejinin **aynı task** üzerindeki doğruluğunu ayrıca karşılaştırır.

| Outcome | Anlamı |
|---|---|
| `QUALITY_PRESERVED` | Baseline doğru, Router doğru |
| `ROUTING_QUALITY_IMPROVEMENT` | Baseline yanlış, Router doğru |
| `ROUTING_QUALITY_REGRESSION` | Baseline doğru, Router yanlış |
| `BOTH_FAILED` | İkisi de yanlış |
| `INSUFFICIENT_DATA` | Güvenilir karşılaştırma için bilgi eksik |

**Örnek:**

```text
Task: task_011
Always-Small: Correct
Router: Incorrect
Quality Outcome: ROUTING_QUALITY_REGRESSION
```

Bu örnekte daha düşük maliyet veya gecikme, kritik kalite kaybını ortadan kaldırmaz.

---

## 10. Sonuç dosyaları ve dizin yapısı

Her live çalıştırma ayrı bir klasöre kaydedilir:

```text
results/
└── lab05_model_routing/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── routing_measurements.json
        ├── routing_measurements.csv
        ├── task_comparison.csv
        ├── strategy_summary.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Öğrenme açısından kullanım |
|---|---|
| `experiment_manifest.json` | Model, konfigürasyon ve experiment metadata |
| `experiment_plan.csv` | Hangi task'ın hangi stratejiyle değerlendirildiği |
| `routing_measurements.json/.csv` | Her API çağrısının token, model, routing ve kalite ölçümü |
| `task_comparison.csv` | Aynı task'ta baseline ve routed sonuçların karşılaştırılması |
| `strategy_summary.csv` | Strateji bazında doğruluk, seçim oranı, maliyet ve latency |
| `experiment_summary.json` | Grand Test özeti |
| `analysis_report.md` | Teknik analiz, kalite regresyonları ve deneysel sınırlamalar |
| `charts/` | Görsel analiz |

**Önerilen inceleme sırası:**

1. `analysis_report.md`
2. `strategy_summary.csv`
3. `task_comparison.csv`
4. `routing_measurements.csv`
5. `charts/`

Bu laboratuvarda modelin ham response metni varsayılan sonuç dosyalarına kaydedilmez; evaluasyon durumları, karakter sayısı ve sayısal ölçümler saklanır.

---

## 11. Grafiklerin yorumlanması

LAB05, kullanılabilir veriler için aşağıdaki grafik dosyalarını üretmeyi hedefler:

| Grafik | Hangi soruya cevap arıyoruz? |
|---|---|
| `model_selection_distribution.png` | Router'ın Small/Large seçim oranı nedir? |
| `accuracy_by_strategy.png` | Hangi strateji kaç görevi doğru yanıtladı? |
| `cost_by_strategy.png` | Stratejilerin toplam maliyet farkı ne? |
| `latency_by_strategy.png` | Stratejilerin gözlemlenen latency dağılımı nasıl? |
| `cost_per_task.png` | Maliyet farkları hangi task'larda oluşuyor? |
| `quality_regression_matrix.png` | Router nerede kaliteyi korudu veya kaybetti? |
| `token_consumption_breakdown.png` | Token tüketiminin bileşenleri neler? |
| `routing_decision_analysis.png` | Hangi routing kuralları ne sıklıkta eşleşti? |
| `quality_cost_tradeoff_dashboard.png` | Kalite ve maliyet birlikte ne gösteriyor? |

Grafikleri değerlendirirken yalnızca ortalama maliyete odaklanmayın. Özellikle `quality_regression_matrix.png` ile `cost_by_strategy.png` grafiklerini birlikte okuyun.

### 11.1 Sınırlamalar

- Pilot ve Default profiller az sayıda görev içerir.
- Üç stratejili Extended deney bile genel model üstünlüğünü kanıtlamaz.
- Provider load değişkenliği latency'yi etkileyebilir.
- Routing kuralı prompt metnindeki sözcüklere bağlıdır; gerçek semantik karmaşıklığı tam olarak ölçmez.
- Quality evaluator kendi sınırlarına sahiptir.
- Router'ın ek API maliyeti bulunmaması, yalnızca bu **deterministic rule-based** tasarım için geçerlidir.

---

## 12. Backfill — API çağrısı yapmadan grafik üretme

Mevcut sonuçlardan tekrar grafik üretmek için:

```powershell
docker compose run --rm lab-runner lab05 backfill --config config/default.toml
```

Belirli bir deney için:

```powershell
docker compose run --rm lab-runner lab05 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasörünün adını yazın.

Backfill:

- Yeni Groq API çağrısı yapmaz.
- API key gerektirmez.
- Kaydedilmiş ölçümlerden grafikleri yeniden üretir.
- Mevcut JSON/CSV/Markdown ölçümlerini yeniden yazmaz.

---

## 13. Sınıf içi uygulamalar

### Görev 1 — Dry-Run ile router kararlarını analiz edin

Çalıştırın:

```powershell
docker compose run --rm lab-runner lab05 all --profile default --dry-run --config config/default.toml
```

Bir tablo oluşturun:

| Task | Router'ın seçtiği model | Gerekçe | Siz hangi modeli seçerdiniz? |
|---|---|---|---|
| task_001 | | | |
| task_003 | | | |
| task_005 | | | |
| task_007 | | | |
| task_009 | | | |
| task_011 | | | |

Şu soruyu tartışın:

**Router kararları gerçekten task complexity'yi mi ölçüyor, yoksa prompt içerisindeki belirli kelimelere mi tepki veriyor?**

### Görev 2 — Default Grand Test

Default deneyini çalıştırın ve aşağıdaki tabloyu doldurun:

| Metrik | Always-Small | Rule-Based Routing |
|---|---:|---:|
| Tasks | | |
| Correct Answers | | |
| Accuracy | | |
| Small Model Calls | | |
| Large Model Calls | | |
| Total Estimated Cost | | |
| Mean Latency (ms) | | |

**Soru:** Router daha pahalıysa, bunun karşılığında kalite kazanımı var mı?

### Görev 3 — Task-Level Quality Regression

`task_comparison.csv` dosyasını açın.

Her görev için:

- Baseline Correct
- Routed Correct
- Cost Delta
- Latency Delta
- Quality Outcome

alanlarını inceleyin.

En az iki görevin sonucunu yazılı olarak karşılaştırın.

Özellikle `ROUTING_QUALITY_REGRESSION` görüldüyse, bunu göz ardı etmeyin.

### Görev 4 — Extended Grand Test

Extended profili çalıştırın:

```powershell
docker compose run --rm lab-runner lab05 all --profile extended --live --config config/live.toml
```

Üç stratejiyi karşılaştırın:

| Metrik | Always-Small | Always-Large | Router |
|---|---:|---:|---:|
| Correct Answers | | | |
| Estimated Cost | | | |
| Mean Latency | | | |
| Small Calls | | | |
| Large Calls | | | |

**Sorular:**

- Always-Large gerçekten bütün görevlerde daha doğru muydu?
- Router hangi durumlarda daha küçük modeli seçti?
- Router, Always-Large'a kıyasla düşük maliyet sağlayabildi mi?
- Always-Small'a göre kalite farkı var mı?

### Görev 5 — Cost vs Quality

Aşağıdaki senaryo **temsili**:

```text
Always-Small:
  Accuracy: 4/6
  Cost: $0.0010

Routing:
  Accuracy: 6/6
  Cost: $0.0014

Always-Large:
  Accuracy: 6/6
  Cost: $0.0020
```

Şu kararları gerekçelendirin:

1. Sadece en düşük maliyet isteniyorsa hangi strateji?
2. Doğru cevap kritikse hangi strateji?
3. Başarılı görev başına maliyet hesaplansa karar değişir mi?
4. Gerçek dünyada yanlış cevap maliyetini bu karşılaştırmaya nasıl dahil ederdiniz?

Bu rakamlar gerçek LAB05 sonucu değildir.

### Görev 6 — Router Policy Eleştirisi

Şu örnekleri değerlendirin:

- Uzun fakat basit bir prompt
- Kısa fakat zor bir mantık sorusu
- `if` kelimesi içeren fakat kolay bir görev
- Kod üretimi istemeyen fakat `python` kelimesi geçen bir açıklama

**Soru:** Rule-based router yanlış model seçebilir mi?

Olası false positive / false negative durumlarını tartışın.

---

## 14. Öğrenci tartışma soruları

1. Model Routing ile Model Benchmarking arasındaki fark nedir?
2. `Always-Small` neden gerekli bir kontrol koşuludur?
3. `Always-Large` hangi durumda üretimde ekonomik açıdan verimsiz olabilir?
4. Uzun prompt neden otomatik olarak zor görev anlamına gelmez?
5. `routing_reason` kaydetmenin production observability açısından faydası nedir?
6. `complexity_label` değerini router'a vermek neden bilimsel açıdan yanlış olur?
7. Hangi maliyet daha kritiktir: API tüketimi mi, yanlış cevabın iş etkisi mi?
8. Bir router'ın başarısını sadece `accuracy` ile ölçebilir miyiz?
9. Routing sistemi neden `quality regression` takibi yapmalıdır?
10. Aynı model seçimi farklı latency sonuçları üretebilir mi?

---

## 15. Troubleshooting

### `LiveExecutionNotAllowed`

`--live` var fakat konfigürasyonda `execution.live_api_enabled = false` olabilir.

Çözüm:

1. `config/live.toml` dosyasının varlığını kontrol edin.
2. `[execution]` bölümünde `live_api_enabled = true` olduğundan emin olun.
3. `docker compose build lab-runner` ile image'ı yeniden oluşturun.
4. Komutta `--config config/live.toml` kullanın.

### `GROQ_API_KEY` eksik

`.env` dosyasına geçerli anahtarı ekleyin; anahtarınızı paylaşmayın.

### `planned calls exceed lab05.max_api_calls`

Planlanan çağrılar konfigürasyondaki sınırdan fazladır.

Varsayılan çağrı limitini artırmak yerine önce `pilot` veya `default` profiline dönün.

### HTTP 429 / Rate Limit

Provider kotası veya rate limit devreye girmiş olabilir.

- Deneyi sürekli yeniden başlatmayın.
- Groq Console üzerinden ilgili limiti kontrol edin.
- Daha küçük profile dönün.

### `estimated_cost_usd` alanı `N/A`

Seçilen modelin fiyatı konfigürasyonda eksik veya usage alanları kullanılamıyor olabilir.

Eksik maliyeti sıfır kabul etmeyin.

### `INCORRECT`, `INCOMPLETE` veya `INVALID_FORMAT`

Bu durum API hata koduyla aynı değildir. Model başarılı bir API yanıtı üretmiş ancak evaluation contract'ını karşılamamış olabilir.

### `STRUCTURAL_VALIDATION_ONLY`

Kod üretimindeki yapısal kontrol tam bir unit test değildir; kodun çalışma doğruluğu ayrıca doğrulanmamıştır.

### Aynı run ID kullanılmış

Yeni bir `--experiment-run-id` seçin. LAB05 tamamlanmış ölçümlerin üzerine yazmayı reddeder.

### Grafik üretilemedi

Ölçüm dosyalarının oluştuğunu doğrulayın. Ardından `lab05 backfill` deneyin.

---

## 16. Eğitimi LAB05 ile kapatırken

LAB05'ten beklenen en önemli öğrenim kazanımları:

- **Small model her zaman en doğru seçenek değildir.**
- **Large model her zaman en ekonomik veya en doğru seçenek değildir.**
- **Routing ek bir karar mekanizmasıdır ve yanlış yönlendirebilir.**
- **Routing kararları açıklanabilir ve denetlenebilir olmalıdır.**
- **Token ve maliyet tasarrufu, cevap doğruluğundan ayrı değerlendirilmemelidir.**
- **Quality regression, başarılı bir optimization raporunun içinde gizlenmemelidir.**

> **Ana çıkarım:** İyi bir Model Routing sistemi, yalnızca daha ucuz modeli seçen sistem değildir. Görevin gerektirdiği kaliteyi ve performansı sağlayarak toplam işletim maliyetini kontrol etmeye çalışan sistemdir.

---

## 17. Kaynak kodu incelemek isteyen öğrenciler için

| Dosya | İncelenecek konu |
|---|---|
| `src/labs/lab05_model_routing/golden_dataset.json` | 13 golden task ve expected answers |
| `src/labs/lab05_model_routing/golden_dataset.py` | Pilot/Default task seçimleri |
| `src/labs/lab05_model_routing/routing.py` | Rule-based model selection |
| `src/labs/lab05_model_routing/matrix.py` | Profile × Strategy deney planı |
| `src/labs/lab05_model_routing/runner.py` | Gerçek API çağrısı ve ölçüm |
| `src/labs/lab05_model_routing/evaluation.py` | Task-level deterministic değerlendirme |
| `src/labs/lab05_model_routing/stats.py` | Strategy comparison ve quality regression |
| `src/labs/lab05_model_routing/cost.py` | Input/output maliyet ayrımı |
| `src/labs/lab05_model_routing/charts.py` | Otomatik grafikler |
| `src/labs/lab05_model_routing/report.py` | Analiz raporu |
| `config/default.toml` | LAB05 default ayarları |

**Önerilen repository konumu:**

```text
docs/participant/lab05-model-routing.md
```

Ana `docs/participant/README.md` içine eklenebilecek bağlantı:

```markdown
## LAB05 — Model Routing

[Open LAB05 Participant Guide](lab05-model-routing.md)
```
