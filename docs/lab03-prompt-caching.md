# LAB03 — Prompt Caching & Cache-Friendly Prompt Design

> **Katılımcı rehberi** · Groq API · Python · Docker Compose\
> **Kapsam:** `repeated_prefix`, `cache_unfriendly`, `cache_probe`, maliyet analizi ve cache gözlemlenebilirliği.\
> **Amaç:** Bir LLM API çağrısında tekrar eden prompt prefix'inin token maliyetini ve gecikmeyi nasıl etkileyebildiğini, gerçek provider usage verileriyle ölçmek.

## 1. Öğrenme hedefleri

LAB03'ün sonunda aşağıdaki soruları ölçüm sonuçlarına dayanarak cevaplayabilmelisiniz:

1. **Prompt caching** nedir, hangi input parçaları tekrar kullanılabilir?
2. **Cache-friendly prompt design** ile `cache_unfriendly` prompt tasarımı arasındaki fark nedir?
3. `prompt_tokens` ile `cached_tokens` arasındaki ilişki nedir?
4. Bir istekte **cache hit** oluştuğunu nasıl doğrularız?
5. `CACHE_MISS` ile `CACHE_USAGE_UNAVAILABLE` neden aynı değildir?
6. **Request Cache Hit Rate** ile **Token Cache Ratio** neden farklı metriklerdir?
7. Cache kullanımının **estimated input cost** ve **total request cost** üzerindeki etkisi nedir?
8. Cache hit gözlemlendiğinde latency mutlaka daha düşük mü çıkar?
9. Tekrar eden isteğin gerçekten aynı olduğunu **hash** ile nasıl doğrularız?
10. Gerçek API ölçümleriyle ilgili hangi sonuçları çıkarabilir, hangilerini çıkaramayız?

> **Temel ilke:** Cache-friendly bir prompt hazırlamak, provider'ın kesinlikle cache hit döndüreceği anlamına gelmez. LAB03, caching'i *zorlamaz*; gerçekleşen durumu ölçer.

---

## 2. Prompt caching nedir?

LLM API isteklerinde çoğu zaman tekrar eden sabit bilgiler bulunur:

- System instructions
- Uzun kurum politikaları ve knowledge base içerikleri
- Tool tanımları ve JSON şemaları
- Few-shot örnekler
- Aynı doküman üzerinden sorulan farklı sorular

Bu içerikler sonraki isteklerde **aynı prefix** olarak kalıyorsa, desteklenen modellerde provider önceki hesaplamanın bir kısmını yeniden kullanabilir.

Groq'un desteklenen GPT-OSS modellerinde prompt caching **otomatik** çalışır; bu laboratuvarda bir `cache_control` parametresi gönderilmez. Cache hit için prefix eşleşmesi, uygun uzunluk ve provider tarafındaki geçici cache durumu önemlidir. Bu mekanizma **son cevabı kopyalamak** değil, input işlemenin bir bölümünü yeniden kullanmaktır.

**Cache kullanımını ölçmek için gereken alanlar:**

```text
usage.prompt_tokens
usage.prompt_tokens_details.cached_tokens
usage.completion_tokens
usage.total_tokens
```

`cached_tokens`, `prompt_tokens` değerinin **içindedir**. Aynı tokenları iki kez toplamamalısınız.

### 2.1 Örnek — token dağılımı

Aşağıdaki sayılar **temsili**, gerçek deney çıktısı değildir:

```text
prompt_tokens:       4,000
cached_tokens:       3,000
uncached_tokens:     1,000
completion_tokens:     250
total_tokens:        4,250
```

Burada:

\[
\text{Uncached Input Tokens}=4{,}000-3{,}000=1{,}000
\]

\[
\text{Request Cache Ratio}=\frac{3{,}000}{4{,}000}\times100=75\%
\]

Toplam token kullanımı 4.250'dir; **cached tokenlar input tokenlarından düşülerek total usage azaltılmış gibi gösterilmez**. Etki, aynı input'un bir bölümünün farklı maliyetle işlenebilmesidir.

### 2.2 Groq'a özgü sınırlar

LAB03'ün mevcut konfigürasyonu caching karşılaştırmasını varsayılan olarak `openai/gpt-oss-20b` üzerinde yapar. Alternatif olarak `openai/gpt-oss-120b` konfigürasyonda listelenmiştir.

Groq dokümantasyonuna göre:

- Caching otomatik çalışır; API çağrısında özel aktivasyon gerekmez.
- Başarılı cache hit, cached input tokenları için indirim sağlayabilir.
- Cache geçicidir; süresiz saklanmaz (dokümantasyonda kullanılmadan **2 saat** sonra expiration belirtilir).
- Cache hit **garanti değildir**.
- Cacheable minimum prompt uzunluğu modele göre değişebilir; tek bir sabit eşik varsaymayın.
- Promptun ilk karakterlerinden itibaren değişen içerik aynı prefix'in yeniden kullanımını engelleyebilir.

Referans: [Groq — Prompt Caching](https://console.groq.com/docs/prompt-caching)

---

## 3. LAB03'ün deney tasarımı

Bu laboratuvar iki farklı deney biçimi içerir:

| Deney | CLI | Amacı |
|---|---|---|
| **Paired Comparison** | `lab03 all` | Sabit prefix ile değişken prefix'i karşılaştırmak |
| **Exact-Request Probe** | `lab03 cache_probe` | Birebir aynı API isteklerinde cache kullanımını gözlemlemek |

Bunların yanında `repeated_prefix` ve `cache_unfriendly` koşulları ayrı ayrı çalıştırılabilir.

### 3.1 Condition A — `repeated_prefix`

Sistemin aynı kurum knowledge base'ini farklı kullanıcılara sunduğunu düşünün. Her çağrıdaki **system message** şu sırada oluşturulur:

```text
[STABLE SESSION TAG]
[STATIC ACME CLOUD KNOWLEDGE BASE]
[STATIC RESPONSE RULES]

[USER MESSAGE: QUESTION N]
[ANSWER FORMAT INSTRUCTION]
```

**Önemli:** Değişen kullanıcı sorusu daha sonda yer alırken uzun knowledge base ve kurallar sabit kalır. Prefix, sonraki isteklerde tekrar kullanılmaya elverişlidir.

### 3.2 Condition B — `cache_unfriendly`

Bu koşulda aynı knowledge base ve aynı cevap kuralları korunur; ancak **system message'ın en başına**, her çift için değişen aynı uzunlukta bir session tag yerleştirilir:

```text
[CHANGING SESSION TAG]
[STATIC ACME CLOUD KNOWLEDGE BASE]
[STATIC RESPONSE RULES]

[USER MESSAGE: QUESTION N]
[ANSWER FORMAT INSTRUCTION]
```

Tag'in uzunluğu iki koşulda aynıdır. Temel fark, ilk karakterlerden itibaren değişmesi ve dolayısıyla eşleşebilecek leading prefix'i bozmasıdır.

Her paired turda **iki koşul aynı soruyu cevaplar**. Böylece cevap görevindeki fark kontrol altında tutulur.

> Bu koşul Groq'un caching mekanizmasını **devre dışı bırakmaz**. Sadece bu deneyin sabit knowledge base'i için prefix tekrarını elverişsiz hale getirir. Provider'ın başka, önceden var olan cache durumları tamamen kontrol edilemez.

### 3.3 Condition C — `cache_probe`

Bu modda yalnızca prefix değil, **tam istek** tekrar edilir:

```text
Request 1: [Same Model][Same System][Same User Question][Same Parameters]
Request 2: [Same Model][Same System][Same User Question][Same Parameters]
Request 3: [Same Model][Same System][Same User Question][Same Parameters]
```

Uygulama gerçek request payload'larından SHA-256 hash üretir ve **byte-identical** olduklarını offline kontrol eder.

Ancak:

- **Aynı hash:** Aynı request oluşturulduğunu gösterir.
- **`cached_tokens > 0`:** Gerçek bir provider-reported cache hit kanıtıdır.
- **Aynı hash + `cached_tokens` yok:** Cache hit gözlemlendiğini kanıtlamaz.

Bu nedenle `cache_probe`, daha güçlü bir kontrol deneyi olsa da provider'ın cache hit üretmesi garanti değildir.

### 3.4 Kullanılan soru ve bilgi tabanı

Kurum senaryosu, sentetik **ACME CLOUD Enterprise Customer Support Knowledge Base** üzerinedir. Dört soru dönüşümlü kullanılır:

| Policy | Soru konusu | Beklenen cevap |
|---|---|---:|
| Section 2 — Refund Policy | Onaylı iadenin maksimum iş günü | `5` |
| Section 3 — Escalation Policy | İnsan temsilciye eskalasyona yol açan ardışık başarısız deneme | `2` |
| Section 4 — SLA Definitions | Standard-priority ilk yanıt süresi (saat) | `24` |
| Section 5 — Ticket Priority Rules | Tanımlı öncelik seviyesi sayısı | `3` |

Modelden **yalnızca tek bir tam sayı** döndürmesi istenir.

`all` modunda her eşleştirilmiş tur için iki condition aynı soruyu kullanır; `cache_probe` ise her çağrıda **ilk soruyu** aynen tekrarlar.

---

## 4. Deney profilleri ve API çağrı sayısı

| Uygulama | Docker CLI seçeneği | Planlanan mantıksal çağrı |
|---|---|---:|
| Pilot | `all --repetitions 2` | 4 |
| **Default Grand Test** | `all` | **6** |
| Extended | `all --repetitions 4` | 8 |
| Repeated Prefix Only | `repeated_prefix` | 3 |
| Cache-Unfriendly Only | `cache_unfriendly` | 3 |
| Exact-Request Probe | `cache_probe` | 3 |

Formül (`all`):

\[
\text{Planned Logical Calls}=\text{Repetitions}\times2
\]

Varsayılan `lab03.max_api_calls = 8` değeridir. `--repetitions 5` ile 10 çağrılık bir `all` planı oluşturmak canlı yürütmede bu limite takılır.

**Logical call**, ders senaryosundaki bir API işlemidir. HTTP 429 gibi hatalarda operational retry gerçekleşirse gerçek HTTP denemesi sayısı daha yüksek olabilir.

### 4.1 Paired ve interleaved çalışma sırası

Varsayılan altı çağrı şöyle planlanır:

```text
Pair 1: repeated_prefix   -> Question 1 (expected 5)
Pair 1: cache_unfriendly  -> Question 1 (expected 5)

Pair 2: repeated_prefix   -> Question 2 (expected 2)
Pair 2: cache_unfriendly  -> Question 2 (expected 2)

Pair 3: repeated_prefix   -> Question 3 (expected 24)
Pair 3: cache_unfriendly  -> Question 3 (expected 24)
```

Bu yapı, önce bir koşulun bütün çağrılarını yapıp ardından diğerine geçmek yerine koşulları **eşleştirerek sırayla çalıştırır**. Böylece provider yükündeki zamanla ilişkili değişikliklerin tek bir koşula yığılmasını azaltmaya çalışır; tamamen ortadan kaldırmaz.

---

## 5. Docker kurulumu — Windows PowerShell

Komutları repository kök klasöründe çalıştırın. Python veya `uv` bilgisayarda ayrıca kurulu olmak zorunda değildir; Docker gereklidir.

### 5.1 Proje klasörüne geçin

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Projeniz başka dizindeyse yolu değiştirin.

### 5.2 Groq API anahtarı

```powershell
Copy-Item .env.example .env
```

`.env` içine kendi anahtarınızı koyun:

```dotenv
GROQ_API_KEY=your_api_key_here
```

**API anahtarını GitHub'a commit etmeyin, kimseyle paylaşmayın.**

### 5.3 Güvenli live config oluşturun

```powershell
Copy-Item config/default.toml config/live.toml
```

`config/live.toml` içindeki mevcut `[execution]` bölümünde şu ayarı değiştirin:

```toml
live_api_enabled = true
```

Dosyanın diğer bölümlerini koruyun. Yeni bir ikinci `[execution]` bölümü eklemeyin.

LAB03'ün live çalışması **iki şartın birlikte sağlanmasına** bağlıdır:

1. CLI'de `--live` verilmesi.
2. Seçilen konfigürasyonda `execution.live_api_enabled = true` olması.

### 5.4 Docker image oluşturun

```powershell
docker compose build lab-runner
```

`config/live.toml` image içine build sırasında kopyalanır. Dosyayı oluşturduktan veya değiştirdikten sonra **yeniden build** edin.

### 5.5 CLI parametrelerini inceleyin

```powershell
docker compose run --rm lab-runner lab03 --help
```

Bu komut gerçek Groq API çağrısı oluşturmaz.

---

## 6. LAB03 Docker komutları

### A — Önce Dry Run (önerilen)

```powershell
docker compose run --rm lab-runner lab03 all --dry-run --config config/default.toml
```

- **6 mantıksal çağrı** planlar.
- Planı `experiment_plan.csv` dosyasına kaydeder.
- API key gerektirmez, API çağrısı yapmaz.
- Gerçek cache hit, latency, maliyet veya token kullanımını ölçmez.

### B — Default Grand Test (6 çağrı)

```powershell
docker compose run --rm lab-runner lab03 all --live --config config/live.toml
```

Bu, öğrencilerin iki condition'ı birlikte karşılaştırması için ana deneydir.

### C — Pilot (4 çağrı)

```powershell
docker compose run --rm lab-runner lab03 all --repetitions 2 --live --config config/live.toml
```

API kullanımını azaltan ilk canlı kontrol için uygundur.

### D — Extended (8 çağrı)

```powershell
docker compose run --rm lab-runner lab03 all --repetitions 4 --live --config config/live.toml
```

Bir ek paired soru ile deney kapsamını genişletir. Daha fazla çağrı, cache hit'i garanti etmez.

### E — Cache-friendly condition

```powershell
docker compose run --rm lab-runner lab03 repeated_prefix --live --config config/live.toml
```

Varsayılan **3 çağrı**.

### F — Cache-unfriendly condition

```powershell
docker compose run --rm lab-runner lab03 cache_unfriendly --live --config config/live.toml
```

Varsayılan **3 çağrı**.

### G — Exact-request cache probe (3 çağrı)

**LAB03'ün cache gözlemlenebilirliği için en önemli ikinci deneyi:**

```powershell
docker compose run --rm lab-runner lab03 cache_probe --live --config config/live.toml
```

Bu modda arka arkaya gönderilen üç request birebir aynıdır. Terminalde request hash ve prefix stability kontrolünü de göreceksiniz.

### H — Cache probe için güvenli Dry Run

```powershell
docker compose run --rm lab-runner lab03 cache_probe --dry-run --config config/default.toml
```

API çağrısı yapmadan `All 3 planned requests byte-identical: True` sonucunu doğrulamayı deneyin. Bu sonuç **gerçek cache hit** kanıtı değildir.

### I — Belirli bir run ID

```powershell
docker compose run --rm lab-runner lab03 all --experiment-run-id lab03-class-group-a --live --config config/live.toml
```

Sonuç dizini:

```text
results/lab03_prompt_caching/lab03-class-group-a/
```

LAB03'te aynı `experiment-run-id` ile mevcut dolu sonuç klasörüne tekrar kayıt yapma girişimi reddedilir; her deney için yeni bir ID seçin.

---

## 7. Terminal çıktısını yorumlamak

Her çağrı için LAB03 aşağıdaki ölçümleri raporlar. Aşağıdaki şema **örnek format**tır; sayılarınız provider'dan gelecektir.

```text
Condition: repeated_prefix
Run: 2/3

Prompt Tokens:          ...
Cached Tokens:          ...
Uncached Prompt Tokens: ...
Cache Hit:              ...
Cache Status:           ...
Cache Ratio:            ...%
Completion Tokens:      ...
Reasoning Tokens:       ...
Latency:                ... ms
Estimated Cost:         $...
Estimated Cache Savings:$...
Finish Reason:          ...
Answer Correct:         True (correct)
```

Öncelikle şunlara bakın:

- `Prompt Tokens` — isteklerin input büyüklüğü birbirine yakın mı?
- `Cached Tokens` — provider cache kullanımını gerçekten raporladı mı?
- `Cache Status` — hit, miss, unavailable veya geçersiz usage mı?
- `Estimated Cache Savings` — cache kullanımının tahmini maliyet etkisi nedir?
- `Answer Correct` — beklenen sayısal cevap döndü mü?
- `Latency` — ölçülen client elapsed time nasıl değişti?

### 7.1 Dört farklı Cache Status

| Status | Anlamı | Nasıl yorumlanır? |
|---|---|---|
| `CACHE_HIT` | `cached_tokens > 0` ve raporlanan usage tutarlı | **Gözlemlenen cache hit** |
| `CACHE_MISS` | `cached_tokens = 0` raporlanmış | Provider cache kullanımını raporlamış, bu istek için hit yok |
| `CACHE_USAGE_UNAVAILABLE` | `cached_tokens` alanı hiç raporlanmamış | **Hit yok** demek değildir; ölçüm bulunmuyor |
| `INVALID_USAGE` | Örneğin `cached_tokens > prompt_tokens` | Usage tutarsız, normal hit/miss olarak değerlendirmeyin |

Bunun iki ayrı örneğini düşünün:

```json
{"prompt_tokens": 4200, "prompt_tokens_details": {"cached_tokens": 0}}
```

Bu sonuç: **`CACHE_MISS`**.

```json
{"prompt_tokens": 4200}
```

Bu sonuç: **`CACHE_USAGE_UNAVAILABLE`**.

İkinci örnekte `cached_tokens` eksik olduğu için değeri kendiniz **0'a çevirmemelisiniz**.

### 7.2 `Cache Verification: OBSERVED / NOT OBSERVED`

`cache_probe` veya başka bir canlı LAB03 çalışmasında `Cache Verification` raporunu görebilirsiniz:

- **`OBSERVED`**: En az bir başarılı çağrıda gerçek `cached_tokens > 0` raporlanmıştır.
- **`NOT OBSERVED`**: Bu çalışmada hit doğrulanamamıştır; detay satırı `CACHE_MISS` mi yoksa usage eksikliği mi olduğunu açıklar.

`NOT OBSERVED`, cache'in ürün genelinde çalışmadığına veya kodun bozuk olduğuna dair kanıt değildir.

---

## 8. Cache hit rate ve token cache ratio farkı

### 8.1 Request Cache Hit Rate

Bu metrik, cache kullanım verisi raporlanmış **başarılı çağrılar** arasında en az bir cached token içeren isteklerin oranıdır.

\[
\text{Observed Request Hit Rate}=
\frac{\#(cached\_tokens>0)}{\#(cached\_tokens\ reported)}\times100
\]

Örnek: 4 başarılı çağrının üçünde `cached_tokens` alanı mevcut, bu üçünün ikisinde >0 ise:

\[
\frac{2}{3}\times100=66.7\%
\]

Eksik usage raporu paydada **0-hit kabul edilmez**.

### 8.2 Token Cache Ratio

Bu metrik, usage raporlayan başarılı çağrılarda cached input token toplamının prompt token toplamına oranıdır.

\[
\text{Token Cache Ratio}=
\frac{\sum cached\_tokens}{\sum prompt\_tokens}\times100
\]

Bu iki metrik aynı şeyi ölçmez:

- Request Hit Rate = **Kaç istekte cache kullanıldı?**
- Token Cache Ratio = **Raporlanan input tokenlarının ne kadarı cached?**

Bir istekte yalnızca az sayıda cached token olsa bile o istek `CACHE_HIT` olarak sayılır. Bu nedenle %100 request hit rate, %100 token cache ratio anlamına gelmez.

---

## 9. Maliyet hesaplama — gerçek deneyin ana kısmı

### 9.1 Cache-aware input maliyeti

`P_input` ve `P_cached` değerlerinin **milyon token başına USD** olduğunu düşünün:

\[
C_{input}=\frac{T_{uncached}P_{input}+T_{cached}P_{cached}}{1{,}000{,}000}
\]

Tam request'in tahmini maliyeti:

\[
C_{total}=C_{input}+\frac{T_{completion}P_{output}}{1{,}000{,}000}
\]

**Reasoning tokens**, completion tokens içindedir; `completion_tokens` üzerine ayrıca eklemeyin.

### 9.2 Without-cache counterfactual

LAB03, aynı ölçülmüş usage için *cache uygulanmamış olsaydı* input maliyetinin ne olacağını da hesaplar:

\[
C_{input,without\ cache}=\frac{T_{prompt}P_{input}}{1{,}000{,}000}
\]

\[
\text{Estimated Cache Savings}=C_{input,without\ cache}-C_{input,with\ cache}
\]

Bu **ikinci bir API çağrısıyla ölçülmüş alternatif senaryo değil**, tek çağrının aynı token kullanımı üzerinde yapılan tahmini yeniden fiyatlandırmadır.

### 9.3 Sayısal örnek — tamamı varsayımsal

Şu fiyatları varsayalım (güncel fiyat iddiası değildir):

```text
Input price:        $0.10 / 1M tokens
Cached input price: $0.05 / 1M tokens
Output price:       $0.30 / 1M tokens
```

Şu usage'ı varsayalım:

```text
prompt_tokens:     4,000
cached_tokens:     3,000
completion_tokens:   250
```

Hesap:

```text
Uncached input tokens = 1,000
With-cache input cost = $0.00025
Without-cache input cost = $0.00040
Estimated input savings = $0.00015
Output cost = $0.000075
Estimated total request cost = $0.000325
```

Dikkat: **Input maliyet tasarrufu oranı**, toplam request maliyetinden elde edilen tasarruf oranıyla aynı olmak zorunda değildir. Output maliyeti değişmeden kalabilir.

### 9.4 Eksik veri ne anlama gelir?

`cached_tokens` raporlanmadıysa LAB03 cache tasarrufunu kesin olarak hesaplayamaz; tasarruf değeri **N/A** kalmalıdır. Projedeki tahmini maliyet hesabı, cached usage eksikken ihtiyatlı olarak tüm input'u uncached fiyatla değerlendirebilir; bu **gerçek cache miss kanıtı** değildir.

---

## 10. Latency yorumlama

LAB03'te bazı ölçümler:

| Metrik | Açıklama |
|---|---|
| `total_latency_ms` | İstemcinin tüm çağrıda gözlemlediği elapsed time |
| `provider_prompt_time_ms` | Provider raporlarsa prompt processing time |
| `provider_completion_time_ms` | Provider raporlarsa completion processing time |
| `provider_queue_time_ms` | Provider raporlarsa queue time |
| `reasoning_tokens` | Modelin completion içindeki reasoning overhead'i |

Cache hit, input processing zamanını azaltabilir; fakat **tüm request'in latency'sinin mutlaka daha düşük olacağı anlamına gelmez**.

Özellikle:

- Network gecikmesi
- Queueing
- Output token sayısı
- Reasoning token sayısı
- Provider'ın anlık yükü

sonuçları etkileyebilir.

Bu nedenle tek bir başarılı cache hit üzerinden “cache latency'yi kesin olarak X ms düşürdü” demeyin. Aynı koşul için daha çok tekrarlı ve kontrollü ölçüm gerekir.

---

## 11. Answer Correctness — Cache performansı kaliteyi bozdu mu?

LAB03, her koşulda beklenen tam sayı cevabı değerlendirir.

Örnek:

| Model cevabı | Sonuç |
|---|---|
| `5` (beklenen 5 ise) | `correct` |
| `4` (beklenen 5 ise) | `incorrect` |
| `Refunds take 5 days.` | `format_invalid` |
| Boş cevap | `missing_response` |

Cache hit tek başına cevap kalitesini ölçmez. Doğru değerlendirme için:

1. API çağrısı başarılı mı?
2. Çıktı istenen formatta mı?
3. Cevap beklenen sonuca eşit mi?

sorularını ayrı ayrı cevaplayın.

---

## 12. Sonuç dosyalarının konumu

Her live run'ın ayrı bir klasörü vardır:

```text
results/
└── lab03_prompt_caching/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── cache_measurements.json
        ├── cache_measurements.csv
        ├── cache_diagnostics.json
        ├── condition_summary.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Kullanım amacı |
|---|---|
| `experiment_manifest.json` | Model, konfigürasyon, yürütme ve versiyon bilgileri |
| `experiment_plan.csv` | Çağrı sırası, condition, pair, question index |
| `cache_measurements.json/.csv` | Her request'in token, latency, status ve cost ölçümleri |
| `cache_diagnostics.json` | Cache usage field presence, request hash gibi tanılama verileri |
| `condition_summary.csv` | Condition bazlı özet karşılaştırmalar |
| `experiment_summary.json` | Toplam sonuçlar ve grafik üretiminde kullanılan ölçümler |
| `analysis_report.md` | Yorumlama, karşılaştırma ve deneysel sınırlamalar |
| `charts/` | Ölçümlerden türetilmiş PNG grafikler |

**Önerilen inceleme sırası:**

1. `analysis_report.md` — deneyin ne söylediğini okuyun.
2. `condition_summary.csv` — iki condition'ı kıyaslayın.
3. `cache_measurements.csv` — çağrı bazında `cached_tokens`, `cache_status`, latency ve maliyeti inceleyin.
4. `cache_diagnostics.json` — usage alanı hiç gelmemiş mi, 0 mı gelmiş kontrol edin.
5. `charts/` — gözlenen ilişkileri görselleştirin.

İstek/cevap metinlerinin tamamını raw içerik olarak saklamaz; LAB03 esas olarak usage ve ölçüm metadata'sını toplar.

---

## 13. Grafikler — Neye bakılmalı?

LAB03, yeterli ölçüm varsa aşağıdaki grafiklerden uygun olanları üretir:

| Grafik dosyası | Nasıl yorumlanır? |
|---|---|
| `cached_vs_uncached_tokens.png` | Hangi input token kısmı cached/uncached raporlanmış? |
| `cache_status_by_request.png` | Request sırasına göre HIT/MISS/UNAVAILABLE/INVALID dağılımı |
| `cache_ratio_by_request.png` | Tekil çağrılardaki cache ratio değişimi |
| `cache_friendly_vs_unfriendly_comparison.png` | İki condition'ın cache metrikleri |
| `probe_sequence.png` | Exact-request probe çağrılarının sıralı sonucu (probe çalıştıysa) |
| `estimated_cost_by_condition.png` | Condition bazında tahmini maliyet |
| `estimated_cost_savings.png` | Cache sayesinde hesaplanan tahmini input tasarrufu |
| `latency_by_condition.png` | İki condition arasındaki latency karşılaştırması |
| `cached_tokens_vs_latency.png` | Cached tokens ve latency'nin birlikte gözlemlenmesi |
| `summary_dashboard.png` | Deneyin toplu görünümü |

**Not:** Grafik sayısı deney moduna ve kullanılabilir verilere bağlıdır. `cache_probe` yalnızca bir condition çalıştırır; bazı paired comparison grafiklerinin anlamı sınırlı olabilir.

### 13.1 Grafik okurken sorulacak sorular

- `cached_tokens` gerçekten 0'dan büyük mü, yoksa N/A mı?
- İki koşulun `prompt_tokens` büyüklükleri benzer mi?
- `cached_tokens` yüksek olan çağrıların `estimated_cost_usd` değeri nasıl değişmiş?
- Latency farkını açıklayabilecek output/reasoning farklılığı var mı?
- `CACHE_USAGE_UNAVAILABLE` için grafikte bilinmeyen değerler sıfır gibi mi gösterilmiş? **Gösterilmemeli.**

---

## 14. Offline chart backfill

Daha önceki sonuç dosyalarından grafikleri yeniden üretmek için:

```powershell
docker compose run --rm lab-runner lab03 backfill --config config/default.toml
```

Tek bir run için:

```powershell
docker compose run --rm lab-runner lab03 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasörünün gerçek adını yazın.

Backfill:

- Gerçek API çağrısı yapmaz.
- API key gerektirmez.
- Ölçüm JSON/CSV ve rapor Markdown dosyalarını değiştirmez.
- Var olan sonuçlardan PNG grafikleri yeniden oluşturur.

`--run-id` olmadan eski flat-file sonuçlar için legacy chart backfill da çalışabilir.

---

## 15. Sınıf uygulaması — Ölç, karşılaştır, yorumla

### Görev 1 — Cache-friendly vs cache-unfriendly

Şu komutla başlayın:

```powershell
docker compose run --rm lab-runner lab03 all --live --config config/live.toml
```

Sonuçlarla tabloyu doldurun:

| Metrik | `repeated_prefix` | `cache_unfriendly` |
|---|---:|---:|
| Successful Calls | | |
| Observed Request Hit Rate | | |
| Token Cache Ratio | | |
| Total Cached Tokens | | |
| Mean Client Latency (ms) | | |
| Total Estimated Cost (USD) | | |
| Estimated Cache Savings (USD) | | |
| Correct Answers | | |

**Tartışma:** Sabit prefix kullanmak her zaman bir cache hit göstermiş mi? Göstermediyse sonuç `CACHE_MISS` mi, `CACHE_USAGE_UNAVAILABLE` mı?

### Görev 2 — Exact-request cache probe

```powershell
docker compose run --rm lab-runner lab03 cache_probe --live --config config/live.toml
```

Aşağıdaki alanları inceleyin:

```text
All planned requests byte-identical
Request Hash
Cache Verification
Cached Tokens
Cache Status
```

**Tartışma:** Hash'ler aynıyken `Cache Verification: NOT OBSERVED` görülebilir mi? Bu durum neden otomatik olarak yazılım hatası değildir?

### Görev 3 — Bir çağrının cache ekonomisini hesaplayın

`cache_measurements.csv` dosyasından `cached_tokens` alanı raporlanmış başarılı bir çağrı seçin. Mevcut fiyat konfigürasyonuna göre şu değerleri doğrulayın:

```text
prompt_tokens
cached_tokens
uncached_prompt_tokens
estimated_input_cost_without_cache_usd
estimated_input_cost_with_cache_usd
estimated_cache_savings_usd
estimated_cost_usd
```

**Tartışma:** Total request cost ile input-only savings'i neden karıştırmamalıyız?

### Görev 4 — Latency ve doğruluk

Paired condition'ların latency, completion token, reasoning token ve doğru cevap değerlerini karşılaştırın.

**Tartışma:** Daha yüksek cache ratio, daha düşük total latency'yi garantiliyor mu? Veriniz hangi yorumu destekliyor?

### Görev 5 — Metriklerin yanlış yorumlanması

Şu üç ifadeyi değerlendirin:

1. “`cached_tokens = null` ise cache miss oluşmuştur.”
2. “`cached_tokens = 3000` ise toplam prompt uzunluğundan 3000 token düşer.”
3. “Aynı SHA-256 request hash'leri, provider cache hit'inin ispatıdır.”

**Üçü de yanlıştır.** Her ifadenin neden yanlış olduğunu ölçümlere referansla açıklayın.

---

## 16. Deneysel sınırlamalar

Bu laboratuvarın default Grand Test'i yalnızca **3 paired repetition (6 çağrı)** içerir. Bu, öğretim açısından yeterli bir başlangıçtır fakat istatistiksel olarak güçlü bir benchmark değildir.

Şu sınırlamaları raporunuzda belirtin:

- Provider tarafındaki cache yönetimi kontrol edilemez.
- Önceden oluşturulmuş veya süresi dolmuş cache sonuçları etkileyebilir.
- `CACHE_USAGE_UNAVAILABLE` durumunda hit/miss kesin olarak anlaşılamaz.
- Eşleştirme ve interleaving, sağlayıcının anlık yük farklarını tamamen ortadan kaldırmaz.
- Completion length, reasoning token kullanımı ve queue time değişebilir.
- Eşit sayıda request, her koşul için eşit provider CPU/GPU iş yükünü garanti etmez.
- Maliyetler konfigüre fiyatlarla hesaplanan **tahmini** değerlerdir.
- Provider yanıtında eksik olan ölçümler uydurulmaz veya otomatik 0 kabul edilmez.

**Doğru sonuç cümlesi örneği:**

> Bu LAB03 çalıştırmasında `repeated_prefix` koşulundaki raporlanmış cached token oranı `cache_unfriendly` koşulundan yüksek ölçülmüştür. Gözlemlenen tahmini maliyet farkı ve latency değerleri raporda ayrıca incelenmelidir.

**Yanlış genelleme örneği:**

> Prompt caching her zaman çalışır ve her API çağrısını kesin olarak %50 daha ucuza getirir.

Cache indirimleri yalnızca gerçekten cached olarak işlenen uygun input tokenları için geçerli olabilir; toplam request maliyetini de aynı oranda azaltmak zorunda değildir.

---

## 17. Troubleshooting

### 17.1 `LiveExecutionNotAllowed`

- `--live` parametresini kontrol edin.
- `config/live.toml` içinde `live_api_enabled = true` olduğundan emin olun.
- `docker compose build lab-runner` ile image'ı yeniden oluşturun.

### 17.2 `config/live.toml` bulunamıyor

```powershell
Copy-Item config/default.toml config/live.toml
docker compose build lab-runner
```

Ardından dosyada yalnızca mevcut `[execution]` içindeki `live_api_enabled` değerini değiştirin.

### 17.3 `CACHE_USAGE_UNAVAILABLE`

Bu, alanın provider response içinde raporlanmadığı anlamına gelir. Kod `prompt_tokens_details.cached_tokens` alanını inceleyebilir; raporlanmaması tek başına parsing hatası sayılmaz.

Önce `cache_diagnostics.json` içindeki field-presence bilgilerini inceleyin. Üst üste aynı isteği tekrar tekrar çağırarak kota tüketmeyin.

### 17.4 `Cache Verification: NOT OBSERVED`

Gözlemlenen hit yoktur. Bu bir yazılım başarısızlık kodu değildir. `CACHE_MISS` ile `CACHE_USAGE_UNAVAILABLE` ayrımını inceleyin ve öğretici sonucu not edin.

### 17.5 `planned calls exceed lab03.max_api_calls`

Varsayılan limit 8 mantıksal çağrıdır. `all --repetitions 5`, 10 çağrı planlar ve sınırı aşar. Pilot veya default moda dönün.

### 17.6 HTTP 429 / Rate limit

Groq kota ve rate limit kuralları uygulanır. Kullanılan model, hesap ve kalan limit önemlidir. Küçük pilot seçin; yoğun art arda tekrarlar oluşturmayın.

### 17.7 `estimated_cost_usd` veya savings için `N/A`

Fiyat konfigürasyonu veya provider usage alanı eksik olabilir. **N/A = 0** değildir. Özellikle `cached_tokens` raporlanmadığında savings hesaplaması eksik kalabilir.

### 17.8 Grafik oluşturulmadı

Ölçümlerin bulunduğunu kontrol edin ve `backfill` deneyin. Bir condition veya başarısız çağrılarla bazı karşılaştırmalar üretilemeyebilir.

### 17.9 Aynı run ID daha önce kullanılmış

Yeni `--experiment-run-id` seçin. LAB03 bitmiş ölçüm klasörüne otomatik append/resume yapmaz.

---

## 18. LAB02 ve LAB10 ile bağlantı

**LAB02 — Context Growth & Context Bloat:** Context uzadıkça `prompt_tokens` ve cumulative token tüketimi artabilir. Bunun maliyeti ve latency etkisi incelenir.

**LAB03 — Prompt Caching:** Context'i mutlaka küçültmeden, **tekrar kullanılan sabit prefix'in** daha ekonomik işlenip işlenmediği ölçülür.

**LAB10 — Conversation Summarization:** Eski konuşma daha kısa bir summary ile temsil edilir; summary üretiminin ek maliyeti ve bilgi koruma kalitesi değerlendirilir.

Bu üç yöntem aynı değildir:

| LAB | Temel soru |
|---|---|
| LAB02 | History / context büyürse ne olur? |
| **LAB03** | Aynı context tekrar kullanıldığında cache tasarrufu oluşur mu? |
| LAB10 | Context summary ile küçültülürse toplam maliyet ve kalite ne olur? |

---

## 19. LAB03 tamamlandığında

Aşağıdaki çıkarımları kendi deney ölçümlerinizle destekleyebilmelisiniz:

- **Stable prefix**, cache-friendly tasarımın önemli bir parçasıdır.
- **Identical request hash**, request eşitliğini gösterir; **cache hit'i ispatlamaz**.
- **`cached_tokens > 0`**, raporlanan geçerli usage için gerçek cache hit gözlemidir.
- **`CACHE_MISS` ve `CACHE_USAGE_UNAVAILABLE`** birbirinden ayrılmalıdır.
- **`cached_tokens`, `prompt_tokens` içinde bulunur**; iki kez sayılmaz.
- **Request Hit Rate** ile **Token Cache Ratio** farklıdır.
- **Estimated cache savings**, aynı ölçümün counterfactual fiyatlandırmasıdır; ikinci bir API çağrısı değildir.
- Cache tasarrufunun input üzerindeki etkisi, tüm request maliyetinden farklı olabilir.
- Latency farkı tek başına cache'e nedensel olarak atfedilemez.
- Gerçek API verisi görülmeden “cache çalıştı/çalışmadı” sonucuna varılmaz.

> **Temel çıkarım:** Prompt caching, input token miktarını ortadan kaldıran bir yöntem değildir. Desteklenen sağlayıcı/model koşullarında tekrar kullanılan prefix'in hesaplama ve fiyatlandırma maliyetini azaltabilen bir mekanizmadır. Etkisi yalnızca **provider-reported usage**, güvenilir cost accounting ve deneysel sınırlamalar birlikte değerlendirilerek anlaşılır.

---

## Kaynak kod ve ileri okuma

Repository içindeki ilgili dosyalar:

- `src/labs/lab03_prompt_caching/experiments.py` — Prompt fixture, paired questions, prefix kurgusu
- `src/labs/lab03_prompt_caching/matrix.py` — Paired/interleaved execution plan
- `src/labs/lab03_prompt_caching/cache_probe.py` — Byte-identical request probe
- `src/labs/lab03_prompt_caching/runner.py` — Groq API ölçümleri
- `src/labs/lab03_prompt_caching/cache_status.py` — Dört durumlu cache classification
- `src/labs/lab03_prompt_caching/cache_metrics.py` — Request hit rate ve token ratio
- `src/labs/lab03_prompt_caching/cache_cost.py` — Cache-aware cost ve estimated savings
- `src/labs/lab03_prompt_caching/diagnostics.py` — Request hash ve usage diagnostics
- `src/labs/lab03_prompt_caching/charts.py` — Grafikler
- `src/labs/lab03_prompt_caching/report.py` — Analiz raporu
- `config/default.toml` — Varsayılan LAB03 konfigürasyonu

Resmî kaynak: [Groq Prompt Caching Documentation](https://console.groq.com/docs/prompt-caching)
