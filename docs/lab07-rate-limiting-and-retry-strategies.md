# LAB07 — Rate Limiting & Retry Strategies

> **Katılımcı rehberi** · Python · Docker Compose · Discrete-Event Simulation\
> **Kapsam:** HTTP 429, Retry, Exponential Backoff, Jitter, `Retry-After`, RPM/TPM quota pressure ve retry collisions.\
> **Ana deney:** `lab07 all --profile default` — **12 simulation run, 0 gerçek API çağrısı**.

---

## 1. Bu laboratuvarın amacı

LLM API uygulamaları bazen **HTTP 429 Too Many Requests** hatalarıyla karşılaşır. Bu hata, istemcinin sağlayıcının belirlediği rate limit veya kota sınırlarına ulaşmasıyla ilişkili olabilir.

Bir API çağrısı başarısız olduğunda **hemen tekrar istek göndermek**, özellikle çok sayıda istemci aynı davranışı gösteriyorsa, sorunu büyütebilir. Retry stratejileri başarısız isteklerin **ne zaman ve hangi koşullarda** yeniden deneneceğini belirler.

LAB07'nin temel araştırma sorusu:

> Aynı hata ve kapasite koşullarında farklı Retry stratejileri, başarı oranını, toplam istek sayısını, 429 yanıtlarını, tekrar deneme zamanlamasını ve sistemin toparlanmasını nasıl etkiliyor?

Bu deney, gerçek bir LLM çağrısının fiyatını veya hızını ölçmez. **Rate Limiting ve Retry stratejilerinin davranışını kontrollü simülasyon üzerinde gösterir.**

### 1.1 Öğrenme hedefleri

LAB07 sonunda şunları açıklayabilmelisiniz:

1. `HTTP 429` ile retry mekanizmasının ilişkisi.
2. `max_attempts` ve `retry_count` farkı.
3. `No Backoff`, `Exponential Backoff`, `Exponential Backoff with Jitter`, `Rate-Limit-Aware` stratejileri.
4. `Retry-After` header'ının geçerli, eksik veya hatalı olması durumunda nasıl davranılması gerektiği.
5. Eşzamanlı istemcilerin oluşturduğu **Thundering Herd / Retry Collision** problemi.
6. RPM ve TPM kısıtlarının **ayrı kota boyutları** olduğu.
7. `success_rate`, `total_attempts`, `total_429_responses`, `total_scheduled_wait_seconds` ve `completion_time_seconds` farkları.
8. Güvenilirlik ile verimlilik metriklerinin neden aynı puanda birleştirilmemesi gerektiği.

---

## 2. Çok önemli: LAB07 neden diğer laboratuvarlardan farklı?

LAB01–LAB06'nın birçok deneyinde gerçek Groq API çağrıları yapılıyor. LAB07 ise **tamamen offline** çalışır.

| Özellik | LAB07 davranışı |
|---|---|
| Groq API çağrısı | **Yok** |
| `GROQ_API_KEY` | **Gerekmez** |
| İnternet erişimi | Simülasyonun kendisi için gerekmez |
| Gerçek HTTP isteği | **Yok** |
| Gerçek RPM/TPM usage | **Ölçülmez** |
| Gerçek token tüketimi | **Ölçülmez** |
| Gerçek API maliyeti | **$0** |
| Simüle edilmiş HTTP 429 | **Var** |
| Virtual Clock | **Var** |
| Gerçek bekleme (`time.sleep`) | **Yok** |
| Sonuç dosyaları ve grafikler | **Var** |

LAB07'nin `--live` seçeneği eski CLI ile uyumluluk için kabul edilir, fakat **yok sayılır**. Bu parametreyi kullanarak LAB07'yi gerçek Groq testine dönüştüremezsiniz.

Bir deney çıktısında:

```text
Execution Mode: SIMULATION
Real API Calls: 0
Virtual Clock: Enabled
```

görmeniz beklenir.

**Öğretim açısından ayrım:** Virtual Time'da 30 saniyelik bir bekleme hesaplanmış olması, bilgisayarın gerçek dünyada 30 saniye beklediği anlamına gelmez.

---

## 3. Karşılaştırılan dört retry stratejisi

LAB07'de aşağıdaki dört strateji vardır.

| Strateji ID | Yaklaşım |
|---|---|
| `no_backoff` | Hata sonrasında beklemeden yeniden dene |
| `exponential_backoff` | Retry gecikmesini üstel artır |
| `exponential_backoff_with_jitter` | Üstel gecikmeye rastlantısallık ekle |
| `rate_limit_aware` | Geçerli `Retry-After` varsa onu uygula, yoksa backoff kullan |

### 3.1 No Backoff

Bir çağrı `429` döndürürse istemci **0 saniye bekleyerek** tekrar dener.

```text
T=0.0  Attempt 1 -> 429
T=0.0  Attempt 2 -> 429
T=0.0  Attempt 3 -> 429
...
```

Bu, gerçek sistemlerde **retry storm** riskini artırabilir.

**Kritik soru:** Servis henüz toparlanmamışsa daha hızlı tekrar denemek gerçekten yardımcı olur mu?

### 3.2 Exponential Backoff

Retry gecikmesi artarak uygulanır:

\[
d_n=\min(d_{\max},d_{\mathrm{base}}\times 2^{n-1})
\]

Burada \(n\), ilk retry için 1 olacak şekilde retry numarasıdır.

`base_delay_seconds=1`, `max_delay_seconds=30` için örnek:

| Retry | Scheduled delay | Virtual timestamp (bir önceki denemeler başarısızsa) |
|---|---:|---:|
| İlk attempt | — | 0 s |
| Retry 1 | 1 s | 1 s |
| Retry 2 | 2 s | 3 s |
| Retry 3 | 4 s | 7 s |
| Retry 4 | 8 s | 15 s |

**Dikkat:** `max_attempts=5` değeri **ilk denemeyi de** kapsar; dolayısıyla bir istemci en fazla 4 retry yapabilir.

### 3.3 Exponential Backoff with Jitter

Sabit üstel bekleme süreleri, çok sayıda istemci aynı anda başladığında yeniden senkronize olabilir.

**Full Jitter** yaklaşımında bekleme, üstel üst sınırdan rastgele seçilir:

\[
d_n \sim U(0,\min(d_{\max},d_{\mathrm{base}}\times 2^{n-1}))
\]

Bu, çok sayıda istemcinin aynı anda tekrar deneme olasılığını azaltabilir.

Ancak **jitter her senaryoda daha yüksek success rate garanti etmez**. Rastgele bir kısa gecikme, toparlanma başlamadan aynı hatayı tekrar üretmeye de neden olabilir.

LAB07, reproducibility için seeded RNG kullanır. Aynı seed ve parametrelerle aynı simülasyon davranışı üretilebilir.

### 3.4 Rate-Limit-Aware

Bu strateji, simüle edilen sunucudan dönen `Retry-After` header'ını okumaya çalışır.

```text
HTTP/1.1 429 Too Many Requests
Retry-After: 5
```

Geçerli bir header olduğunda istemci belirtilen süreyi beklemeyi planlar.

Header eksik veya geçersiz olduğunda LAB07, **Exponential Backoff** fallback'ine döner.

**Önemli:** Header'ın parse edilebilir olması, bekleme süresinin sistemin toplam `wait budget` sınırına sığacağı anlamına gelmez. Geçerli fakat çok uzun süreler ayrı bir budget kararı gerektirir.

---

## 4. Experiment Profiles ve simülasyon kapsamı

| Profil | Senaryo sayısı | Strateji sayısı | Simulation runs | Gerçek API |
|---|---:|---:|---:|---:|
| `pilot` | 1 | 4 | **4** | 0 |
| `default` | 3 | 4 | **12** | 0 |
| `extended` | 5 | 4 | **20** | 0 |

**Simulation run**, belirli bir `(scenario, strategy)` kombinasyonudur.

Bu sayılar **HTTP istek sayısı değildir**. Her simulation run içinde bir veya birden çok istemci ve birden çok sanal HTTP attempt bulunabilir.

Örneğin Default Grand Test:

```text
3 scenarios
× 4 strategies
= 12 simulation runs

Real Groq API calls = 0
```

Bu laboratuvarın ana testi **Default Grand Test**'tir.

---

## 5. Beş simulation scenario

### Scenario 1 — Fixed Transient Failures

**ID:** `transient_failures`\
**Varsayılan:** 1 istemci, ilk **2** attempt `429`, sonrakiler başarılı.

Örnek:

```text
Attempt 1 -> 429
Attempt 2 -> 429
Attempt 3 -> 200
```

Bu senaryoda bütün stratejiler aynı hata dizisini görür, ancak farklı bekleme süreleri uygular.

**Öğrenme hedefi:** Retry stratejisi, belirli sayıda geçici hata sonrasında başarıyı elde ederken toplam beklemeyi nasıl etkiler?

Bu bir **simüle edilmiş sabit başarısızlık dizisidir**; gerçek Groq davranışı olarak yorumlanmaz.

### Scenario 2 — Server Recovery Window

**ID:** `server_recovery_window`\
**Varsayılan:** Sunucu sanal zamanın **3. saniyesinden önce** bütün çağrıları reddeder.

```text
Virtual Time < 3 s  -> 429
Virtual Time >= 3 s -> 200
```

Bu senaryoda:

- `no_backoff`, denemelerini T=0'da hızla tüketebilir.
- `exponential_backoff`, T=1 ve T=3 gibi noktalarda tekrar deneyebilir.
- `jitter`, seçilen rastgele gecikmelere göre farklı sanal zamanlarda deneyecektir.
- `rate_limit_aware`, geçerli header yoksa fallback kullanacaktır.

**Öğrenme hedefi:** Beklemeden tekrar denemek neden toparlanma süresine erişmeden retry budget'ını tüketebilir?

### Scenario 3 — Shared Contention / Thundering Herd

**ID:** `shared_contention`\
**Varsayılan:** **8 istemci**, her **1 saniyelik sanal pencerede en fazla 2 kabul edilen istek**.

Çok sayıda istemci aynı anda API çağrısı yaparsa sunucunun ortak kapasitesi dolabilir.

```text
Virtual Time: T=0.0

Client 0 ---> Accepted
Client 1 ---> Accepted
Client 2 ---> 429
Client 3 ---> 429
...
```

**Öğrenme hedefi:** Birbirinden bağımsız istemcilerin aynı zamanda retry yapması kapasite çakışmalarını nasıl artırır?

Burada `retry_collision_count`, **paylaşılan kapasiteye takılan sanal 429 kayıtlarının sayısıdır**. Bütün 429 yanıtları bu alan kapsamında collision sayılmaz.

### Scenario 4 — Retry-After Variations

**ID:** `retry_after_variations`\
Yalnızca **Extended** profilde bulunur.

Sekiz sanal istemciye farklı header değerleri verilir:

| Varyant | Örnek değer |
|---|---|
| Geçerli tam sayı | `5` |
| Geçerli ondalık sayı | `2.5` |
| Header eksik | `None` |
| Hatalı metin | `not-a-number` |
| Sıfır | `0` |
| Negatif | `-5` |
| Wait budget'ı aşan | `80` |
| HTTP-date | Sanal saatle ilişkilendirilmiş tarih |

**Öğrenme hedefi:** `Retry-After` değeri geldiğinde parser, fallback ve wait budget kararları nasıl ayrılır?

LAB07'nin parser'ı:

- Pozitif sayıları ve sıfırı değerlendirebilir.
- Negatif, `NaN`, sonsuz veya parse edilemeyen değerleri reddeder.
- HTTP-date biçimini injected **virtual clock** ile değerlendirir.
- Şüpheli derecede büyük header değerlerinde fallback kullanır.
- Geçerli fakat mevcut senaryonun 10 saniyelik toplam wait budget'ına sığmayan beklemeyi **sessizce kısaltmaz**.

**Ek not:** Decimal seconds, bir provider extension'ı olarak yorumlanabilir; HTTP'nin standart `Retry-After` tanımında `delta-seconds` tam sayıdır.

### Scenario 5 — Request/Token Quota Pressure

**ID:** `quota_pressure`\
Yalnızca **Extended** profilde bulunur.

Varsayılan simülasyon parametreleri:

```text
Clients:                      3
Request cap per virtual window: 5
Token cap per virtual window:   250
Token demand per request:       100
Virtual window size:            60 s
```

Bu senaryoda RPM benzeri request cap ve TPM benzeri token cap aynı anda uygulanır.

**Önemli:** `100` token demand sayısı Groq'tan ölçülmüş gerçek token tüketimi değildir; simülasyon girdisidir.

**Öğrenme hedefi:** Request sayısı limite ulaşmadan önce token kotasının dolması mümkün müdür? Hangi limit önce devreye girer?

---

## 6. Kurulum: Docker Desktop ve PowerShell

### 6.1 Gereksinimler

- Docker Desktop
- Docker Compose
- LAB07'yi içeren `llm-token-economics-lab` repository

**LAB07 için `.env`, `GROQ_API_KEY` veya `config/live.toml` oluşturmanız gerekmez.**

### 6.2 Proje klasörüne geçin

```powershell
cd C:\Users\pc\llm-token-economics-lab
```

Farklı klasör kullanıyorsanız yolu değiştirin.

### 6.3 Docker image build

```powershell
docker compose build lab-runner
```

İlk build işlemi gerekli imaj/dependency dosyalarını indirirken internet kullanabilir. Simülasyonun kendisi, image hazır olduğunda dış API bağlantısına ihtiyaç duymaz.

### 6.4 LAB07 yardım ekranı

```powershell
docker compose run --rm lab-runner lab07 --help
```

Bu komut geçerli experiment ve CLI seçeneklerini gösterir.

---

## 7. LAB07 — Tüm Docker komutları

### A. Default Grand Test — Önerilen ana ders

```powershell
docker compose run --rm lab-runner lab07 all --profile default --config config/default.toml
```

**Beklenen deney planı:**

```text
Profile: default
Scenarios: 3
Strategies: 4
Planned Simulation Runs: 12
Real API Calls: 0
```

Bu, sınıfta dört retry stratejisini **aynı üç senaryo üzerinden** karşılaştırmak için önerilen komuttur.

### B. Pilot — 4 simulation run

```powershell
docker compose run --rm lab-runner lab07 all --profile pilot --config config/default.toml
```

Yalnızca `transient_failures` senaryosunu dört stratejiyle çalıştırır.

### C. Extended — 20 simulation run

```powershell
docker compose run --rm lab-runner lab07 all --profile extended --config config/default.toml
```

Beş senaryoyu dört stratejiyle çalıştırır. `Retry-After` varyasyonları ve RPM/TPM simülasyonu da dahil edilir.

### D. Yalnızca No Backoff

```powershell
docker compose run --rm lab-runner lab07 no_backoff --profile default --config config/default.toml
```

Default profilde üç senaryonun yalnızca `no_backoff` stratejisiyle simülasyonunu yapar.

### E. Yalnızca Exponential Backoff

```powershell
docker compose run --rm lab-runner lab07 exponential_backoff --profile default --config config/default.toml
```

### F. Yalnızca Exponential Backoff with Jitter

```powershell
docker compose run --rm lab-runner lab07 exponential_backoff_with_jitter --profile default --config config/default.toml
```

### G. Yalnızca Rate-Limit-Aware

```powershell
docker compose run --rm lab-runner lab07 rate_limit_aware --profile default --config config/default.toml
```

### H. Tek bir senaryo

```powershell
docker compose run --rm lab-runner lab07 all --scenario shared_contention --config config/default.toml
```

Bir senaryoyu dört stratejiyle çalıştırır. **4 simulation run, 0 gerçek API çağrısı**.

### I. Retry-After özel senaryosu

```powershell
docker compose run --rm lab-runner lab07 all --scenario retry_after_variations --config config/default.toml
```

Extended profili çalıştırmadan yalnızca bu gelişmiş senaryoyu test etmek için kullanılabilir.

### J. RPM / TPM özel senaryosu

```powershell
docker compose run --rm lab-runner lab07 all --scenario quota_pressure --config config/default.toml
```

### K. Sabit başarısızlık sayısını değiştirme — Legacy override

```powershell
docker compose run --rm lab-runner lab07 all --fail-count 3 --config config/default.toml
```

**Dikkat:** `--fail-count` parametresi, profil ne olursa olsun yalnızca `transient_failures` senaryosunu çalıştırır. Bu, CLI'nin bilinçli legacy davranışıdır.

### L. Retry-After header enjekte etme — Legacy override

```powershell
docker compose run --rm lab-runner lab07 all --fail-count 2 --retry-after-seconds 5 --config config/default.toml
```

Bu da yalnızca sabit transient failure senaryosunu çalıştırır. Simüle edilen başarısız yanıtlara `Retry-After: 5` ekler.

### M. Seed değiştirme

```powershell
docker compose run --rm lab-runner lab07 all --profile default --seed 123 --config config/default.toml
```

Jitter stratejisinin rastgele bekleme seçimleri farklılaşabilir. Aynı parametreler ve seed ile simülasyon yeniden üretilebilir.

### N. Deneye özel Run ID verme

```powershell
docker compose run --rm lab-runner lab07 all --profile default --experiment-run-id class-lab07-group-a --config config/default.toml
```

Sonuçlar:

```text
results/lab07_rate_limiting/class-lab07-group-a/
```

**Bir run ID için kayıtlı ölçümler varsa aynı ID tekrar kullanılamaz.** Yeni deneyde yeni isim verin.

### O. Eski `--live` parametresini neden kullanmıyoruz?

Teknik olarak aşağıdaki komut kabul edilir:

```powershell
docker compose run --rm lab-runner lab07 all --profile default --live --config config/default.toml
```

Ancak `--live` açıkça **ignored** olarak raporlanır; yine gerçek API çağrısı yapılmaz.

**Sınıf komutlarında `--live` kullanmayın.** Offline simülasyon ile gerçek API deneylerini kavramsal olarak ayıralım.

### P. Dry-Run var mı?

LAB07'nin normal deneyleri zaten **offline simülasyondur**. CLI'de `--dry-run` adlı bir seçenek bulunmaz.

Plan ve simülasyon çıktısını görmek için doğrudan `lab07 all` kullanın. `--dry-run` eklemeyin.

---

## 8. LAB07 konfigürasyonunu anlama

Proje `config/default.toml` içerisindeki `[lab07]` ayarlarını kullanır.

| Parametre | Varsayılan | Anlamı |
|---|---:|---|
| `enabled` | `true` | LAB07 çalışsın mı? |
| `default_profile` | `default` | Seçilen simülasyon kapsamı |
| `randomization_seed` | `42` | Jitter RNG seed'i |
| `max_simulation_events` | `10000` | Simülasyon olay üst sınırı |
| `max_clients` | `50` | Senaryo istemci üst sınırı |
| `max_attempts` | `5` | İlk attempt dahil toplam deneme üst sınırı |
| `base_delay_seconds` | `1.0` | Backoff temel gecikmesi |
| `max_delay_seconds` | `30.0` | Backoff üst sınırı |
| `max_total_wait_seconds` | `120.0` | Bir istemcinin toplam planlanmış bekleme bütçesi |
| `transient_fail_count` | `2` | Senaryo 1'in ilk başarısız deneme sayısı |
| `recovery_time_seconds` | `3.0` | Senaryo 2 toparlanma zamanı |
| `contention_client_count` | `8` | Senaryo 3 istemci sayısı |
| `contention_capacity_per_window` | `2` | Senaryo 3 request kapasitesi |
| `contention_window_seconds` | `1.0` | Senaryo 3 sanal pencere süresi |
| `quota_client_count` | `3` | Senaryo 5 istemci sayısı |
| `quota_rpm` | `5` | Simüle request cap |
| `quota_tpm` | `250` | Simüle token cap |
| `quota_token_demand_per_request` | `100` | Her request'in sanal token ihtiyacı |
| `quota_window_seconds` | `60.0` | Kota penceresi |

**Önemli:** Bu ayarlar, gerçek Groq hesabınızın limitlerini temsil etmez.

`[lab07]` retry ayarları, diğer laboratuvarların gerçek operasyonel API retry yapılandırmasından **ayrıdır**. LAB07'deki deney değerlerini değiştirmek başka laboratuvarların retry politikasını değiştirmek için kullanılmamalıdır.

---

## 9. Ölçüm alanlarını anlamak

LAB07'nin bütün değerleri **simülasyon ölçümüdür**.

### 9.1 Attempt düzeyindeki veriler

`attempt_timeline.csv` içinde her sanal attempt için örnek alanlar:

| Alan | Anlamı |
|---|---|
| `scenario_id` | Simülasyon senaryosu |
| `strategy_id` | Kullanılan retry stratejisi |
| `simulation_run_id` | Senaryo-strateji simülasyon kimliği |
| `client_id` | Sanal istemci |
| `attempt_number` | İlk deneme dahil deneme sırası |
| `retry_number` | Kaçıncı retry olduğu (`attempt_number - 1`) |
| `virtual_timestamp_seconds` | Sanal zaman |
| `http_status` | Simüle edilen `200` veya `429` |
| `retry_after_header` | Simüle sunucunun ham header değeri |
| `scheduled_delay_seconds` | Planlanan sanal bekleme |
| `delay_reason` | Bekleme kararının gerekçesi |
| `final_outcome` | Son attempt'te istemcinin durumu |
| `collision` | Paylaşılan kapasite çakışması mı? |

### 9.2 Client düzeyindeki veriler

`raw_measurements.json/.csv` içindeki temel alanlar:

- `succeeded`
- `outcome`
- `total_attempts`
- `retry_count`
- `rate_limit_response_count`
- `total_scheduled_wait_seconds`
- `virtual_completion_time_seconds`

Bu alanlar **bir istemcinin** sonuçlarını açıklar.

### 9.3 Strategy Summary

Bir `(scenario, strategy)` kombinasyonu için:

| Metrik | Anlamı |
|---|---|
| `total_clients` | Simüle istemci sayısı |
| `successful_clients` | Başarıyla tamamlayanlar |
| `failed_clients` | Başarısız tamamlayanlar |
| `success_rate` | `successful_clients / total_clients` |
| `total_attempts` | Bütün istemcilerin toplam sanal attempt sayısı |
| `total_retry_count` | Toplam sanal retry |
| `total_429_responses` | Simüle sunucudan alınan bütün 429 yanıtları |
| `mean_attempts_per_client` | İstemci başına ortalama attempt |
| `total_scheduled_wait_seconds` | Bütün istemcilerin planlanan beklemelerinin toplamı |
| `mean_wait_seconds` | İstemci başına ortalama sanal bekleme |
| `max_wait_seconds` | En yüksek istemci beklemesi |
| `mean_recovery_time_seconds` | Başarılı istemcilerin ortalama sanal tamamlanma zamanı |
| `completion_time_seconds` | Başarılı istemciler arasındaki en geç tamamlanma zamanı |
| `retry_collision_count` | Shared-capacity kaynaklı çakışma sayısı |

**Kritik ayrım:** Birden fazla istemcinin beklemeleri toplanınca çıkan `total_scheduled_wait_seconds`, bir insanın veya tek işlemin beklediği gerçek süre değildir.

---

## 10. Sonuçları nasıl yorumlarız?

### 10.1 Success Rate

\[
\mathrm{SuccessRate}=
\frac{\mathrm{SuccessfulClients}}{\mathrm{TotalClients}}
\]

Örneğin, 8 istemciden 6'sı sanal senaryoda başarıyla tamamladıysa:

\[
\frac{6}{8}=75\%
\]

### 10.2 Retry Count vs Attempts

```text
Initial request = Attempt 1
Retry 1         = Attempt 2
Retry 2         = Attempt 3
```

Bu örnekte:

```text
Total Attempts = 3
Retry Count    = 2
```

`max_attempts=5`, **5 retry** anlamına gelmez.

### 10.3 Virtual Completion Time

`completion_time_seconds`, başarılı istemciler arasındaki son başarı zamanıdır.

Tüm istemciler başarısızsa bu alan `None` olabilir. Bu durumda `0` yazıp “anında tamamlandı” gibi yanlış bir yorum yapmayın.

### 10.4 Retry Collisions

`retry_collision_count`, sunucunun ortak capacity mekanizması nedeniyle reddettiği sanal attempt'leri ifade eder.

Her `429` collision değildir:

- Fixed Transient Failures: `429`, fakat shared-capacity collision değil.
- Recovery Window: `429`, fakat collision değil.
- Shared Contention: `429`, capacity collision olabilir.

### 10.5 En az bekleme = en iyi strateji mi?

**Hayır.**

Örnek temsili sonuç:

| Strateji | Success Rate | Virtual Wait |
|---|---:|---:|
| No Backoff | %0 | 0 s |
| Exponential Backoff | %100 | 3 s |

Sadece “en az bekleme”yi seçmek doğru sonuç vermez. İkinci strateji daha uzun süre beklemiş olsa da işe yaramış olabilir.

LAB07, **Reliability** ile **Efficiency** metriklerini ayrı raporlar.

---

## 11. Terminal çıktısını okuma

LAB07 simülasyonunda şu yapıda sonuçlar görürsünüz:

```text
LAB07 -- Rate Limiting & Retry Strategies

Execution Mode: SIMULATION
Real API Calls: 0

Profile: default
Scenarios: 3 (...)
Strategies: 4 (...)

Planned Simulation Runs: 12
Virtual Clock: Enabled
Random Seed: 42
```

Bir strateji için:

```text
Scenario: server_recovery_window
Strategy: exponential_backoff

Attempts:                 ...
Retries:                  ...
HTTP 429 Responses:       ...
Succeeded:                .../...
Success Rate:             ...

Total Scheduled Wait:     ... seconds
Virtual Completion Time:  ... seconds
Retry Collisions:         ...

Retry Timeline (client 0):
  T=0.00s  Attempt 1 -> 429
  T=1.00s  Attempt 2 -> 429
  T=3.00s  Attempt 3 -> 200
```

Yukarıdaki timeline, ilgili varsayılan senaryonun deterministik örneğidir. Konfigürasyon değişirse sonuçlar değişebilir.

Final değerlendirme şöyle bir ayrım yapar:

```text
Reliability Comparison: highest success rate = ...
Retry Traffic / Wait Comparison: lowest total wait = ...
```

**Bunlar ayrı sıralamalardır.** Sonuçta en düşük wait değerine sahip strateji, en yüksek başarı oranına sahip olmak zorunda değildir.

---

## 12. Sonuç dosyaları

Her çalıştırma ayrı bir run klasörüne yazılır:

```text
results/
└── lab07_rate_limiting/
    └── <experiment_run_id>/
        ├── experiment_manifest.json
        ├── experiment_plan.csv
        ├── attempt_timeline.csv
        ├── raw_measurements.json
        ├── raw_measurements.csv
        ├── strategy_summary.csv
        ├── scenario_comparison.csv
        ├── experiment_summary.json
        ├── analysis_report.md
        └── charts/
```

| Dosya | Kullanım |
|---|---|
| `experiment_manifest.json` | Konfigürasyon ve deney metadata |
| `experiment_plan.csv` | Planlanan `(scenario, strategy)` kombinasyonları |
| `attempt_timeline.csv` | Her sanal HTTP attempt'in zaman çizelgesi |
| `raw_measurements.json/.csv` | İstemci bazında simülasyon sonuçları |
| `strategy_summary.csv` | Strateji bazında başarı, bekleme, 429 ve collision metrikleri |
| `scenario_comparison.csv` | Aynı senaryoda dört stratejinin karşılaştırılması |
| `experiment_summary.json` | Tüm simülasyon özeti; `real_api_calls=0` |
| `analysis_report.md` | Bulgular ve deneysel sınırlamalar |
| `charts/` | Simülasyondan türetilen grafikler |

**Önerilen inceleme sırası:**

1. `analysis_report.md`
2. `scenario_comparison.csv`
3. `strategy_summary.csv`
4. `attempt_timeline.csv`
5. `charts/`

---

## 13. Grafikler

LAB07 şu dokuz grafik dosyasını üretmeyi hedefler:

| Dosya | Ne öğretir? |
|---|---|
| `retry_timeline.png` | Sanal denemeler zaman boyunca ne zaman gerçekleşiyor? |
| `exponential_backoff_schedule.png` | Retry delay'leri nasıl büyüyor? |
| `backoff_vs_jitter_distribution.png` | Jitter zamanlamayı nasıl dağıtıyor? |
| `success_rate_by_strategy.png` | Hangi strateji kaç istemciyi başarıya ulaştırdı? |
| `total_429_responses.png` | Stratejiler kaç sanal 429 üretti? |
| `virtual_recovery_time.png` | Başarılı istemciler ne zaman tamamlandı? |
| `retry_collision_analysis.png` | Shared contention kaynaklı çakışma sayıları |
| `retry_after_compliance.png` | `Retry-After` kararlarının analizi |
| `resilience_tradeoff_dashboard.png` | Reliability ve efficiency birlikte nasıl görünüyor? |

**Önemli:** Grafiklerdeki zaman ekseni **Virtual Time** değerlerini gösterir. Bunlar gerçek Groq latency ölçümleri değildir.

Tek senaryolu çalıştırmalarda bazı grafikler veri yokluğu nedeniyle sınırlı olabilir.

---

## 14. Backfill — API çağrısı olmadan grafik üretme

LAB07 zaten offline çalışır. Backfill, daha önce kaydedilen sonuçlardan grafik dosyalarını yeniden üretir.

Bütün mevcut run'lar:

```powershell
docker compose run --rm lab-runner lab07 backfill --config config/default.toml
```

Belirli bir run:

```powershell
docker compose run --rm lab-runner lab07 backfill --run-id YOUR_RUN_ID --config config/default.toml
```

`YOUR_RUN_ID` yerine sonuç klasörünün adını yazın.

Backfill:

- Gerçek API çağrısı yapmaz.
- API key gerektirmez.
- Mevcut ölçüm sonuçlarını okur.
- JSON/CSV/Markdown ölçümlerini değiştirmez.
- Uygun PNG grafiklerini yeniden üretir.

---

## 15. Uygulama görevleri

### Görev 1 — Pilot: Dört stratejinin zamanlaması

Çalıştırın:

```powershell
docker compose run --rm lab-runner lab07 all --profile pilot --config config/default.toml
```

`transient_failures` senaryosu için tabloyu doldurun.

| Strateji | Attempts | Retries | HTTP 429 | Virtual Completion (s) | Success |
|---|---:|---:|---:|---:|---|
| No Backoff | | | | | |
| Exponential | | | | | |
| Jitter | | | | | |
| Rate-Limit-Aware | | | | | |

**Soru:** Başarı oranları aynı olsa da neden tamamlanma zamanları değişebilir?

### Görev 2 — Recovery Window: No Backoff neden başarısız olabilir?

```powershell
docker compose run --rm lab-runner lab07 all --scenario server_recovery_window --config config/default.toml
```

İlgili sonuçlarda her stratejinin **attempt time** değerlerini karşılaştırın.

Sorular:

1. Sunucu kaçıncı sanal saniyede toparlanıyor?
2. No Backoff bu zamana ulaşabiliyor mu?
3. Exponential Backoff kaçıncı attempt'te başarılı olabiliyor?
4. Jitter ile aynı sonuç garanti mi?

### Görev 3 — Thundering Herd

```powershell
docker compose run --rm lab-runner lab07 all --scenario shared_contention --config config/default.toml
```

Sonuçları kaydedin:

| Strateji | Successful Clients | Total 429 | Retry Collisions | Virtual Completion |
|---|---:|---:|---:|---:|
| No Backoff | | | | |
| Exponential | | | | |
| Jitter | | | | |
| Rate-Limit-Aware | | | | |

**Soru:** Çok sayıda istemcinin aynı anda tekrar denemesi neden ek başarısızlık oluşturuyor?

### Görev 4 — Retry-After davranışı

```powershell
docker compose run --rm lab-runner lab07 all --scenario retry_after_variations --config config/default.toml
```

`attempt_timeline.csv` dosyasında `rate_limit_aware` stratejisini inceleyin.

Karşılaştırın:

- Geçerli tam sayı
- Geçerli ondalık değer
- Header eksik
- Hatalı header
- Negatif değer
- Wait budget'ı aşan değer
- HTTP-date

**Soru:** Header parse etmek ile belirtilen beklemeyi bütçe içinde uygulayabilmek neden farklı konulardır?

### Görev 5 — RPM ve TPM baskısı

```powershell
docker compose run --rm lab-runner lab07 all --scenario quota_pressure --config config/default.toml
```

Öğrenciler şu sorulara cevap vermeli:

1. RPM mi, TPM mi önce tükeniyor?
2. Bir retry, yeni sanal kota penceresine denk gelirse ne değişiyor?
3. Daha fazla retry başarılı çağrı sayısını artırıyor mu?
4. RPM ve TPM'yi aynı sınırlama olarak ele almak neden yanlış?

### Görev 6 — Seed ile Jitter

İlk çalıştırma:

```powershell
docker compose run --rm lab-runner lab07 all --scenario shared_contention --seed 42 --config config/default.toml
```

İkinci çalıştırma:

```powershell
docker compose run --rm lab-runner lab07 all --scenario shared_contention --seed 99 --config config/default.toml
```

İki run'ın `attempt_timeline.csv` dosyalarını karşılaştırın.

**Sorular:**

- Jitter attempt zamanları değişiyor mu?
- Success rate, collisions ve virtual completion değişti mi?
- Aynı seed ile tekrar çalıştırılırsa sonuçlar tekrarlanabilir mi?

---

## 16. Tartışma soruları

1. Rate Limiting ile Retry neden aynı kavram değildir?
2. HTTP 429 yanıtı alındığında hemen yeniden denemek neden riskli olabilir?
3. Exponential Backoff nasıl çalışır?
4. Jitter neden özellikle eşzamanlı istemcilerde önemlidir?
5. `Retry-After` header'ı eksik olduğunda ne yapılmalıdır?
6. Hatalı bir Retry-After değerine neden körü körüne uyulmaz?
7. `max_attempts=5`, neden beş retry anlamına gelmez?
8. Retry sayısını artırmak her zaman success rate'i yükseltir mi?
9. Birden fazla istemcinin toplam bekleme süresi ile workflow tamamlanma süresi neden farklıdır?
10. Thundering Herd problemi nasıl ortaya çıkar?
11. RPM ve TPM limitlerinin aynı anda uygulanması ne sonuç doğurabilir?
12. En kısa bekleme süresine sahip strateji neden en güvenilir strateji olmayabilir?
13. Retry storm ile normal kullanıcı trafiğinin birleşmesi kapasiteyi nasıl etkileyebilir?
14. Retry politikasının idempotency ve yan etki içeren istekler açısından ayrıca değerlendirilmesi neden gerekir?

---

## 17. Troubleshooting

### 17.1 `--live` neden gerçek çağrı başlatmıyor?

Çünkü LAB07 simülasyon-only tasarlanmıştır. `--live` yalnızca eski CLI ile uyumluluk için kabul edilir ve yok sayılır.

**LAB07 için Groq API key gerekmez.**

### 17.2 `--dry-run` hata veriyor

LAB07 CLI'de `--dry-run` seçeneği bulunmaz. Simülasyon zaten offline olduğundan şu komutu çalıştırın:

```powershell
docker compose run --rm lab-runner lab07 all --profile pilot --config config/default.toml
```

### 17.3 `config/live.toml` yok

LAB07 için gerekli değildir. `config/default.toml` kullanın.

### 17.4 `experiment_run_id` zaten kullanılmış

Dolu bir run ID tekrar kullanılmaz. Yeni `--experiment-run-id` seçin.

### 17.5 `max_clients` veya simulation event limiti

LAB07, çok büyük simülasyonların kaynakları tüketmesini önlemek için `lab07.max_clients` ve `lab07.max_simulation_events` sınırlarını uygular.

Bu sınırlar gerçek API kota limitleri değildir.

### 17.6 Çıktıda `FAILED_MAX_ATTEMPTS`

İstemcinin sanal attempt bütçesi tükenmiştir. Bunun nedeni sunucunun toparlanma zamanına ulaşamaması veya kapasite reddi olabilir.

### 17.7 Çıktıda `FAILED_RETRY_BUDGET`

Bekleme kararı toplam izin verilen sanal wait budget'ı aşmıştır.

### 17.8 Çıktıda `FAILED_DEADLINE`

Sanal deadline sınırına ulaşılmıştır. Default konfigürasyonda deadline tanımlı değildir; ancak farklı ayarlarda görülebilir.

### 17.9 Grafik oluşturulamadı

İlk olarak `experiment_summary.json` dosyasının oluştuğunu kontrol edin. Ardından `backfill` kullanın.

### 17.10 Gerçek terminal süresi simülasyonun virtual time'ından farklı

Bu beklenen davranıştır.

```text
Virtual Completion Time: 120 s
```

ifadesi, Docker komutunun 120 gerçek saniye sürdüğünü söylemez. Simülasyon saatinin hesaplanan değeridir.

---

## 18. Deneysel sınırlamalar

LAB07, gerçek Groq altyapısını değil **kontrollü, sadeleştirilmiş bir sanal sunucuyu** kullanır.

Aşağıdakiler bu laboratuvarın ölçüm kapsamı dışındadır:

- Gerçek sağlayıcıdan alınan HTTP 429 sıklığı
- Gerçek `Retry-After` header dağılımı
- Gerçek API maliyeti
- Gerçek token usage
- Gerçek server-side queueing
- Network latency
- Gerçek model inference time
- Production ölçekte dağıtık koordinasyon
- Gerçek istek idempotency garantileri

Simülatör `200` ve `429` davranışlarını çalışır. Diğer HTTP hata sınıflarına ilişkin production politikaları bu laboratuvarda doğrudan ölçülmez.

Dolayısıyla LAB07 sonuçları **retry stratejilerinin temel mekanizmalarını açıklar**, fakat Groq veya başka bir provider'da hangi politikanın her zaman üstün olacağını kanıtlamaz.

---

## 19. LAB07'nin diğer laboratuvarlarla ilişkisi

- **LAB01:** Token türleri ve model inference maliyetinin ölçülmesi.
- **LAB02:** Context growth'un toplam token tüketimine etkisi.
- **LAB03:** Prompt caching ile input maliyetinin ilişkisi.
- **LAB04:** Output budget ve gereksiz çağrıların azaltılması.
- **LAB05:** Göreve uygun model seçimi.
- **LAB06:** Streaming ve algılanan bekleme.
- **LAB07:** **Retry ve rate limiting nedeniyle iş yükünün nasıl değiştiği.**

Production uygulamalarda API maliyeti yalnızca tek başarılı çağrının tokenlarından oluşmayabilir; başarısız ve tekrar denenen çağrılar, iş yükünü, latency'yi ve kullanım limitlerini etkileyebilir.

LAB07, bu etkiyi **gerçek API faturası oluşturmadan** öğretir.

---

## 20. Ana çıkarım

**En agresif retry stratejisi en iyi strateji olmayabilir.**

Bir retry politikasını değerlendirirken yalnızca “en hızlı retry yapan” stratejiyi değil, şu faktörleri birlikte düşünün:

- Başarı oranı
- Başarısız deneme sayısı
- 429 yanıtlarının sayısı
- Thundering Herd etkisi
- Retry collision sayısı
- Virtual recovery/completion zamanı
- Toplam wait budget
- `Retry-After` uyumluluğu

> **Güvenilir API tasarımının amacı, başarısız istekleri en kısa aralıkla tekrarlamak değil; gerektiğinde, sınırlı ve kontrollü biçimde tekrar deneyerek hem istemciyi hem sunucuyu korumaktır.**

---

## 21. Kaynak kodlar

| Repository dosyası | İncelenecek konu |
|---|---|
| `src/labs/lab07_rate_limiting/cli.py` | CLI ve simulation orchestration |
| `src/labs/lab07_rate_limiting/matrix.py` | Pilot, Default ve Extended planlar |
| `src/labs/lab07_rate_limiting/scenarios.py` | Beş senaryo ve test parametreleri |
| `src/labs/lab07_rate_limiting/server_simulator.py` | Sanal sunucu ve 429 üretimi |
| `src/labs/lab07_rate_limiting/runner.py` | Virtual clock ve retry kararları |
| `src/labs/lab07_rate_limiting/retry_after.py` | Retry-After parser |
| `src/labs/lab07_rate_limiting/models.py` | Attempt/Client veri modelleri |
| `src/labs/lab07_rate_limiting/stats.py` | Reliability ve efficiency metrikleri |
| `src/labs/lab07_rate_limiting/persistence.py` | Sonuç dosyaları |
| `src/labs/lab07_rate_limiting/charts.py` | Dokuz otomatik grafik |
| `src/labs/lab07_rate_limiting/report.py` | Analysis report |
| `config/default.toml` | LAB07 parametreleri |

---

## 22. Repository'de önerilen konum

Dosyayı şuraya ekleyin:

```text
docs/participant/lab07-rate-limiting-and-retry-strategies.md
```

Ana `docs/participant/README.md` dosyasına eklenebilecek bağlantı:

```markdown
## LAB07 — Rate Limiting & Retry Strategies

[Open LAB07 Participant Guide](lab07-rate-limiting-and-retry-strategies.md)
```
