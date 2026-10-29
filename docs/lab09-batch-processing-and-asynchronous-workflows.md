# LAB09 — Batch Processing & Asynchronous Workflows

> **Katılımcı rehberi** · Python / Groq API / Docker Compose\
> **Kapsam:** Aynı sabit iş yükü üzerinde üç execution stratejisi: **Sequential**, **Async Concurrent**, **Groq Batch API**.\
> **Amaç:** Hangi stratejinin ne zaman daha hızlı, ne zaman daha ucuz, ve her ikisinde de **doğruluğun** korunup korunmadığını ölçmek.

## 1. Bu laboratuvarda ne öğreneceksiniz?

Aynı görevleri Groq API'ye üç farklı şekilde gönderebilirsiniz:

1. **Sequential** — istekler birer birer, öncekinin cevabı gelmeden bir sonraki gönderilmez.
2. **Async Concurrent** — istekler aynı anda, sınırlı bir eşzamanlılık (`max_concurrent_requests`) altında gönderilir.
3. **Groq Batch API** — tüm istekler bir JSONL dosyası olarak sağlayıcıya teslim edilir; sağlayıcı bunları kendi zamanlamasında (dakikalar - 7 gün) işler ve karşılığında **%50 indirimli** fiyatlandırma sunar.

LAB09 bu üçünü **aynı golden görev seti** üzerinde, aynı model/`reasoning_effort`/`max_completion_tokens` ile karşılaştırır. Değişen tek şey **execution mekanizmasıdır**.

### 1.1 Test edilen hipotezler

| | Hipotez |
|---|---|
| **H1** | Concurrent execution, sequential'a göre toplam wall-clock süresini düşürebilir. |
| **H2** | Daha yüksek concurrency, **tek bir isteğin** gecikmesini otomatik olarak düşürmez. |
| **H3** | Batch API, aynı iş yükü için daha düşük fiyat sunabilir (sağlayıcının belgelediği politikaya göre). |
| **H4** | Batch processing, optimizasyon hedefini "düşük gerçek-zamanlı gecikme"den "maliyet-etkin, gecikmeli execution"a kaydırır. |
| **H5** | Daha hızlı/ucuz bir strateji, **aynı golden cevaplarla** doğruluğu korumuyorsa otomatik olarak "daha iyi" sayılmaz. |

Bu beş hipotez `analysis_report.md`'nin "Technical Conclusions" bölümünde doğrudan değerlendirilir.

## 2. Golden dataset ve profiller

`data/lab09/golden_batch_dataset_v1.json`, 16 sentetik, gerçekçi iş görevi içerir (sınıflandırma, bilgi çıkarma, basit işletme aritmetiği, boolean guard, formatlama). Her görev:

- bağımsız olarak çalıştırılabilir,
- deterministik bir `expected_answer`'a sahiptir,
- `exact_match` (büyük/küçük harf duyarsız, tam eşleşme) veya `numeric_match` (sayısal tolerans) ile **tam değer** karşılaştırmasıyla değerlendirilir — **asla substring araması ile değil**.

| Profile | Görev sayısı | Real-time çağrı (sequential + concurrent) | Batch input request |
|---|---:|---:|---:|
| `pilot` | 4 | 8 | 4 |
| `default` | 8 | 16 | 8 |
| `extended` | 16 | 32 | 16 |

> **Önemli:** 8 input request içeren bir Batch job, 8 ek real-time API çağrısı **değildir**. Bu iki sayı her raporda ayrı ayrı gösterilir, birbirine eklenmez.

`pilot` ⊂ `default` ⊂ `extended` — profiller her zaman aynı sıralı görev listesinin bir ön ekidir, böylece profiller arası karşılaştırma elma-elma kalır.

## 3. Docker kurulumu

Depo kökünde (ve `config/live.toml` zaten oluşturulmuş; bkz. ana `README.md`'nin **Setup** bölümü):

```powershell
docker compose build
docker compose run --rm lab-runner lab09 --help
```

## 4. Deney A — Sequential + Concurrent (`lab09 all`)

### Dry-run (hiçbir API çağrısı yapılmaz)

```powershell
docker compose run --rm lab-runner lab09 all --profile default --dry-run --config config/default.toml
```

Bu komut:
- `experiment_plan.csv` ve `golden_workload_snapshot.json`'ı yazar,
- planlanan real-time çağrı sayısını ve Batch input request sayısını ayrı ayrı yazdırır,
- **Conservative Preflight Cost Ceiling** (ölçülmüş değil, üst sınır) gösterir,
- gerçek bir Groq client **asla** oluşturulmaz.

### Canlı çalıştırma

```powershell
docker compose run --rm lab-runner lab09 all --profile default --live --experiment-run-id my-run --config config/live.toml
```

Bu komut sırasıyla:
1. **Sequential** koşulunu çalıştırır (8 görev, default profilde),
2. **Concurrent** koşulunu çalıştırır (aynı 8 görev, `max_concurrent_requests` altında),
3. Batch input JSONL dosyasını **yazar ama göndermez** (`batch_job.json` → `NOT_SUBMITTED`),
4. `condition_summary.csv`, `quality_comparison.csv`, `experiment_summary.json`, `analysis_report.md` ve 8 grafik üretir.

> **Batch job `all` içinde otomatik gönderilmez.** Bu bilinçli bir tasarım: Batch submission, terminal komutu bittikten çok sonra (dakikalar - 7 gün) tamamlanabilecek bağımsız bir işlemdir; asla sessizce başlatılmaz.

Terminal çıktısında her koşul için ayrı özet görürsünüz (başarılı istek sayısı, wall-clock süresi, median gecikme, doğru cevap sayısı, tahmini maliyet).

## 5. Deney B — Groq Batch API (adım adım)

Batch job'ı göndermek **her zaman** ayrı, açık bir komuttur:

```powershell
docker compose run --rm lab-runner lab09 submit_batch --experiment-run-id my-run --live --config config/live.toml
```

Bu, `my-run` koşusunun `batch_input.jsonl` dosyasını yükler, bir Batch job oluşturur, ve `batch_job.json`'a `input_file_id`/`batch_id`/`SUBMITTED` durumunu kalıcı olarak yazar. **Zaten gönderilmiş bir run için tekrar çalıştırılırsa reddedilir** — aynı işi iki kez göndermez.

> Yükleme başarılı olup job oluşturma başarısız olursa, durum `SUBMISSION_UNCERTAIN` olarak kaydedilir. Bu durumda Groq panelinizden gerçek durumu kontrol etmeden otomatik olarak yeniden göndermez.

Durumu kontrol edin (bloklamadan, tek seferlik):

```powershell
docker compose run --rm lab-runner lab09 check_batch_status --experiment-run-id my-run --live --config config/live.toml
```

Groq'un belgelediği durumlar: `validating`, `in_progress`, `finalizing`, `completed`, `failed`, `expired`, `cancelling`, `cancelled`. Bu komut sürekli polling yapmaz — siz ne zaman isterseniz (dakikalar veya saatler sonra) tekrar çalıştırırsınız.

Sonuçları alın (`output_file_id` hazır olduğunda):

```powershell
docker compose run --rm lab-runner lab09 fetch_batch_results --experiment-run-id my-run --live --config config/live.toml
```

Bu komut:
- `batch_output.jsonl`/`batch_errors.jsonl` dosyalarını indirir,
- her satırı **`custom_id`'ye göre** (asla satır sırasına göre değil) planlanan görevlerle eşleştirir,
- eksik, başarısız, yinelenen ve beklenmeyen `custom_id`'leri raporlar (asla sessizce atmaz),
- deterministik değerlendirmeyi çalıştırır,
- standard vs batch fiyat karşılaştırmasını hesaplar,
- `experiment_summary.json`, `analysis_report.md` ve grafikleri günceller.

### 5.1 Üç farklı ID'yi karıştırmayın

| ID | Ne zaman oluşur | Örnek |
|---|---|---|
| `input_file_id` | `submit_batch`'in Groq'a yüklediği JSONL dosyasının ID'si | `file-abc123` |
| `batch_id` | Batch job'ın kendi ID'si | `batch_xyz789` |
| `output_file_id` | Job tamamlandığında sonuç dosyasının ID'si | `file-def456` |

Bu üçü de `batch_job.json`'da kalıcı olarak saklanır; participant bunları elle takip etmek zorunda değildir.

## 6. Fiyatlandırma: Standard vs Batch

[console.groq.com/docs/batch](https://console.groq.com/docs/batch) (doğrulandı: 2026-10-29):

> "Pricing is at a 50% cost discount compared to synchronous API pricing."\
> "The batch discount does not stack with prompt caching discounts."

Bu nedenle `estimated_batch_cost_usd`, **tam** `prompt_tokens` sayısı üzerinden (cache indirimi hiç uygulanmadan) standard input/output fiyatının yarısı olarak hesaplanır — gerçek zamanlı modda aynı istek cache'e düşmüş olsaydı bile. `cost_comparison.csv` şu alanları içerir:

```
estimated_standard_cost_usd, estimated_batch_cost_usd,
estimated_batch_discount_usd, estimated_savings_percent,
pricing_source, pricing_verified_date, pricing_policy_version
```

Eksik `usage` verisi **asla** 0 olarak değil, `None`/`N/A` olarak raporlanır.

## 7. `evaluate` ve `backfill` — tamamen offline

```powershell
docker compose run --rm lab-runner lab09 evaluate --experiment-run-id my-run --config config/default.toml
```

Bu komut **hiçbir API çağrısı yapmaz**: `my-run` dizinindeki mevcut ölçümlerden değerlendirmeyi ve karşılaştırmaları yeniden hesaplar. Batch sonuçları henüz içeri alınmadıysa `BATCH_PENDING` olarak raporlar — asla sahte sayı üretmez. `--experiment-run-id` olmadan veya var olmayan bir run için çağrılırsa açıkça hata mesajı verir, sessizce boş bir şey yapmaz.

```powershell
docker compose run --rm lab-runner lab09 backfill --run-id my-run --config config/default.toml
```

Mevcut `experiment_summary.json`'dan PNG grafiklerini yeniden üretir; hiçbir ham ölçümü değiştirmez.

## 8. Tamamen offline eğitim modu (API anahtarı gerekmez)

Bazı Groq hesapları Batch API'ye erişemez (plan kısıtlaması), ya da bir sınıf oturumunda gerçek bir job'ın tamamlanmasını (dakikalar - 7 gün) beklemek pratik değildir. `tests/fixtures/lab09/` altında, her biri **"SYNTHETIC FIXTURE -- NOT LIVE GROQ TELEMETRY"** olarak etiketlenmiş sentetik dosyalar bulunur:

- `sample_batch_output.jsonl` — 4 görevin tamamı başarılı.
- `partial_batch_output.jsonl` — sadece 2 görev döner (diğer 2 `MISSING` olarak raporlanır).
- `malformed_batch_output.jsonl` — yinelenen `custom_id`, bilinmeyen `custom_id`, ve bozuk bir JSON satırı.
- `sample_batch_job.json` — tamamlanmış örnek bir job metadata'sı.

`tests/labs/test_lab09_offline_fixtures.py`, bu dosyalar üzerinden reconciliation → evaluation → batch pricing → rapor/grafik üretimi zincirinin **tamamını**, `GROQ_API_KEY` olmadan çalıştırır. Bu, sınıfta "Batch job'ın tamamlanmasını beklemeden" lifecycle'ın tüm adımlarını göstermenin önerilen yoludur.

## 9. Sonuç dosyaları

```
results/lab09_batch_processing/<experiment_run_id>/
    experiment_manifest.json       golden_workload_snapshot.json
    experiment_plan.csv            batch_input.jsonl
    batch_job.json                 batch_output.jsonl / batch_errors.jsonl
    sequential_measurements.(json|csv)   concurrent_measurements.(json|csv)
    batch_measurements.(json|csv)        request_timeline.csv
    condition_summary.csv          quality_comparison.csv
    cost_comparison.csv            experiment_summary.json
    analysis_report.md             charts/*.png (8 grafik)
```

Her `--experiment-run-id` kendi izole dizinine yazar; aynı run-id'yi yeniden kullanmak (ölçüm zaten varsa) reddedilir.

## 10. Eski CLI ile davranış farkları (önemli)

Bu laboratuvarın önceki (2.0 öncesi) sürümünde `prepare_batch_input` ayrı bir komuttu ve iş yükü sabit 8 aritmetik prompttu. 2.0'da:

- `prepare_batch_input` ayrı bir komut olarak **kaldırıldı** — `batch_input.jsonl`, `lab09 all` çalıştığında otomatik olarak (ama gönderilmeden) yazılır.
- İş yükü artık `data/lab09/golden_batch_dataset_v1.json`'daki 16 gerçekçi iş görevi — eski 8 aritmetik prompt **kaldırıldı** (git geçmişinde hâlâ görülebilir).
- `--batch-id`/`--output-file-id` eski CLI argümanları **hâlâ çalışır** ama "detached" modda: hiçbir şey diske kalıcı olarak yazılmaz. Önerilen yol her zaman `--experiment-run-id`'dir.

## 11. Alıştırmalar

1. `lab09 all --profile pilot --live` çalıştırın. Sequential ve concurrent wall-clock sürelerini karşılaştırın — H1'i destekliyor mu?
2. Aynı run için concurrent koşulun **median tek istek gecikmesinin** sequential'dan daha yüksek olup olmadığına bakın (H2).
3. `submit_batch` → `check_batch_status` → `fetch_batch_results` zincirini çalıştırın. `cost_comparison.csv`'deki `estimated_savings_percent`'i not edin.
4. Offline fixture testini okuyun (`tests/labs/test_lab09_offline_fixtures.py`) ve `partial_batch_output.jsonl` ile neden iki görevin `MISSING` olarak işaretlendiğini açıklayın.
5. `max_batch_input_requests` config değerini 2'ye düşürüp `submit_batch`'in neden reddettiğini gözlemleyin.

## 12. Sorun giderme

- **`403` / Batch API erişimi yok:** Groq hesabınızın planı Batch API'yi desteklemiyor olabilir. Bölüm 8'deki offline fixture modunu kullanın.
- **`LiveExecutionNotAllowed`:** `config/live.toml` oluşturup `--config config/live.toml --live` kullandığınızdan emin olun (ana `README.md`, **Setup**).
- **Model bulunamadı:** `config/live.toml`'daki `[lab09] model` değerini hesabınızın desteklediği bir modelle güncelleyin.
