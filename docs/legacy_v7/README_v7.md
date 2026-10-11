# MBMF Q1 Research Bundle — v7

این نسخه migration واقعی v6 است و هدفش این است که **تمام مدل‌ها زیر یک protocol واحد و leakage-safe** اجرا، انتخاب، تست، رتبه‌بندی، bootstrap، ablation و plot شوند. هیچ score مصنوعی/دودی (smoke) در این بسته به‌عنوان نتیجهٔ علمی گزارش نشده است.

## مدل‌ها

Baselineها:

- `MF-Bias` — فاکتورگیری ماتریسی استاندارد با bias؛ baseline کمکی و خارج از Methods 1–7.
- `MBMF-F`, `MBMF-C`, `MBMF-N` — بازسازی خانوادهٔ MBMF پایه.

مدل‌های اصلی:

1. `Method-1-AE-LatentMagnitude` — مدل نزدیک به formulation پایان‌نامه: norm کدهای latent به‌عنوان شعاع MBMF.
2. `Method-2-DualViewFusion` — دو predictor هندسی user/item و fusion؛ در preset اصلی وزن consistency از identity دقیق MSE به‌دست می‌آید.
3. `Method-3-PriorResidualMagnitude` — prior تاریخی + residual ضربی در log-radius؛ bounded نسبت به prior، ولی الزاماً range-safe نیست.
4. `Method-4-BoundedPriorResidual` — **مدل جدید v7**: residual در logit-radius space با hard analytical rating bound.
5. `Method-5-DVMBMF-Raw` — انتقال faithful ایدهٔ decoder-coupled dual-view از ZIP قدیمی با اصلاح leakage/zero-mask و پروتکل مشترک.
6. `Method-6-DVMBMF-RangeSafe` — همان coupling، با نگاشت شعاع range-safe و تضمین بازهٔ rating.
7. `Method-7-UAIBMF` — نسخهٔ بازبینی‌شدهٔ UC-IBMF؛ در v7 آن را *Uncertainty-Aware Inequality-Bounded MF* می‌نامیم و «calibrated» بودن را از قبل فرض نمی‌کنیم.

> دو ZIP `mbmf_q1_bundle` و `01_ucibmf_redesign_v2` از نظر source اصلی یکسان بودند؛ بنابراین یک Method مستقل از آن‌ها ساخته شده است، نه دو مدل تکراری.

## Method 4: Bounded Prior-Residual MBMF

فرض کنید ratingها در `[a,b]` باشند:

\[
m=(a+b)/2,\qquad h=(b-a)/2,\qquad c=\sqrt h.
\]

ابتدا prior شعاع فقط از train ساخته می‌شود و به بازهٔ feasible پرتاب می‌شود:

\[
r_0^{safe}\in(\epsilon,c-\epsilon),\qquad q_0=r_0^{safe}/c.
\]

شبکه فقط shift محدودشدهٔ \(\delta\) را یاد می‌گیرد:

\[
q=\sigma(\operatorname{logit}(q_0)+\delta),\qquad r=cq.
\]

در نتیجه همیشه \(0<r<c\). برای user/item:

\[
r_u r_i<h,
\]

و چون جهت‌ها unit-norm هستند:

\[
|w_u^T h_i|\le r_u r_i<h.
\]

پس بدون output clipping:

\[
a<\hat r_{ui}<b.
\]

در `delta=0` مدل دقیقاً prior قابل‌اجرا را بازیابی می‌کند. regularizer روی delta نیز مستقیماً regularization روی log-odds deviation از prior است.

## پروتکل v7

- split seed و model seed جدا هستند.
- rating range باید از مستندات dataset صریحاً تنظیم شود؛ از test برای تعیین scale استفاده نمی‌شود.
- encoderها، historical priors و history features فقط train را می‌بینند.
- Method 7 برای حفظ مقیاس‌پذیری source redesign با edge mini-batch + AdamW آموزش می‌بیند؛ selection/evaluation آن همچنان دقیقاً زیر pipeline مشترک v7 است.
- انتخاب configuration و checkpoint فقط با validation انجام می‌شود.
- جدول candidateها هیچ `test_*` ندارد؛ test performance فقط بعد از انتخاب validation محاسبه می‌شود.
- در ranking نهایی، interactionهای train **و validation** از کاندیدهای recommendation حذف می‌شوند.
- ranking exact full-catalog است و برای سرعت state مدل قبل از chunking cache می‌شود.
- خروجی اصلی آماری ابتدا model-seedها را داخل هر outer split میانگین می‌کند و سپس بین splitها خلاصه می‌کند.
- علاوه بر bootstrap جفت‌شدهٔ user در هر split، `hierarchical_paired_bootstrap.csv` split و user را به‌ترتیب سطوح بالا/داخل bootstrap می‌کند.

## معیارها

Rating prediction:

- RMSE, MAE
- OOB rate
- maximum rating-bound violation
- train-threshold macro/micro F1

Ranking:

- exact Precision@K, Recall@K, NDCG@K

Bound diagnostics:

- structural bound violations
- rating-bound violations
- radius/gate/prior diagnostics برای مدل‌هایی که آن state را دارند

Uncertainty برای Method 7 در حالت robust:

- Gaussian NLL
- PICP 50/80/90/95
- absolute coverage gap در همان سطوح
- MPIW و normalized MPIW
- mean absolute coverage-calibration error

در `m7_mse_only` این معیارهای uncertainty عمداً گزارش نمی‌شوند، چون variance در آن ablation با objective آموزش داده نشده است.

## اجرا

### اجرای سریع smoke

```bash
python run_v7.py \
  --data-dir /path/to/movielens \
  --rating-min 1 --rating-max 5 \
  --fast-debug \
  --output-dir results_smoke
```

### اجرای مطالعهٔ اصلی

```bash
python run_v7.py \
  --data-dir /path/to/ml-1m \
  --rating-min 1 --rating-max 5 \
  --split-seeds 2026,2027,2028,2029,2030 \
  --model-seeds 11,22,33,44,55 \
  --latent-dims 10,20,40 \
  --ranking-ks 10,20 \
  --output-dir results_v7
```

### ابلیشن‌ها

```bash
python run_v7.py \
  --data-dir /path/to/ml-1m \
  --rating-min 1 --rating-max 5 \
  --run-ablations \
  --output-dir results_v7_ablation
```

`make_ablation_variants` در حال حاضر 36 configuration ازپیش‌تعریف‌شده دارد و هر family دارای `full` reference است.

## خروجی‌ها

خروجی معمول شامل این فایل‌هاست:

- `candidate_validation_runs.csv`
- `selected_by_validation_candidates.csv`
- `locked_test_runs.csv`
- `locked_test_run_summary.csv`
- `locked_test_split_means.csv`
- `locked_test_summary.csv` ← خلاصهٔ اصلی برای مقاله
- `per_user_test_metrics.csv`
- `paired_user_bootstrap.csv`
- `hierarchical_paired_bootstrap.csv`
- `split_assignments.csv`
- `training_histories.json`
- `dataset_manifest.json`
- `environment.json`
- `plots/`

## تست

```bash
pytest -q
python mbmf_q1_final_v7.py
```

در زمان ساخت این bundle، unit tests، synthetic self-tests، cache-parity، end-to-end locked-test و تمام 36 ablation روی دیتای synthetic/smoke اجرا شده‌اند. این runها فقط **مهندسی** هستند و نباید به‌عنوان نتیجهٔ علمی نقل شوند.

## محدودیت‌های آگاهانه

- v7 loader اصلی را از v6 حفظ کرده و برای MovieLens-style `ratings.csv` / `ratings.dat` طراحی شده است. adapterهای Jester/Book-Crossing عمداً در این migration مخفیانه اضافه نشده‌اند؛ برای مقایسهٔ چند dataset باید adapter مستقل و تست‌شده اضافه شود.
- Methods 5–6 در v7 لایهٔ اول sparse و دقیقاً معادل Linear روی `[values, mask]` دارند. Methods 1/3/4 همچنان dense-history هستند و برای datasetهای بسیار بزرگ ممکن است نیاز به sparse first-layer مستقل داشته باشند.
- theoremهای hard rating-range برای preset اصلی Methods 4, 6, 7 برقرارند. Methods 1, 3, 5 در حالت raw theorem ساختاری/وابسته به شعاع دارند، نه تضمین بازهٔ rating.
- نتایج واقعی Q1 باید با چند outer split، چند initialization، و competitorهای خارجی روی split manifests یکسان تولید شوند؛ smoke score اعتبار مقاله‌ای ندارد.

برای جزئیات تغییرات و سوتی‌های اصلاح‌شده، `docs/MIGRATION_AUDIT_V6_TO_V7_FA.md` و برای فرمول‌ها `docs/THEORETICAL_PROPERTIES_V7.md` را ببینید.
