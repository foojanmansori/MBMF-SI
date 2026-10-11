# بسته نهایی MBMF/Q1 — نسخه 8

## اصل غیرقابل‌تغییر

نسخه 8 **فقط infrastructure** است. فایل علمی زیر از V7 کاربر عیناً و byte-for-byte نگه داشته شده است:

`mbmf_q1_final_v7.py`

SHA-256 هسته علمی:

`6da726cb155ab6ad291f5f608febc37f42d1c25ac3924910f56917908c1f124c`

در V8 هیچ معماری، تابع زیان، فرمول شعاع/زاویه، forward، یا trainer علمی Methods 1–7 و MBMF-F/C/N ویرایش یا monkey-patch نشده است. فایل `SCIENTIFIC_CORE_MANIFEST_V8.json` علاوه بر هش کل فایل، هش source نمادهای علمی اصلی را ثبت می‌کند.

## چه چیزهایی اضافه شد

- پشتیبانی first-class از Synthetic، Jester، Book-Crossing، ML-100K، ML-1M و ML-10M.
- انتخاب configuration فقط با validation و test قفل‌شده تا بعد از انتخاب.
- گزارش train/validation/test برای checkpoint انتخاب‌شده.
- ذخیره/لود پایدار selected models همراه با SHA هسته و signature داده train.
- AutoRec و AEMC-NE به عنوان reimplementation؛ و adapter مشترک برای خروجی author-code روش‌های BMC-ADMM، Deep-MMMF و DRPRA.
- معیارهای گسترده accuracy، tail error، correlation، OOB، individual bounds، ranking، uncertainty، runtime/inference/memory و stability.
- Wilcoxon-Holm و Friedman/Nemenyi/critical-difference.
- نمودارها هم PNG و هم PDF وکتوری.
- K اصلی برابر 10/20/50.
- 38 ablation یکتا؛ دو ablation تکراری V7 حذف و با ablationهای مکانیزمی جایگزین شدند.
- hyperparameter scan جدا از ablation.
- timestamp ناقص دیگر با صفر پر نمی‌شود؛ time feature فقط وقتی فعال است که تمام timestampهای train معتبر باشند.
- notebook دارای cell-idهای صریح و RUN_* switches است.

## Double-check عدم تغییر روش‌ها

علاوه بر مقایسه byte-level هسته V7، تست‌های زیر اجرا می‌شوند:

1. dense/sparse prediction equivalence برای Methods 1–4؛
2. **one-step Adam/update equivalence برای Methods 1–6**؛ یعنی از initialization یکسان، یک epoch در مسیر مرجع و مسیر scalable اجرا می‌شود و پارامترهای نهایی باید در tolerance عددی یکسان باشند؛
3. checkpoint round-trip؛
4. held-out-label invariance برای Methods 1–7؛
5. geometry tests برای hyperspherical conversion؛
6. range-safety smoke test برای Methods 4/6/7.

Method 7 از trainer edge-minibatch خود هسته frozen V7 استفاده می‌کند و تغییر داده نشده است.

## اجرای سریع

```bash
pip install -r requirements.txt
python run_v8.py --self-test-only
```

دانلود دیتاست‌ها:

`download_datasets_q1_v8.ipynb`

Pilot:

```bash
python run_v8.py --dataset ml-1m --quick --output ./results_q1_v8/ml1m_quick
```

اجرای اصلی ML-1M:

```bash
python run_v8.py \
  --dataset ml-1m \
  --data-root ./datasets \
  --output ./results_q1_v8/ml-1m \
  --split-seeds 2026,2027,2028,2029,2030 \
  --model-seeds 11,22,33,44,55 \
  --k 10,20,50
```

همه شش دیتاست:

```bash
python run_v8.py --dataset all --data-root ./datasets --output ./results_q1_v8/all
```

Ablation:

```bash
python run_v8.py --dataset ml-1m --data-root ./datasets --output ./results_q1_v8/ml1m_ablation --ablation
```

Hyperparameter scan:

```bash
python run_v8.py --dataset ml-1m --data-root ./datasets --output ./results_q1_v8/ml1m_hyper --hyper-scan
```

## قانون مقاله

هیچ K/init/hyperparameter از test انتخاب نشود. Candidate table فقط train+validation دارد. پس از validation selection، checkpoint locked روی train/validation/test ارزیابی و در `selected_models/` ذخیره می‌شود.

نتایج smoke/integration فقط صحت مهندسی را نشان می‌دهند و نباید به‌عنوان نتیجه پژوهشی گزارش شوند.
