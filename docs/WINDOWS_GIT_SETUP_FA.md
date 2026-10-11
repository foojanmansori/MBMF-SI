# راهنمای Git و راه‌اندازی V8 در Windows / VS Code

## رفع خطای `resource`

در Python ویندوز ماژول Unix-only به نام `resource` وجود ندارد. اصلاح
`mbmf_q1_infrastructure_v8.py` باعث می‌شود `resource` فقط در fallback
لینوکس/macOS بارگذاری شود؛ در ویندوز پایش RSS از `psutil` استفاده می‌کند.
مدل‌ها، روش‌های رقیب، دیتاست‌ها و ابلیشن‌ها دست‌نخورده‌اند.

## نصب Git (PowerShell)

```powershell
git --version
# اگر git پیدا نشد:
winget install --id Git.Git -e --source winget
# ترمینال/VS Code را مجدداً باز کنید:
git --version
git config --global user.name "YOUR_GITHUB_USERNAME"
git config --global user.email "YOUR_EMAIL"
```

## تفاوت ZIP و Clone

فولدر استخراج‌شده ZIP معمولاً `.git` ندارد؛ داخل آن `git pull` کار نمی‌کند.
آن فولدر را برای بکاپ نگه دارید و ریپوی مستقل بسازید:

```powershell
cd "$HOME\Desktop\RUN"
git clone https://github.com/prestigegitserp/mb_v8.git
cd mb_v8
git status
code .
```

پروژه private است؛ برای کلون/پوش احتمالاً به تأیید GitHub نیاز دارید.

## ساخت محیط Python 3.11

```powershell
conda create -n mbmf_q1 python=3.11 -y
conda activate mbmf_q1
python -m pip install -r requirements.txt
python -m pip install --upgrade psutil
python -c "import mbmf_q1_infrastructure_v8 as pub; print('V8 import OK')"
python -m pytest -q
python -m pytest -q -m integration
```

در VS Code از `Python: Select Interpreter` همان محیط `mbmf_q1` را انتخاب کنید.
Notebook را از **همین فولدر cloneشده** باز کنید، نه از ZIP قدیمی.

## گرفتن تغییرات GitHub بدون از دست دادن کار

```powershell
git status
git pull --ff-only origin main
```

اگر فایل‌های محلی تغییر کرده‌اند، اول `git diff` بگیرید و آنها را commit کنید
یا موقتاً کنار بگذارید:

```powershell
git stash push -u -m "my local work before update"
git pull --ff-only origin main
git stash pop
```

در صورت conflict، فایل‌های دارای علامت conflict را در VS Code باز کنید و نسخه
درست را انتخاب کنید. `git reset --hard`، `git clean -fd` و `git push --force`
را روی کار پژوهشی بدون بکاپ اجرا نکنید.

## گردش کار روزانه: status → diff → commit → push

```powershell
git switch -c fix/my-experiment-notes
git status
git diff
git add mbmf_q1_final_v8.ipynb
git commit -m "Document experiment setup"
git push -u origin fix/my-experiment-notes
```

برای ادغام امن، در GitHub از branch خود به `main` یک Pull Request باز کنید.
بهتر است تغییرات مدل علمی و تغییرات زیرساخت را در commitهای جدا نگه دارید.

برای مشاهده تاریخچه و مشخص کردن نسخه اجرای آزمایش:

```powershell
git log --oneline -5
git rev-parse HEAD
```

SHA آخر را کنار نتیجه هر آزمایش مقاله ثبت کنید. دیتاست‌های خام،
نتایج حجیم و checkpointها طبق `.gitignore` نباید وارد مخزن کد شوند.
