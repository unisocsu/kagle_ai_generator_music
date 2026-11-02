# kagle_ai_generator_music

הפקת מוזיקה עם **ACE-Step 1.5 XL Turbo** על GPU חינמי של Kaggle, בהפעלה מ-GitHub Actions.
בוחרים prompt (ומילים, אם רוצים), מריצים workflow, ומקבלים קובצי WAV ב-Release.

## איך זה עובד

```
GitHub Actions ──► scripts/prepare_kernel.py ──► kaggle kernels push
                                                      │
                                         Kaggle (GPU T4, אינטרנט פתוח)
                                         kernel/run_music.py
                                           1. pip install uv
                                           2. git clone ACE-Step-1.5
                                           3. uv sync  (Python 3.12 מבודד + torch של הפרויקט)
                                           4. kernel/generate.py בתוך הסביבה הזו
                                                      │
GitHub Release (WAV + run_info.json) ◄── kaggle kernels output
```

| קובץ | תפקיד |
|---|---|
| `.github/workflows/generate-music.yml` | תזמור: אימות, שליחה, המתנה, הורדה, פרסום |
| `scripts/prepare_kernel.py` | בדיקת קלט ובניית קובץ יחיד ל-Kaggle (הגדרות + מנוע מוטמעים בו) |
| `kernel/run_music.py` | נקודת הכניסה ב-Kaggle (ספרייה סטנדרטית בלבד): התקנה, הרצה, לוגים |
| `kernel/generate.py` | המנוע: הורדת מודלים, אתחול, יצירה, כתיבת `run_info.json` |

Kaggle מריץ קובץ אחד לכל kernel, ולכן `prepare_kernel.py` מטמיע את `generate.py` ואת ההגדרות בתוך `run_music.py`.
הקבצים נשארים נפרדים ב-Git, והקובץ המאוחד נוצר רק בזמן הריצה (`build/`, לא נשמר).

## הגדרה חד-פעמית

ב-GitHub: Settings → Secrets and variables → Actions:

| Secret | תוכן |
|---|---|
| `kagle_user_name` | שם המשתמש ב-Kaggle |
| `kagle_key` | מפתח API של Kaggle (32 תווי hex או access token) |

ב-Kaggle צריך חשבון עם אימות טלפון (נדרש כדי לקבל GPU ואינטרנט בתוך kernel).

## הפעלה

Actions → **Generate music on Kaggle (ACE-Step 1.5)** → Run workflow.

| שדה | משמעות | ברירת מחדל |
|---|---|---|
| `prompt` | תיאור המוזיקה, עד 512 תווים | חובה |
| `lyrics` | מילים, עד 4096 תווים | ריק |
| `vocal_language` | שפת השירה | `en` |
| `instrumental` | אינסטרומנטלי בלבד | `false` |
| `duration` | אורך בשניות (10–600) | `30` |
| `num_samples` | מספר גרסאות (1–3) | `1` |
| `seed` | זרע קבוע, או `-1` לאקראי | `-1` |
| `wait_for_result` | להמתין, להוריד ולפרסם | `true` |
| `max_wait_minutes` | זמן המתנה מרבי | `240` |

## תוצרים ואבחון

- **Release** `ace-step-run-<run>-<attempt>`: קובצי `music_XX.wav` ו-`run_info.json`.
- **Artifact** `ace-step-run-<run>` נשמר תמיד, גם בכישלון: `generation.log` (הלוג המלא), `failure.txt` (השגיאה), וכן `kaggle-result.txt`.
- בכישלון, ה-workflow מדפיס את `failure.txt` ישירות בלוג של Actions.

## החלטות עיצוב

- **סביבה מבודדת עם `uv sync`**: ACE-Step מגיע עם `uv.lock` ו-torch משלו, ולכן לא מתקינים אותו על חבילות המערכת של Kaggle (זה מה שגרם להתנגשויות setuptools/hatchling).
- **קבצים כבדים מחוץ ל-`/kaggle/working`**: הריפו, הסביבה והמשקלים נשמרים ב-`/kaggle/temp`, כדי ש-`kaggle kernels output` יוריד רק את ה-WAV והלוגים.
- **תאימות לגרסאות**: `generate.py` מעביר ל-ACE-Step רק פרמטרים שהגרסה המותקנת מכירה, ומנסה אתחול חלופי (בלי קוונטיזציה) ואחר כך בלי LM, במקום ליפול על הבדל קטן בחתימה.
- שני כרטיסי T4 אינם מאוחדים לזיכרון אחד; המנוע משתמש בכרטיס הראשון עם offload ל-CPU.

## Worker מתמשך ליצירת שירים

בנוסף ליצירה החד-פעמית, נוסף מצב Worker בסגנון הפרויקט [cpu-and-gpu-ai-runs](https://github.com/unisocsu/cpu-and-gpu-ai-runs):

- דף GitHub Pages: `docs/index.html`.
- בקשות חדשות נשמרות כ-GitHub Issues עם הכותרת `[music-job]`.
- Worker אחד ב-Kaggle קורא את התור ומעבד שירים בזה אחר זה.
- הקבצים עולים ל-GitHub Release נפרד לכל בקשה, והדף מציג נגן והורדה.
- ה־DiT וה־planner נשארים טעונים בזיכרון לאורך חיי ה־Worker. לא מורידים ולא מאתחלים אותם מחדש לכל שיר.
- ברירת המחדל של ה־Worker היא `acestep-v15-turbo` (דגם 2B) יחד עם `acestep-5Hz-lm-0.6B`. זה קטן ומהיר יותר מה־XL Turbo 4B שבמצב החד-פעמי, במחיר פוטנציאלי של ירידה מסוימת באיכות/יכולת התכנון.

### הפעלה ראשונית

1. ודא שקיימים אותם Kaggle secrets שמוגדרים למעלה: `kagle_user_name` ו־`kagle_key`.
2. ב־GitHub → Settings → Pages, בחר **Deploy from a branch**, ענף `main`, תיקייה `/docs`.
3. פתח את דף ה־Pages. לחץ ⚙️ והדבק Fine-grained PAT שמוגבל לריפו הזה עם הרשאות:
   - **Issues: Read and write**
   - **Actions: Read and write**
4. לחץ **הפעל Worker**. ההפעלה הראשונה מורידה ומאתחלת את המודלים, ולכן היא איטית יותר מהבקשות הבאות.
5. אחרי שהמצב מציג **מחובר · מוכן**, שלח תיאור שיר. הבקשות נכנסות לתור, וה־Worker יוצר אותן ברצף.
6. לחץ **עצור Worker** כדי לבקש כיבוי מסודר. אפשר גם להמתין לכיבוי אוטומטי אחרי 30 דקות ללא בקשות.

ה־Worker מוגבל לאורך החיים של Kaggle ולמכסות שלו: הוא אינו שרת תמידי ללא הגבלה. ברירת המחדל היא כיבוי אחרי 30 דקות ללא פעילות או אחרי עד 5 שעות עבודה; אפשר להפעיל אותו מחדש מאותו דף.

**פרטיות:** הריפו הזה ציבורי. תוכן הבקשות, תגובות הסטטוס וקובצי ה־WAV שמפורסמים ב־Releases יהיו ציבוריים. אל תשלח מידע פרטי. ה־PAT נשמר ב־localStorage בדפדפן בלבד; ה־Worker מקבל GITHUB_TOKEN מוגבל של ריצת ה־Actions שמפעילה אותו, וה־Kaggle kernel מוגדר כפרטי.

### קבצי ה־Worker

| קובץ | תפקיד |
|---|---|
| `.github/workflows/start-music-worker.yml` | הפעלת Kernel מתמשך והמתנה לסיום |
| `.github/workflows/stop-music-worker.yml` | בקשת כיבוי דרך Issue הסטטוס |
| `scripts/push_music_worker.py` | בניית Kernel פרטי ודחיפתו ל־Kaggle |
| `scripts/wait_music_worker.py` | המתנה למצב הסיום של Kaggle |
| `kernel/music_worker_bootstrap.py` | התקנת ACE-Step בסביבת Python 3.12 והפעלת המנוע |
| `kernel/music_worker.py` | טעינת המודלים פעם אחת, תור הבקשות ופרסום קובצי האודיו |
| `docs/index.html` | ממשק יצירה, מצב Worker, תור וניגון תוצאות |

