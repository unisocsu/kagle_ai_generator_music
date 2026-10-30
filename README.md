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
