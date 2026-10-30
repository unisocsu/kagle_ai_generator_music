# kagle_ai_generator_music

הפקת מוזיקה עם מודלי MusicGen על GPU של Kaggle, בהפעלה מ-GitHub Actions.
מזינים הנחיות בטופס ה-dispatch, ה-workflow שולח הרצה ל-Kaggle, ממתין לסיום ומעלה את קבצי ה-WAV כ-artifact.

## הגדרה חד-פעמית

ב-GitHub: Settings → Secrets and variables → Actions → New repository secret. צריך שני סודות:

| שם | תוכן |
|---|---|
| `kagle_key` | מפתח ה-API של Kaggle (השדה `key` בקובץ `kaggle.json`) |
| `kagle_user_name` | שם המשתמש ב-Kaggle (השדה `username`) |

הערות:
- שמות סודות מכילים רק אותיות, ספרות וקו תחתון (אין מקף), ואינם תלויים באותיות גדולות או קטנות.
- בחשבון ה-Kaggle צריך שאפשר יהיה להפעיל Internet בנוטבוקים (דורש אימות טלפון), כדי שההרצה תוכל להוריד את המודל.

## הפעלה

Actions → **Generate music on Kaggle** → Run workflow, ואז ממלאים:

| שדה | משמעות | ברירת מחדל |
|---|---|---|
| `prompt` | תיאור המוזיקה, באנגלית | חובה |
| `model` | `musicgen-small` / `musicgen-medium` / `musicgen-large` | `musicgen-large` |
| `duration` | אורך בשניות (1-30) | 15 |
| `num_samples` | כמה גרסאות להפיק (1-5) | 1 |
| `seed` | זרע לשחזור תוצאה, `-1` לאקראי | -1 |
| `wait_for_result` | להמתין לסיום ולהוריד את התוצאה | כן |
| `max_wait_minutes` | זמן המתנה מקסימלי | 120 |

## תוצאות

- אחרי שהרצה מצליחה: ה-workflow אורז את הכול ל-zip ומפרסם **Release** חדש (`music-run-<מספר>`). הקישור לעמוד ה-Release ולהורדה ישירה מודפס בלוג של שלב **Zip the results and publish a release**, ובסיכום ההרצה (Summary). ה-zip כולל את קבצי ה-WAV, לוג ו-`run_info.json`.
- גיבוי: הקבצים מועלים גם כ-artifact בשם `music-output` (נשמר לזמן מוגבל), גם כשההרצה נכשלה.
- אם הריפו ציבורי, גם ה-Releases ציבוריים. כדי שהמוזיקה תישאר פרטית, הגדר את הריפו כ-Private.
- ה-workflow זקוק להרשאת כתיבה ל-contents (מוגדר בקובץ). אם ההרצה נכשלת ביצירת ה-Release, בדוק ב-Settings → Actions → General → Workflow permissions.
- אם כיבית את `wait_for_result`, ההרצה ממשיכה ב-Kaggle והתוצאות יהיו בנוטבוק הפרטי `kagle-ai-music-generator` בחשבון שלך.

## הערות

- MusicGen מבין תיאורים באנגלית בלבד.
- המודל אומן על קטעים של עד 30 שניות.
- כל הרצה נספרת במכסת ה-GPU השבועית ב-Kaggle, וטוענת את המודל מחדש.
- מתבצעת הרצה אחת בכל פעם (ההרצות הבאות ממתינות בתור).

## מבנה הריפו

```
.github/workflows/generate-music.yml   # ה-workflow עם workflow_dispatch
scripts/prepare_kernel.py              # בדיקת הקלט ובניית תיקיית הנוטבוק
kernel/run_music.py                    # הקוד שרץ בתוך Kaggle
requirements.txt                       # להרצה מקומית בלבד
```
