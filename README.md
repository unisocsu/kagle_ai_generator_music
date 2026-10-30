# kagle_ai_generator_music

הפקת מוזיקה באמצעות **ACE-Step 1.5 XL Turbo** על GPU של Kaggle, בהפעלה דרך GitHub Actions.

## ארכיטקטורה

1. **GitHub Actions** – קבלת פרמטרים, אימות Kaggle, שליחת Kernel, מעקב והורדת תוצרים גם במקרה כישלון.
2. **Kernel package** – חבילת הריצה שנשלחת ל-Kaggle: הגדרות JSON, נקודת כניסה ומנוע.
3. **ACE-Step engine** – התקנת המנוע, טעינת המודלים והפקת WAV.

```
.github/workflows/generate-music.yml  # תזמור CI
scripts/prepare_kernel.py             # אימות ובניית חבילת Kaggle
kernel/run_music.py                   # נקודת כניסה, לוגים וטיפול בשגיאות
kernel/ace_step_engine.py             # טעינת מודלים והפקת מוזיקה
```

ההגדרות נשלחות כ-`config.json` נפרד ולא מוזרקות לתוך קוד Python.

## הגדרה חד-פעמית

ב-GitHub: Settings → Secrets and variables → Actions:

| Secret | תוכן |
|---|---|
| `kagle_key` | מפתח API של Kaggle |
| `kagle_user_name` | שם המשתמש ב-Kaggle |

יש לאפשר Internet ב-Kaggle כדי להוריד את קוד ACE-Step ואת משקלי המודלים.

## הפעלה

Actions → **Generate music on Kaggle (ACE-Step 1.5)** → Run workflow.

| שדה | משמעות | ברירת מחדל |
|---|---|---|
| `prompt` | תיאור מוזיקה בעברית או באנגלית | חובה |
| `lyrics` | מילים לשירה | ריק |
| `vocal_language` | שפת השירה | en |
| `instrumental` | אינסטרומנטלי בלבד | false |
| `duration` | אורך בשניות (10–600) | 30 |
| `num_samples` | מספר גרסאות (1–3) | 1 |
| `seed` | זרע קבוע או ‎-1 לאקראי | -1 |
| `wait_for_result` | להמתין, להוריד ולפרסם תוצאות | true |
| `max_wait_minutes` | זמן המתנה מרבי | 240 |

## תוצרים ואבחון

- WAV ו-`run_info.json` מתפרסמים ב-Release רק אם קיימים קובצי WAV.
- Artifact בשם `ace-step-run-<run>` נשמר גם כאשר Kaggle נכשל, ככל שפלט זמין.
- `generation.log` נשמר במהלך הריצה; `failure.txt` נכתב במקרה של חריגה.
- סטטוס Kaggle נשמר ב-`kaggle-result.txt`.

## הערות

- `enable_gpu=true` מבקש GPU מ-Kaggle; סוג המאיץ ומספר הכרטיסים בפועל נקבעים בצד Kaggle.
- שתי יחידות T4 אינן מאוחדות לזיכרון GPU יחיד.
- בכל ריצה חדשה ייתכן ש-Kaggle יוריד ויטען מחדש את משקלי המודל.
