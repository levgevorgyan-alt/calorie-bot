import json
import logging
import os
import sqlite3
import sys
import base64
from datetime import datetime, timedelta, time, timezone

from groq import Groq
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
PORT = int(os.environ.get("PORT", "10000"))

DB_PATH = os.environ.get("DB_PATH", "calories.db")
DEFAULT_CALORIE_LIMIT = 1800
DAILY_WATER_TARGET_ML = 2000

groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a nutrition assistant. Given a meal description, estimate calories "
    "and macronutrients for each item and provide totals. Consider the cooking "
    "method (fried, boiled, grilled, raw) when estimating. "
    "Return ONLY valid JSON with no extra text:\n"
    '{"items": [{"name": "<food>", "portion": "<weight or description>", '
    '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>, '
    '"per_100g_raw": <int>, "per_100g_cooked": <int>}], '
    '"total": <int>, "total_protein_g": <int>, "total_fat_g": <int>, '
    '"total_carbs_g": <int>}\n\n'
    "If the user does not specify a weight, assume a reasonable portion and state it "
    "in the 'portion' field (e.g. '150g', '1 medium'). Always include per_100g_raw "
    "and per_100g_cooked for each item. If raw and cooked values are the same "
    "(e.g. for bread, drinks), use the same number for both. "
    "If a description is unclear, make your best estimate and note assumptions "
    "in a short 'note' field.\n\n"
    "Example input: 2 scrambled eggs, 1 toast with butter\n"
    "Example output: "
    '{"items": [{"name": "Scrambled eggs", "portion": "2 large (100g)", '
    '"calories": 182, "protein_g": 12, "fat_g": 14, "carbs_g": 2, '
    '"per_100g_raw": 143, "per_100g_cooked": 182}, '
    '{"name": "Toast with butter", "portion": "1 slice (40g bread + 10g butter)", '
    '"calories": 178, "protein_g": 4, "fat_g": 9, "carbs_g": 21, '
    '"per_100g_raw": 265, "per_100g_cooked": 265}], '
    '"total": 360, "total_protein_g": 16, "total_fat_g": 23, "total_carbs_g": 23}'
)

VISION_PROMPT = (
    "You are a nutrition assistant. Look at this photo of a meal and identify "
    "each food item visible. Estimate calories and macronutrients for each item. "
    "Consider the cooking method visible in the photo. "
    "Return ONLY valid JSON with no extra text:\n"
    '{"items": [{"name": "<food>", "portion": "<estimated weight or description>", '
    '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>, '
    '"per_100g_raw": <int>, "per_100g_cooked": <int>}], '
    '"total": <int>, "total_protein_g": <int>, "total_fat_g": <int>, '
    '"total_carbs_g": <int>}\n\n'
    "Estimate portion sizes from visual cues (plate size, utensils, etc). "
    "Always include per_100g_raw and per_100g_cooked for each item. "
    "If you cannot identify a food clearly, make your best guess and add a "
    "'note' field explaining your assumptions."
)

PROFILE_PROMPT = (
    "You are a nutrition expert. Based on the following person's measurements, "
    "recommend their optimal daily calorie intake and macronutrient targets "
    "for maintaining a healthy weight at their specified activity level. "
    "Return ONLY valid JSON with no extra text:\n"
    '{"daily_calories": <int>, "daily_protein_g": <int>, '
    '"daily_fat_g": <int>, "daily_carbs_g": <int>}\n\n'
    "Use established nutrition science (Mifflin-St Jeor for BMR). "
    "Activity multipliers: sedentary=1.2, light=1.375, moderate=1.55, "
    "active=1.725, very_active=1.9. "
    "Protein should be ~1.6-2.0g per kg bodyweight. "
    "Fat should be ~25-30% of calories. Remaining calories from carbs."
)

ACTIVITY_LEVELS = {
    "sedentary": "little or no exercise",
    "light": "exercise 1-3 days/week",
    "moderate": "exercise 3-5 days/week",
    "active": "exercise 6-7 days/week",
    "very_active": "hard exercise daily or physical job",
}

GOALS = {
    "lose_weight": "calorie deficit for fat loss",
    "maintain": "maintain current weight",
    "gain_muscle": "calorie surplus for muscle gain",
}

MEALPLAN_PROMPT = (
    "You are a professional dietitian. Create a complete 7-day meal plan "
    "based on the user's profile, goals, and preferences below. "
    "Return ONLY valid JSON with no extra text.\n\n"
    "Rules:\n"
    "- Each day must hit the daily calorie and macro targets closely.\n"
    "- Respect the excluded foods strictly - never include them.\n"
    "- Vary meals across the week - do not repeat the same meal on consecutive days.\n"
    "- Use commonly available ingredients.\n"
    "- Keep the total weekly shopping cost within the specified budget.\n"
    "- Consolidate the shopping list: combine quantities, no duplicate items.\n"
    "- Estimate realistic local prices in the specified currency.\n"
    "- For lose_weight goal: prioritize high-protein, high-fiber, lower-calorie meals.\n"
    "- For gain_muscle goal: prioritize high-protein, calorie-dense meals.\n\n"
    "JSON format:\n"
    '{"days": [{"day": "Monday", "meals": [{"slot": "<meal name>", "time": "<HH:MM>", '
    '"name": "<dish name>", "ingredients": ["<item qty>", ...], '
    '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>}], '
    '"day_total": {"calories": <int>, "protein_g": <int>, "fat_g": <int>, '
    '"carbs_g": <int>}}], '
    '"shopping_list": [{"item": "<name>", "quantity": "<amount>", '
    '"estimated_cost": <float>}], "total_cost": <float>, "currency": "<code>"}'
)

# Water reminder schedule: (hour, minute, amount_ml) in UTC
WATER_SCHEDULE = [
    (11, 0, 500),
    (14, 0, 500),
    (16, 0, 250),
    (18, 0, 500),
    (20, 0, 250),
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    conn = db_connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT,
            chat_id INTEGER,
            meal_text TEXT,
            calories INTEGER NOT NULL,
            protein_g INTEGER NOT NULL DEFAULT 0,
            fat_g INTEGER NOT NULL DEFAULT 0,
            carbs_g INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS limits (
            user_id INTEGER PRIMARY KEY,
            daily_limit INTEGER NOT NULL DEFAULT 1800,
            daily_protein_g INTEGER NOT NULL DEFAULT 0,
            daily_fat_g INTEGER NOT NULL DEFAULT 0,
            daily_carbs_g INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY,
            height_cm INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            activity TEXT NOT NULL DEFAULT 'moderate',
            rec_calories INTEGER NOT NULL DEFAULT 0,
            rec_protein_g INTEGER NOT NULL DEFAULT 0,
            rec_fat_g INTEGER NOT NULL DEFAULT 0,
            rec_carbs_g INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS water (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT,
            chat_id INTEGER,
            amount_ml INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reminder_chats (
            chat_id INTEGER PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS diet_prefs (
            user_id INTEGER PRIMARY KEY,
            goal TEXT NOT NULL DEFAULT 'maintain',
            schedule TEXT NOT NULL DEFAULT '[]',
            excludes TEXT NOT NULL DEFAULT '[]',
            budget_amount REAL NOT NULL DEFAULT 0,
            budget_currency TEXT NOT NULL DEFAULT 'EUR'
        );
        CREATE TABLE IF NOT EXISTS meal_plans (
            user_id INTEGER PRIMARY KEY,
            week_start TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            shoplist_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    # Migrate older meals table that may lack macro columns
    try:
        conn.execute("ALTER TABLE meals ADD COLUMN protein_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE meals ADD COLUMN fat_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE meals ADD COLUMN carbs_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    # Migrate limits table
    try:
        conn.execute("ALTER TABLE limits ADD COLUMN daily_protein_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE limits ADD COLUMN daily_fat_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE limits ADD COLUMN daily_carbs_g INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE profiles ADD COLUMN activity TEXT NOT NULL DEFAULT 'moderate'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def save_meal(user_id: int, username: str, chat_id: int,
              meal_text: str, calories: int,
              protein_g: int = 0, fat_g: int = 0, carbs_g: int = 0) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO meals (user_id, username, chat_id, meal_text, calories,"
        " protein_g, fat_g, carbs_g, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, username, chat_id, meal_text, calories,
         protein_g, fat_g, carbs_g,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_today_meals(user_id: int) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = db_connect()
    rows = conn.execute(
        "SELECT meal_text, calories, protein_g, fat_g, carbs_g, created_at"
        " FROM meals WHERE user_id = ? AND created_at LIKE ?",
        (user_id, f"{today}%"),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_today_totals(user_id: int) -> dict:
    """Return today's total calories and macros."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = db_connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(calories), 0) AS calories,"
        " COALESCE(SUM(protein_g), 0) AS protein_g,"
        " COALESCE(SUM(fat_g), 0) AS fat_g,"
        " COALESCE(SUM(carbs_g), 0) AS carbs_g"
        " FROM meals WHERE user_id = ? AND created_at LIKE ?",
        (user_id, f"{today}%"),
    ).fetchone()
    conn.close()
    return dict(row)


def get_week_summary(user_id: int) -> list[dict]:
    """Return daily totals for the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")
    conn = db_connect()
    rows = conn.execute(
        "SELECT SUBSTR(created_at, 1, 10) AS day, SUM(calories) AS total,"
        " SUM(protein_g) AS protein_g, SUM(fat_g) AS fat_g,"
        " SUM(carbs_g) AS carbs_g"
        " FROM meals WHERE user_id = ? AND created_at >= ?"
        " GROUP BY day ORDER BY day",
        (user_id, since),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_targets(user_id: int) -> dict:
    """Return daily calorie and macro targets for a user."""
    conn = db_connect()
    row = conn.execute(
        "SELECT daily_limit, daily_protein_g, daily_fat_g, daily_carbs_g"
        " FROM limits WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "daily_limit": DEFAULT_CALORIE_LIMIT,
        "daily_protein_g": 0, "daily_fat_g": 0, "daily_carbs_g": 0,
    }


def set_targets(user_id: int, calories: int,
                protein_g: int = 0, fat_g: int = 0, carbs_g: int = 0) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO limits (user_id, daily_limit, daily_protein_g, daily_fat_g,"
        " daily_carbs_g) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(user_id) DO UPDATE SET daily_limit = excluded.daily_limit,"
        " daily_protein_g = excluded.daily_protein_g,"
        " daily_fat_g = excluded.daily_fat_g,"
        " daily_carbs_g = excluded.daily_carbs_g",
        (user_id, calories, protein_g, fat_g, carbs_g),
    )
    conn.commit()
    conn.close()


def save_profile(user_id: int, height_cm: int, weight_kg: float,
                 age: int, gender: str, activity: str, rec_calories: int,
                 rec_protein_g: int, rec_fat_g: int, rec_carbs_g: int) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO profiles (user_id, height_cm, weight_kg, age, gender, activity,"
        " rec_calories, rec_protein_g, rec_fat_g, rec_carbs_g)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(user_id) DO UPDATE SET height_cm = excluded.height_cm,"
        " weight_kg = excluded.weight_kg, age = excluded.age,"
        " gender = excluded.gender, activity = excluded.activity,"
        " rec_calories = excluded.rec_calories,"
        " rec_protein_g = excluded.rec_protein_g,"
        " rec_fat_g = excluded.rec_fat_g, rec_carbs_g = excluded.rec_carbs_g",
        (user_id, height_cm, weight_kg, age, gender, activity,
         rec_calories, rec_protein_g, rec_fat_g, rec_carbs_g),
    )
    conn.commit()
    conn.close()


def get_profile(user_id: int) -> dict | None:
    conn = db_connect()
    row = conn.execute(
        "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_water(user_id: int, username: str, chat_id: int, amount_ml: int) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO water (user_id, username, chat_id, amount_ml, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (user_id, username, chat_id, amount_ml,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_today_water(user_id: int) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = db_connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(amount_ml), 0) AS total FROM water"
        " WHERE user_id = ? AND created_at LIKE ?",
        (user_id, f"{today}%"),
    ).fetchone()
    conn.close()
    return row["total"]


def add_reminder_chat(chat_id: int) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT OR IGNORE INTO reminder_chats (chat_id) VALUES (?)", (chat_id,)
    )
    conn.commit()
    conn.close()


def remove_reminder_chat(chat_id: int) -> None:
    conn = db_connect()
    conn.execute("DELETE FROM reminder_chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


def get_reminder_chats() -> list[int]:
    conn = db_connect()
    rows = conn.execute("SELECT chat_id FROM reminder_chats").fetchall()
    conn.close()
    return [r["chat_id"] for r in rows]


def reset_today_meals(user_id: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = db_connect()
    conn.execute(
        "DELETE FROM meals WHERE user_id = ? AND created_at LIKE ?",
        (user_id, f"{today}%"),
    )
    conn.commit()
    conn.close()


def reset_today_water(user_id: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = db_connect()
    conn.execute(
        "DELETE FROM water WHERE user_id = ? AND created_at LIKE ?",
        (user_id, f"{today}%"),
    )
    conn.commit()
    conn.close()


def reset_all_data(user_id: int) -> None:
    conn = db_connect()
    conn.execute("DELETE FROM meals WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM water WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM limits WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM diet_prefs WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM meal_plans WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def save_diet_prefs(user_id: int, goal: str, schedule: list,
                    excludes: list, budget_amount: float,
                    budget_currency: str) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO diet_prefs (user_id, goal, schedule, excludes,"
        " budget_amount, budget_currency) VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(user_id) DO UPDATE SET goal = excluded.goal,"
        " schedule = excluded.schedule, excludes = excluded.excludes,"
        " budget_amount = excluded.budget_amount,"
        " budget_currency = excluded.budget_currency",
        (user_id, goal, json.dumps(schedule), json.dumps(excludes),
         budget_amount, budget_currency),
    )
    conn.commit()
    conn.close()


def update_diet_pref(user_id: int, **kwargs) -> None:
    """Update specific diet preference fields."""
    conn = db_connect()
    # Ensure row exists
    conn.execute(
        "INSERT OR IGNORE INTO diet_prefs (user_id, goal, schedule, excludes,"
        " budget_amount, budget_currency) VALUES (?, 'maintain', '[]', '[]', 0, 'EUR')",
        (user_id,),
    )
    for key, value in kwargs.items():
        if key in ("schedule", "excludes"):
            value = json.dumps(value)
        conn.execute(f"UPDATE diet_prefs SET {key} = ? WHERE user_id = ?",
                     (value, user_id))
    conn.commit()
    conn.close()


def get_diet_prefs(user_id: int) -> dict | None:
    conn = db_connect()
    row = conn.execute(
        "SELECT * FROM diet_prefs WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["schedule"] = json.loads(d["schedule"])
    d["excludes"] = json.loads(d["excludes"])
    return d


def save_meal_plan(user_id: int, week_start: str,
                   plan_json: str, shoplist_json: str) -> None:
    conn = db_connect()
    conn.execute(
        "INSERT INTO meal_plans (user_id, week_start, plan_json, shoplist_json,"
        " created_at) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(user_id) DO UPDATE SET week_start = excluded.week_start,"
        " plan_json = excluded.plan_json, shoplist_json = excluded.shoplist_json,"
        " created_at = excluded.created_at",
        (user_id, week_start, plan_json, shoplist_json,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_meal_plan(user_id: int) -> dict | None:
    conn = db_connect()
    row = conn.execute(
        "SELECT * FROM meal_plans WHERE user_id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# AI helpers
# ---------------------------------------------------------------------------

def _parse_ai_json(raw: str) -> dict:
    """Strip code fences and parse JSON from AI response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def estimate_calories(meal_text: str) -> dict:
    """Send meal description to Groq and return parsed JSON."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meal_text},
        ],
    )
    return _parse_ai_json(response.choices[0].message.content)


def estimate_calories_from_photo(image_bytes: bytes, caption: str = "") -> dict:
    """Send a meal photo to Groq vision model and return parsed JSON."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
        },
        {
            "type": "text",
            "text": caption if caption else "What food is in this photo? Estimate calories and macros.",
        },
    ]
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return _parse_ai_json(response.choices[0].message.content)


def calculate_recommendations(height_cm: int, weight_kg: float,
                              age: int, gender: str, activity: str) -> dict:
    """Ask AI to recommend daily calories and macros based on profile."""
    user_msg = (
        f"Height: {height_cm}cm, Weight: {weight_kg}kg, "
        f"Age: {age}, Gender: {gender}, Activity level: {activity}"
    )
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {"role": "system", "content": PROFILE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return _parse_ai_json(response.choices[0].message.content)


def scale_macros(base_calories: int, new_calories: int,
                 protein_g: int, fat_g: int, carbs_g: int) -> tuple[int, int, int]:
    """Scale macros proportionally when calorie limit changes."""
    if base_calories <= 0:
        return (protein_g, fat_g, carbs_g)
    ratio = new_calories / base_calories
    return (round(protein_g * ratio), round(fat_g * ratio), round(carbs_g * ratio))


def generate_meal_plan(targets: dict, profile: dict, prefs: dict) -> dict:
    """Generate a 7-day meal plan via AI based on user profile and preferences."""
    schedule_str = ", ".join(
        f"{s['meal']} at {s['time']}" for s in prefs["schedule"]
    )
    excludes_str = ", ".join(prefs["excludes"]) if prefs["excludes"] else "none"
    goal_desc = GOALS.get(prefs["goal"], prefs["goal"])

    user_msg = (
        f"Daily targets: {targets['daily_limit']} kcal, "
        f"P: {targets['daily_protein_g']}g, F: {targets['daily_fat_g']}g, "
        f"C: {targets['daily_carbs_g']}g\n"
        f"Goal: {prefs['goal']} ({goal_desc})\n"
        f"Meal schedule: {schedule_str}\n"
        f"Excluded foods: {excludes_str}\n"
        f"Weekly budget: {prefs['budget_amount']} {prefs['budget_currency']}\n"
        f"Person: {profile['height_cm']}cm, {profile['weight_kg']}kg, "
        f"age {profile['age']}, {profile['gender']}, "
        f"activity: {profile.get('activity', 'moderate')}"
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": MEALPLAN_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return _parse_ai_json(response.choices[0].message.content)


def format_reply(data: dict, today: dict, targets: dict) -> str:
    """Format calorie estimate with daily progress and macros."""
    lines = ["\U0001f37d Calorie Estimate:"]
    for item in data.get("items", []):
        portion = item.get("portion", "")
        portion_str = f" ({portion})" if portion else ""
        raw = item.get("per_100g_raw", "?")
        cooked = item.get("per_100g_cooked", "?")
        p = item.get("protein_g", 0)
        f = item.get("fat_g", 0)
        c = item.get("carbs_g", 0)
        lines.append(
            f"- {item['name']}{portion_str}: ~{item['calories']} kcal"
            f"\n  P: {p}g | F: {f}g | C: {c}g"
            f"\n  per 100g: {raw} raw / {cooked} cooked"
        )

    tp = data.get("total_protein_g", 0)
    tf = data.get("total_fat_g", 0)
    tc = data.get("total_carbs_g", 0)
    lines.append(f"\nTotal: ~{data['total']} kcal | P: {tp}g | F: {tf}g | C: {tc}g")

    note = data.get("note")
    if note:
        lines.append(f"({note})")

    # Daily progress
    cal_limit = targets["daily_limit"]
    cal_today = today["calories"]
    remaining = cal_limit - cal_today
    if remaining >= 0:
        lines.append(
            f"\n\U0001f4ca Today: {cal_today} / {cal_limit} kcal"
            f" ({remaining} remaining)"
        )
    else:
        lines.append(
            f"\n\u26a0\ufe0f Over limit! {cal_today} / {cal_limit} kcal"
            f" (+{abs(remaining)} over)"
        )

    # Macro targets (only if profile is set)
    tp_target = targets["daily_protein_g"]
    tf_target = targets["daily_fat_g"]
    tc_target = targets["daily_carbs_g"]
    if tp_target > 0:
        lines.append(
            f"   P: {today['protein_g']}/{tp_target}g"
            f" | F: {today['fat_g']}/{tf_target}g"
            f" | C: {today['carbs_g']}/{tc_target}g"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a description of what you ate and I'll estimate the calories "
        "and macros.\n\n"
        "Set up your profile first for personalized targets:\n"
        "  /profile 180 75 28 male\n\n"
        "Or just start logging meals. Type /help to see all commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "\U0001f37d HOW TO USE\n\n"
        "Just type what you ate and I'll estimate calories & macros:\n"
        "  chicken salad with rice\n"
        "  2 eggs, toast with butter, black coffee\n"
        "  big mac, medium fries, diet coke\n\n"
        "\U0001f4f7 You can also send a PHOTO of your meal!\n"
        "Add a caption for better accuracy (e.g. 'about 200g of pasta').\n\n"
        "\U0001f464 PROFILE & TARGETS\n\n"
        "/profile <height_cm> <weight_kg> <age> <gender> <activity>\n"
        "  Set your measurements for personalized recommendations\n"
        "  example: /profile 180 75 28 male moderate\n"
        "  example: /profile 165 60 25 female light\n"
        "  Activity: sedentary, light, moderate, active, very_active\n\n"
        "/myprofile - show your profile and daily targets\n\n"
        "/macros - show today's macro progress (P/F/C)\n\n"
        "\U0001f4ca CALORIE TRACKING\n\n"
        "/today - show all meals logged today with totals\n"
        "/week - 7-day calorie and macro summary\n"
        "/setlimit <kcal> - override daily calorie limit\n"
        "  (macros scale proportionally)\n"
        "  example: /setlimit 2200\n"
        "/limit - check your current daily limit\n\n"
        "\U0001f4a7 WATER TRACKING\n\n"
        "/water <ml> - log water (e.g. /water 500)\n"
        "/watertoday - show today's water intake\n"
        "/reminders on|off - toggle water reminders\n"
        "  Schedule: 11:00, 14:00, 16:00, 18:00, 20:00 UTC\n\n"
        "\U0001f957 DIET PLANNING\n\n"
        "/goal <lose_weight|maintain|gain_muscle>\n"
        "  example: /goal lose_weight\n\n"
        "/schedule <meal time>, <meal time>, ...\n"
        "  example: /schedule breakfast 8:00, lunch 13:00, dinner 19:00\n\n"
        "/exclude <food1>, <food2>, ...\n"
        "  example: /exclude pork, shellfish, peanuts\n"
        "  /exclude clear - remove all exclusions\n\n"
        "/budget <amount> <currency>\n"
        "  example: /budget 80 EUR\n\n"
        "/mealplan - generate 7-day meal plan with shopping list\n"
        "/shoplist - re-show last shopping list\n"
        "/diet - show your current diet preferences\n\n"
        "\U0001f504 RESET DATA\n\n"
        "/reset meals - clear today's meals\n"
        "/reset water - clear today's water\n"
        "/reset all - delete ALL your data\n"
        "  Only affects your own data."
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or len(context.args) < 5:
        activity_list = "\n".join(
            f"  {k} - {v}" for k, v in ACTIVITY_LEVELS.items()
        )
        await update.message.reply_text(
            "Usage: /profile <height_cm> <weight_kg> <age> <gender> <activity>\n\n"
            f"Activity levels:\n{activity_list}\n\n"
            "Example: /profile 180 75 28 male moderate"
        )
        return

    try:
        height_cm = int(context.args[0])
        weight_kg = float(context.args[1])
        age = int(context.args[2])
        gender = context.args[3].lower()
        activity = context.args[4].lower()
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Invalid input. Example: /profile 180 75 28 male moderate"
        )
        return

    if not (100 <= height_cm <= 250):
        await update.message.reply_text("Height must be between 100 and 250 cm.")
        return
    if not (30 <= weight_kg <= 300):
        await update.message.reply_text("Weight must be between 30 and 300 kg.")
        return
    if not (10 <= age <= 120):
        await update.message.reply_text("Age must be between 10 and 120.")
        return
    if gender not in ("male", "female"):
        await update.message.reply_text("Gender must be 'male' or 'female'.")
        return
    if activity not in ACTIVITY_LEVELS:
        activity_list = "\n".join(
            f"  {k} - {v}" for k, v in ACTIVITY_LEVELS.items()
        )
        await update.message.reply_text(
            f"Invalid activity level. Choose one:\n{activity_list}"
        )
        return

    await update.message.reply_text("\u2699\ufe0f Calculating your recommendations...")

    try:
        recs = calculate_recommendations(height_cm, weight_kg, age, gender, activity)
        rec_cal = recs["daily_calories"]
        rec_p = recs["daily_protein_g"]
        rec_f = recs["daily_fat_g"]
        rec_c = recs["daily_carbs_g"]

        save_profile(update.effective_user.id, height_cm, weight_kg, age, gender, activity,
                     rec_cal, rec_p, rec_f, rec_c)
        # Set as active targets
        set_targets(update.effective_user.id, rec_cal, rec_p, rec_f, rec_c)

        await update.message.reply_text(
            f"\u2705 Profile saved!\n\n"
            f"Height: {height_cm}cm | Weight: {weight_kg}kg\n"
            f"Age: {age} | Gender: {gender}\n"
            f"Activity: {activity} ({ACTIVITY_LEVELS[activity]})\n\n"
            f"\U0001f3af Daily Recommendations:\n"
            f"Calories: {rec_cal} kcal\n"
            f"Protein: {rec_p}g\n"
            f"Fat: {rec_f}g\n"
            f"Carbs: {rec_c}g\n\n"
            f"These are now your active daily targets.\n"
            f"Use /setlimit to adjust calories (macros will scale)."
        )
    except Exception:
        logger.exception("Error calculating profile recommendations")
        await update.message.reply_text(
            "Something went wrong calculating recommendations. Please try again."
        )


async def myprofile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    profile = get_profile(update.effective_user.id)
    if not profile:
        await update.message.reply_text(
            "No profile set. Use /profile <height_cm> <weight_kg> <age> <gender> <activity>\n"
            "Example: /profile 180 75 28 male moderate"
        )
        return

    activity = profile.get('activity', 'moderate')
    activity_desc = ACTIVITY_LEVELS.get(activity, '')
    targets = get_targets(update.effective_user.id)
    await update.message.reply_text(
        f"\U0001f464 Your Profile:\n"
        f"Height: {profile['height_cm']}cm | Weight: {profile['weight_kg']}kg\n"
        f"Age: {profile['age']} | Gender: {profile['gender']}\n"
        f"Activity: {activity} ({activity_desc})\n\n"
        f"\U0001f3af AI Recommendations:\n"
        f"Calories: {profile['rec_calories']} kcal\n"
        f"P: {profile['rec_protein_g']}g | F: {profile['rec_fat_g']}g"
        f" | C: {profile['rec_carbs_g']}g\n\n"
        f"\U0001f4ca Active Targets:\n"
        f"Calories: {targets['daily_limit']} kcal\n"
        f"P: {targets['daily_protein_g']}g | F: {targets['daily_fat_g']}g"
        f" | C: {targets['daily_carbs_g']}g"
    )


async def macros_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    today = get_today_totals(user_id)
    targets = get_targets(user_id)

    lines = ["\U0001f4ca Today's Macros:"]
    lines.append(
        f"Calories: {today['calories']} / {targets['daily_limit']} kcal"
    )

    if targets["daily_protein_g"] > 0:
        lines.append(
            f"Protein:  {today['protein_g']} / {targets['daily_protein_g']}g"
        )
        lines.append(
            f"Fat:      {today['fat_g']} / {targets['daily_fat_g']}g"
        )
        lines.append(
            f"Carbs:    {today['carbs_g']} / {targets['daily_carbs_g']}g"
        )
    else:
        lines.append(f"Protein:  {today['protein_g']}g")
        lines.append(f"Fat:      {today['fat_g']}g")
        lines.append(f"Carbs:    {today['carbs_g']}g")
        lines.append("\nSet your profile for personalized targets:")
        lines.append("/profile <height_cm> <weight_kg> <age> <gender>")

    await update.message.reply_text("\n".join(lines))


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    meals = get_today_meals(user_id)
    if not meals:
        await update.message.reply_text("No meals logged today. Just type what you ate!")
        return

    targets = get_targets(user_id)
    total_cal = sum(m["calories"] for m in meals)
    total_p = sum(m["protein_g"] for m in meals)
    total_f = sum(m["fat_g"] for m in meals)
    total_c = sum(m["carbs_g"] for m in meals)
    remaining = targets["daily_limit"] - total_cal

    lines = ["\U0001f4cb Today's meals:"]
    for m in meals:
        t = m["created_at"][11:16]
        lines.append(
            f"- [{t}] {m['meal_text'][:40]}: ~{m['calories']} kcal"
            f" (P:{m['protein_g']}g F:{m['fat_g']}g C:{m['carbs_g']}g)"
        )

    lines.append(f"\nTotal: {total_cal} kcal | P: {total_p}g | F: {total_f}g | C: {total_c}g")
    if remaining >= 0:
        lines.append(f"\U0001f4ca {total_cal} / {targets['daily_limit']} kcal ({remaining} remaining)")
    else:
        lines.append(
            f"\u26a0\ufe0f Over limit! {total_cal} / {targets['daily_limit']} kcal"
            f" (+{abs(remaining)} over)"
        )

    if targets["daily_protein_g"] > 0:
        lines.append(
            f"   P: {total_p}/{targets['daily_protein_g']}g"
            f" | F: {total_f}/{targets['daily_fat_g']}g"
            f" | C: {total_c}/{targets['daily_carbs_g']}g"
        )
    await update.message.reply_text("\n".join(lines))


async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    summary = get_week_summary(user_id)
    if not summary:
        await update.message.reply_text("No data for the past 7 days.")
        return

    targets = get_targets(user_id)
    lines = ["\U0001f4c5 Weekly Summary:"]
    for day in summary:
        diff = day["total"] - targets["daily_limit"]
        status = f"+{diff} over" if diff > 0 else f"{abs(diff)} under"
        lines.append(
            f"- {day['day']}: {day['total']} kcal ({status})"
            f" | P:{day['protein_g']}g F:{day['fat_g']}g C:{day['carbs_g']}g"
        )

    week_total = sum(d["total"] for d in summary)
    avg_p = sum(d["protein_g"] for d in summary) // len(summary)
    avg_f = sum(d["fat_g"] for d in summary) // len(summary)
    avg_c = sum(d["carbs_g"] for d in summary) // len(summary)
    lines.append(f"\nWeek total: {week_total} kcal")
    lines.append(f"Daily avg macros: P: {avg_p}g | F: {avg_f}g | C: {avg_c}g")
    await update.message.reply_text("\n".join(lines))


async def setlimit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /setlimit <kcal>\nExample: /setlimit 2200")
        return
    try:
        value = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Please provide a number. Example: /setlimit 2200")
        return
    if value < 500 or value > 10000:
        await update.message.reply_text("Limit must be between 500 and 10000 kcal.")
        return

    user_id = update.effective_user.id
    # Scale macros proportionally based on profile recommendations
    profile = get_profile(user_id)
    if profile and profile["rec_calories"] > 0:
        new_p, new_f, new_c = scale_macros(
            profile["rec_calories"], value,
            profile["rec_protein_g"], profile["rec_fat_g"], profile["rec_carbs_g"],
        )
        set_targets(user_id, value, new_p, new_f, new_c)
        await update.message.reply_text(
            f"Daily limit set to {value} kcal.\n"
            f"Macros scaled: P: {new_p}g | F: {new_f}g | C: {new_c}g"
        )
    else:
        set_targets(user_id, value)
        await update.message.reply_text(f"Daily calorie limit set to {value} kcal.")


async def limit_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    targets = get_targets(update.effective_user.id)
    msg = f"Your daily calorie limit: {targets['daily_limit']} kcal"
    if targets["daily_protein_g"] > 0:
        msg += (
            f"\nMacro targets: P: {targets['daily_protein_g']}g"
            f" | F: {targets['daily_fat_g']}g"
            f" | C: {targets['daily_carbs_g']}g"
        )
    await update.message.reply_text(msg)


async def water_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /water <ml>\nExample: /water 500")
        return
    try:
        amount = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Please provide a number. Example: /water 500")
        return
    if amount <= 0 or amount > 5000:
        await update.message.reply_text("Amount must be between 1 and 5000 ml.")
        return

    user = update.effective_user
    save_water(user.id, user.username or user.first_name, update.effective_chat.id, amount)
    total = get_today_water(user.id)
    remaining = DAILY_WATER_TARGET_ML - total

    if remaining > 0:
        await update.message.reply_text(
            f"\U0001f4a7 Logged {amount}ml.\n"
            f"Today: {total} / {DAILY_WATER_TARGET_ML}ml ({remaining}ml remaining)"
        )
    else:
        await update.message.reply_text(
            f"\U0001f4a7 Logged {amount}ml.\n"
            f"\u2705 Today: {total} / {DAILY_WATER_TARGET_ML}ml - target reached!"
        )


async def watertoday_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    total = get_today_water(update.effective_user.id)
    remaining = DAILY_WATER_TARGET_ML - total
    if remaining > 0:
        await update.message.reply_text(
            f"\U0001f4a7 Water today: {total} / {DAILY_WATER_TARGET_ML}ml"
            f" ({remaining}ml remaining)"
        )
    else:
        await update.message.reply_text(
            f"\u2705 Water today: {total} / {DAILY_WATER_TARGET_ML}ml - target reached!"
        )


async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /reminders on  or  /reminders off")
        return
    action = context.args[0].lower()
    chat_id = update.effective_chat.id
    if action == "on":
        add_reminder_chat(chat_id)
        schedule = "\n".join(
            f"  {h:02d}:{m:02d} UTC - {a}ml" for h, m, a in WATER_SCHEDULE
        )
        await update.message.reply_text(
            f"\U0001f4a7 Water reminders enabled for this chat!\n\nSchedule:\n{schedule}"
        )
    elif action == "off":
        remove_reminder_chat(chat_id)
        await update.message.reply_text("Water reminders disabled for this chat.")
    else:
        await update.message.reply_text("Usage: /reminders on  or  /reminders off")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/reset meals - clear today's meal logs\n"
            "/reset water - clear today's water logs\n"
            "/reset all - delete all your data (meals, water, profile, diet, limits)"
        )
        return

    action = context.args[0].lower()
    user_id = update.effective_user.id

    if action == "meals":
        reset_today_meals(user_id)
        await update.message.reply_text("\u2705 Today's meal logs cleared.")
    elif action == "water":
        reset_today_water(user_id)
        await update.message.reply_text("\u2705 Today's water logs cleared.")
    elif action == "all":
        reset_all_data(user_id)
        await update.message.reply_text(
            "\u2705 All your data has been deleted (meals, water, profile, diet, limits)."
        )
    else:
        await update.message.reply_text(
            "Unknown option. Use: /reset meals, /reset water, or /reset all"
        )


async def goal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        goals_list = "\n".join(f"  {k} - {v}" for k, v in GOALS.items())
        await update.message.reply_text(
            f"Usage: /goal <goal>\n\nGoals:\n{goals_list}\n\n"
            "Example: /goal lose_weight"
        )
        return

    goal = context.args[0].lower()
    if goal not in GOALS:
        goals_list = "\n".join(f"  {k} - {v}" for k, v in GOALS.items())
        await update.message.reply_text(
            f"Invalid goal. Choose one:\n{goals_list}"
        )
        return

    update_diet_pref(update.effective_user.id, goal=goal)
    await update.message.reply_text(
        f"\u2705 Goal set to: {goal} ({GOALS[goal]})"
    )


async def schedule_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /schedule <meal time>, <meal time>, ...\n\n"
            "Example:\n"
            "/schedule breakfast 8:00, lunch 13:00, snack 16:00, dinner 19:00"
        )
        return

    raw = " ".join(context.args)
    slots = []
    for part in raw.split(","):
        tokens = part.strip().split()
        if len(tokens) < 2:
            await update.message.reply_text(
                f"Invalid slot: '{part.strip()}'. Each slot needs a name and time.\n"
                "Example: breakfast 8:00, lunch 13:00, dinner 19:00"
            )
            return
        meal_name = tokens[0].lower()
        meal_time = tokens[1]
        # Basic time validation
        try:
            h, m = meal_time.split(":")
            int(h); int(m)
        except (ValueError, AttributeError):
            await update.message.reply_text(
                f"Invalid time format: '{meal_time}'. Use HH:MM (e.g. 13:00)"
            )
            return
        slots.append({"meal": meal_name, "time": meal_time})

    if len(slots) < 2 or len(slots) > 7:
        await update.message.reply_text("Please specify between 2 and 7 meals.")
        return

    update_diet_pref(update.effective_user.id, schedule=slots)
    schedule_str = "\n".join(f"  {s['meal']} at {s['time']}" for s in slots)
    await update.message.reply_text(
        f"\u2705 Eating schedule set:\n{schedule_str}"
    )


async def exclude_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /exclude <food1>, <food2>, ...\n"
            "Example: /exclude pork, shellfish, peanuts\n\n"
            "To clear exclusions: /exclude clear"
        )
        return

    raw = " ".join(context.args)
    if raw.strip().lower() == "clear":
        update_diet_pref(update.effective_user.id, excludes=[])
        await update.message.reply_text("\u2705 Exclusion list cleared.")
        return

    excludes = [item.strip() for item in raw.split(",") if item.strip()]
    update_diet_pref(update.effective_user.id, excludes=excludes)
    await update.message.reply_text(
        f"\u2705 Excluded foods: {', '.join(excludes)}"
    )


async def budget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /budget <amount> <currency>\n"
            "Example: /budget 80 EUR\n"
            "Example: /budget 5000 RUB"
        )
        return

    try:
        amount = float(context.args[0])
    except ValueError:
        await update.message.reply_text("Amount must be a number. Example: /budget 80 EUR")
        return
    if amount <= 0:
        await update.message.reply_text("Amount must be positive.")
        return

    currency = context.args[1].upper()
    update_diet_pref(update.effective_user.id,
                     budget_amount=amount, budget_currency=currency)
    await update.message.reply_text(
        f"\u2705 Weekly budget set to {amount} {currency}"
    )


async def diet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current diet preferences at a glance."""
    prefs = get_diet_prefs(update.effective_user.id)
    if not prefs:
        await update.message.reply_text(
            "No diet preferences set yet. Use these commands:\n"
            "/goal <lose_weight|maintain|gain_muscle>\n"
            "/schedule breakfast 8:00, lunch 13:00, dinner 19:00\n"
            "/exclude pork, shellfish\n"
            "/budget 80 EUR"
        )
        return

    goal_desc = GOALS.get(prefs["goal"], prefs["goal"])
    schedule_str = ", ".join(
        f"{s['meal']} {s['time']}" for s in prefs["schedule"]
    ) if prefs["schedule"] else "not set"
    excludes_str = ", ".join(prefs["excludes"]) if prefs["excludes"] else "none"
    budget_str = (
        f"{prefs['budget_amount']} {prefs['budget_currency']}"
        if prefs["budget_amount"] > 0 else "not set"
    )

    await update.message.reply_text(
        f"\U0001f957 Diet Preferences:\n\n"
        f"Goal: {prefs['goal']} ({goal_desc})\n"
        f"Schedule: {schedule_str}\n"
        f"Excluded: {excludes_str}\n"
        f"Weekly budget: {budget_str}"
    )


async def mealplan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a 7-day meal plan."""
    user_id = update.effective_user.id

    # Check prerequisites
    profile = get_profile(user_id)
    if not profile:
        await update.message.reply_text(
            "Please set your profile first:\n"
            "/profile <height_cm> <weight_kg> <age> <gender> <activity>"
        )
        return

    prefs = get_diet_prefs(user_id)
    missing = []
    if not prefs or prefs["goal"] == "maintain" and not prefs["schedule"]:
        missing.append("/goal <lose_weight|maintain|gain_muscle>")
    if not prefs or not prefs["schedule"]:
        missing.append("/schedule breakfast 8:00, lunch 13:00, dinner 19:00")
    if not prefs or prefs["budget_amount"] <= 0:
        missing.append("/budget <amount> <currency>")

    if missing:
        await update.message.reply_text(
            "Please set these preferences first:\n" + "\n".join(missing)
        )
        return

    targets = get_targets(user_id)
    if targets["daily_protein_g"] == 0:
        await update.message.reply_text(
            "Your macro targets are not set. Run /profile to get personalized recommendations."
        )
        return

    await update.message.reply_text(
        "\u2699\ufe0f Generating your 7-day meal plan... This may take a moment."
    )

    try:
        plan = generate_meal_plan(targets, profile, prefs)

        # Save the plan
        today = datetime.now(timezone.utc)
        # Find next Monday (or today if Monday)
        days_until_monday = (7 - today.weekday()) % 7
        if days_until_monday == 0 and today.hour >= 12:
            days_until_monday = 7
        week_start = (today + timedelta(days=days_until_monday)).strftime("%Y-%m-%d")

        shoplist = plan.get("shopping_list", [])
        save_meal_plan(
            user_id, week_start,
            json.dumps(plan), json.dumps(shoplist),
        )

        # Send day-by-day messages
        for day_data in plan.get("days", []):
            lines = [f"\U0001f4c5 {day_data['day']}"]
            for meal in day_data.get("meals", []):
                slot = meal.get("slot", "")
                t = meal.get("time", "")
                lines.append(
                    f"\n\U0001f37d {slot.capitalize()} ({t})\n"
                    f"{meal.get('name', '')}\n"
                    f"~{meal.get('calories', 0)} kcal | "
                    f"P: {meal.get('protein_g', 0)}g | "
                    f"F: {meal.get('fat_g', 0)}g | "
                    f"C: {meal.get('carbs_g', 0)}g"
                )
            dt = day_data.get("day_total", {})
            lines.append(
                f"\nDay total: {dt.get('calories', 0)} kcal | "
                f"P: {dt.get('protein_g', 0)}g | "
                f"F: {dt.get('fat_g', 0)}g | "
                f"C: {dt.get('carbs_g', 0)}g"
            )
            await update.message.reply_text("\n".join(lines))

        # Send shopping list
        currency = plan.get("currency", prefs["budget_currency"])
        total_cost = plan.get("total_cost", 0)
        budget = prefs["budget_amount"]
        diff = budget - total_cost

        shop_lines = [
            f"\U0001f6d2 Weekly Shopping List "
            f"(Budget: {budget} {currency})\n"
        ]
        for item in shoplist:
            shop_lines.append(
                f"- {item.get('item', '')} {item.get('quantity', '')}: "
                f"~{item.get('estimated_cost', 0)} {currency}"
            )
        shop_lines.append(f"\nEstimated total: ~{total_cost} {currency}")
        if diff >= 0:
            shop_lines.append(f"({diff:.0f} {currency} under budget)")
        else:
            shop_lines.append(f"\u26a0\ufe0f ({abs(diff):.0f} {currency} over budget)")

        await update.message.reply_text("\n".join(shop_lines))

    except json.JSONDecodeError:
        logger.exception("Failed to parse meal plan JSON")
        await update.message.reply_text(
            "Sorry, I couldn't generate a valid meal plan. Please try again."
        )
    except Exception:
        logger.exception("Error generating meal plan")
        await update.message.reply_text(
            "Something went wrong generating the meal plan. Please try again."
        )


async def shoplist_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-show the shopping list from the last generated plan."""
    plan_row = get_meal_plan(update.effective_user.id)
    if not plan_row:
        await update.message.reply_text(
            "No meal plan generated yet. Use /mealplan to create one."
        )
        return

    prefs = get_diet_prefs(update.effective_user.id)
    shoplist = json.loads(plan_row["shoplist_json"])
    plan = json.loads(plan_row["plan_json"])

    currency = plan.get("currency", prefs["budget_currency"] if prefs else "EUR")
    total_cost = plan.get("total_cost", 0)
    budget = prefs["budget_amount"] if prefs else 0

    shop_lines = [
        f"\U0001f6d2 Weekly Shopping List"
        f" (Week of {plan_row['week_start']})\n"
    ]
    for item in shoplist:
        shop_lines.append(
            f"- {item.get('item', '')} {item.get('quantity', '')}: "
            f"~{item.get('estimated_cost', 0)} {currency}"
        )
    shop_lines.append(f"\nEstimated total: ~{total_cost} {currency}")
    if budget > 0:
        diff = budget - total_cost
        if diff >= 0:
            shop_lines.append(f"({diff:.0f} {currency} under budget)")
        else:
            shop_lines.append(f"\u26a0\ufe0f ({abs(diff):.0f} {currency} over budget)")

    await update.message.reply_text("\n".join(shop_lines))


# ---------------------------------------------------------------------------
# Meal & photo message handlers
# ---------------------------------------------------------------------------

async def handle_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message as a meal description."""
    meal_text = update.message.text
    if not meal_text:
        return

    user = update.effective_user
    try:
        data = estimate_calories(meal_text)
        calories = data.get("total", 0)
        protein = data.get("total_protein_g", 0)
        fat = data.get("total_fat_g", 0)
        carbs = data.get("total_carbs_g", 0)
        save_meal(user.id, user.username or user.first_name,
                  update.effective_chat.id, meal_text,
                  calories, protein, fat, carbs)
        today = get_today_totals(user.id)
        targets = get_targets(user.id)
        reply = format_reply(data, today, targets)
    except json.JSONDecodeError:
        logger.exception("Failed to parse Groq response")
        reply = "Sorry, I couldn't parse the calorie data. Try rephrasing your meal."
    except Exception:
        logger.exception("Error estimating calories")
        reply = "Something went wrong. Please try again in a moment."

    await update.message.reply_text(reply)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle a photo message as a meal photo."""
    photo = update.message.photo[-1]
    caption = update.message.caption or ""

    user = update.effective_user
    try:
        file = await photo.get_file()
        image_bytes = await file.download_as_bytearray()
        data = estimate_calories_from_photo(bytes(image_bytes), caption)
        calories = data.get("total", 0)
        protein = data.get("total_protein_g", 0)
        fat = data.get("total_fat_g", 0)
        carbs = data.get("total_carbs_g", 0)
        meal_desc = caption if caption else "meal (photo)"
        save_meal(user.id, user.username or user.first_name,
                  update.effective_chat.id, meal_desc,
                  calories, protein, fat, carbs)
        today = get_today_totals(user.id)
        targets = get_targets(user.id)
        reply = format_reply(data, today, targets)
    except json.JSONDecodeError:
        logger.exception("Failed to parse Groq vision response")
        reply = "Sorry, I couldn't identify the food in this photo. Try adding a caption describing the meal."
    except Exception:
        logger.exception("Error estimating calories from photo")
        reply = "Something went wrong analyzing the photo. Please try again."

    await update.message.reply_text(reply)


# ---------------------------------------------------------------------------
# Water reminder job
# ---------------------------------------------------------------------------

async def send_water_reminder(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scheduled job: send water reminder to all registered chats."""
    amount = context.job.data
    chat_ids = get_reminder_chats()
    for chat_id in chat_ids:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"\U0001f4a7 Water Reminder!\n"
                    f"Time to drink ~{amount}ml of water.\n"
                    f"Use /water {amount} to log it."
                ),
            )
        except Exception:
            logger.exception("Failed to send water reminder to chat %s", chat_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    use_polling = "--poll" in sys.argv

    db_init()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Calorie & profile commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("profile", profile_command))
    app.add_handler(CommandHandler("myprofile", myprofile_command))
    app.add_handler(CommandHandler("macros", macros_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("week", week_command))
    app.add_handler(CommandHandler("setlimit", setlimit_command))
    app.add_handler(CommandHandler("limit", limit_command))

    # Water commands
    app.add_handler(CommandHandler("water", water_command))
    app.add_handler(CommandHandler("watertoday", watertoday_command))
    app.add_handler(CommandHandler("reminders", reminders_command))

    # Reset command
    app.add_handler(CommandHandler("reset", reset_command))

    # Diet planning commands
    app.add_handler(CommandHandler("goal", goal_command))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("exclude", exclude_command))
    app.add_handler(CommandHandler("budget", budget_command))
    app.add_handler(CommandHandler("mealplan", mealplan_command))
    app.add_handler(CommandHandler("shoplist", shoplist_command))
    app.add_handler(CommandHandler("diet", diet_command))

    # Meal & photo handlers (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_meal))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Schedule water reminders
    for hour, minute, amount in WATER_SCHEDULE:
        app.job_queue.run_daily(
            send_water_reminder,
            time=time(hour=hour, minute=minute, tzinfo=timezone.utc),
            data=amount,
            name=f"water_{hour:02d}{minute:02d}",
        )

    if use_polling:
        logger.info("Starting bot in polling mode (local dev)")
        app.run_polling(drop_pending_updates=True)
    else:
        if not WEBHOOK_URL:
            logger.error("WEBHOOK_URL is required for webhook mode")
            sys.exit(1)

        webhook_path = "/webhook"
        full_url = f"{WEBHOOK_URL}{webhook_path}"
        logger.info("Starting bot with webhook at %s", full_url)

        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=webhook_path,
            webhook_url=full_url,
            drop_pending_updates=True,
        )


if __name__ == "__main__":
    main()
