import io
import json
import logging
import os
import sys
import base64
from datetime import datetime, timedelta, time, timezone

from groq import Groq
import psycopg2
import psycopg2.extras
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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

DATABASE_URL = os.environ["DATABASE_URL"]
DEFAULT_CALORIE_LIMIT = 1800
DAILY_WATER_TARGET_ML = 2000

groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a precision nutrition assistant. Estimate calories and macronutrients "
    "for each item in a meal description, using USDA FoodData Central values as "
    "your primary reference for standard foods. "
    "Return ONLY valid JSON with no extra text.\n\n"
    "PORTION RULES:\n"
    "- Restaurant/cafe meals: assume portions are 20-30% larger than home-cooked "
    "(restaurants use more oil, butter, and larger serving sizes).\n"
    "- Named fast-food/chain items (Big Mac, Chipotle burrito, etc.): use the "
    "brand's published nutritional data (Big Mac = 550 kcal, medium fries = 320 kcal).\n"
    "- If no weight is specified, assume a realistic portion and state it in 'portion'.\n\n"
    "COOKING METHOD ADJUSTMENTS:\n"
    "- Fried: add 10-20% calories vs boiled/grilled due to oil absorption.\n"
    "- Boiled/steamed: no fat addition. Grilled: minimal fat. Raw: use raw reference.\n\n"
    "JSON schema (return ONLY this, no markdown fences):\n"
    '{"items": [{"name": "<food>", "portion": "<weight or description>", '
    '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>, '
    '"fiber_g": <int>, "per_100g_raw": <int>, "per_100g_cooked": <int>, '
    '"confidence": "<high|medium|low>", '
    '"context_hint": "<home_cooked|restaurant|branded|unknown>"}], '
    '"total": <int>, "total_protein_g": <int>, "total_fat_g": <int>, '
    '"total_carbs_g": <int>, "total_fiber_g": <int>}\n\n'
    "confidence: 'high'=standard food known weight; 'medium'=reasonable estimate; "
    "'low'=ambiguous. Add 'note' field for unclear descriptions.\n\n"
    "Example: 2 scrambled eggs, 1 toast with butter\n"
    '{"items": [{"name": "Scrambled eggs", "portion": "2 large (100g)", '
    '"calories": 182, "protein_g": 12, "fat_g": 14, "carbs_g": 2, "fiber_g": 0, '
    '"per_100g_raw": 143, "per_100g_cooked": 182, '
    '"confidence": "high", "context_hint": "home_cooked"}, '
    '{"name": "Toast with butter", "portion": "1 slice (40g bread + 10g butter)", '
    '"calories": 178, "protein_g": 4, "fat_g": 9, "carbs_g": 21, "fiber_g": 2, '
    '"per_100g_raw": 265, "per_100g_cooked": 265, '
    '"confidence": "high", "context_hint": "home_cooked"}], '
    '"total": 360, "total_protein_g": 16, "total_fat_g": 23, '
    '"total_carbs_g": 23, "total_fiber_g": 2}'
)

VISION_PROMPT = (
    "You are a precision nutrition analyst specializing in food photo analysis. "
    "Return ONLY valid JSON with no extra text.\n\n"
    "ANALYSIS APPROACH:\n"
    "1. IDENTIFY every food item visible including garnishes, sauces, visible oils.\n"
    "2. SCALE using reference objects — standard dinner plate: 26-28 cm diameter; "
    "fork: ~19 cm; tablespoon: ~15 ml. If no reference is visible, use defaults below.\n"
    "3. DEPTH: account for volume, not just surface. A bowl may hold 50% more than it looks.\n"
    "4. COOKING METHOD from visual cues: oil sheen = fried; char marks = grilled; "
    "apply appropriate calorie adjustments.\n"
    "5. CONTEXT: home-cooked (smaller/irregular) vs restaurant (larger/garnished) vs branded.\n\n"
    "PORTION DEFAULTS when no scale reference is visible:\n"
    "- Full dinner plate: 400-600g total | Side plate: 150-300g | Large bowl: 300-500ml\n"
    "- Restaurant plate: 20-30% larger than home-cooked equivalent\n\n"
    "JSON schema (return ONLY this, no markdown fences):\n"
    '{"items": [{"name": "<food>", "portion": "<estimated weight or volume>", '
    '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>, '
    '"fiber_g": <int>, "per_100g_raw": <int>, "per_100g_cooked": <int>, '
    '"confidence": "<high|medium|low>", '
    '"context_hint": "<home_cooked|restaurant|branded|unknown>"}], '
    '"total": <int>, "total_protein_g": <int>, "total_fat_g": <int>, '
    '"total_carbs_g": <int>, "total_fiber_g": <int>}\n\n'
    "confidence: 'high'=item clear and portion estimable; 'medium'=partially obscured; "
    "'low'=unclear food or heavily mixed dish. Add 'note' for unclear items."
)

# PROFILE_PROMPT removed — replaced by pure Python BMR calculation in calculate_recommendations()

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
    "CRITICAL RULES:\n"
    "- Each day MUST hit the daily calorie target within 5% (e.g. if target is "
    "2500 kcal, each day must be 2375-2625 kcal). This is the most important rule.\n"
    "- Each day MUST hit the protein target within 10%.\n"
    "- Distribute calories and macros across all meal slots in the schedule.\n"
    "- Respect the excluded foods strictly - never include them.\n"
    "- Vary meals across the week - do not repeat the same meal on consecutive days.\n"
    "- Use commonly available ingredients.\n"
    "- Keep the total weekly shopping cost within the specified budget.\n"
    "- Consolidate the shopping list: combine quantities, no duplicate items.\n"
    "- Estimate realistic local prices in the specified currency.\n"
    "- For lose_weight goal: prioritize high-protein, high-fiber, lower-calorie meals.\n"
    "- For gain_muscle goal: prioritize high-protein, calorie-dense meals.\n"
    "- All 7 days (Monday through Sunday) MUST be included.\n\n"
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

# Allowlist for diet_prefs columns (prevents SQL injection in update_diet_pref)
DIET_PREF_COLUMNS = frozenset({"goal", "schedule", "excludes", "budget_amount", "budget_currency"})


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def db_connect():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


def db_query(sql, params=(), fetch=None):
    """Execute a query and optionally fetch results. Returns rows as dicts."""
    conn = db_connect()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(sql, params)
        if fetch == "one":
            result = cur.fetchone()
        elif fetch == "all":
            result = cur.fetchall()
        else:
            result = None
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()
    tables = [
        """CREATE TABLE IF NOT EXISTS meals (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            username TEXT,
            chat_id BIGINT,
            meal_text TEXT,
            calories INTEGER NOT NULL,
            protein_g INTEGER NOT NULL DEFAULT 0,
            fat_g INTEGER NOT NULL DEFAULT 0,
            carbs_g INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS limits (
            user_id BIGINT PRIMARY KEY,
            daily_limit INTEGER NOT NULL DEFAULT 1800,
            daily_protein_g INTEGER NOT NULL DEFAULT 0,
            daily_fat_g INTEGER NOT NULL DEFAULT 0,
            daily_carbs_g INTEGER NOT NULL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS profiles (
            user_id BIGINT PRIMARY KEY,
            height_cm INTEGER NOT NULL,
            weight_kg REAL NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            activity TEXT NOT NULL DEFAULT 'moderate',
            rec_calories INTEGER NOT NULL DEFAULT 0,
            rec_protein_g INTEGER NOT NULL DEFAULT 0,
            rec_fat_g INTEGER NOT NULL DEFAULT 0,
            rec_carbs_g INTEGER NOT NULL DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS water (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            username TEXT,
            chat_id BIGINT,
            amount_ml INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS reminder_chats (
            chat_id BIGINT PRIMARY KEY
        )""",
        """CREATE TABLE IF NOT EXISTS diet_prefs (
            user_id BIGINT PRIMARY KEY,
            goal TEXT NOT NULL DEFAULT 'maintain',
            schedule TEXT NOT NULL DEFAULT '[]',
            excludes TEXT NOT NULL DEFAULT '[]',
            budget_amount REAL NOT NULL DEFAULT 0,
            budget_currency TEXT NOT NULL DEFAULT 'EUR'
        )""",
        """CREATE TABLE IF NOT EXISTS meal_plans (
            user_id BIGINT PRIMARY KEY,
            week_start TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            shoplist_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS weight_history (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            weight_kg REAL NOT NULL,
            recorded_at TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_weight_history_user
            ON weight_history(user_id, recorded_at DESC)""",
    ]
    for sql in tables:
        cur.execute(sql)
    # Add body_fat_pct column if it doesn't exist yet (safe migration)
    cur.execute(
        "ALTER TABLE profiles ADD COLUMN IF NOT EXISTS body_fat_pct REAL"
    )
    conn.commit()
    cur.close()
    conn.close()


def save_meal(user_id: int, username: str, chat_id: int,
              meal_text: str, calories: int,
              protein_g: int = 0, fat_g: int = 0, carbs_g: int = 0) -> None:
    db_query(
        "INSERT INTO meals (user_id, username, chat_id, meal_text, calories,"
        " protein_g, fat_g, carbs_g, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, username, chat_id, meal_text, calories,
         protein_g, fat_g, carbs_g, datetime.now(timezone.utc).isoformat()),
    )


def get_today_meals(user_id: int) -> list[dict]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = db_query(
        "SELECT id, meal_text, calories, protein_g, fat_g, carbs_g, created_at"
        " FROM meals WHERE user_id = %s AND created_at LIKE %s ORDER BY created_at",
        (user_id, f"{today}%"), fetch="all",
    )
    return [dict(r) for r in rows]


def get_meal_by_id(meal_id: int, user_id: int) -> dict | None:
    row = db_query(
        "SELECT id, meal_text, calories, protein_g, fat_g, carbs_g, created_at"
        " FROM meals WHERE id = %s AND user_id = %s",
        (meal_id, user_id), fetch="one",
    )
    return dict(row) if row else None


def delete_meal_by_id(meal_id: int, user_id: int) -> bool:
    """Delete a specific meal. Returns True if a row was deleted."""
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM meals WHERE id = %s AND user_id = %s",
            (meal_id, user_id),
        )
        deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def get_meals_for_date(user_id: int, date_str: str) -> list[dict]:
    rows = db_query(
        "SELECT id, meal_text, calories, protein_g, fat_g, carbs_g, created_at"
        " FROM meals WHERE user_id = %s AND created_at LIKE %s ORDER BY created_at",
        (user_id, f"{date_str}%"), fetch="all",
    )
    return [dict(r) for r in rows]


def get_today_totals(user_id: int) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = db_query(
        "SELECT COALESCE(SUM(calories), 0) AS calories,"
        " COALESCE(SUM(protein_g), 0) AS protein_g,"
        " COALESCE(SUM(fat_g), 0) AS fat_g,"
        " COALESCE(SUM(carbs_g), 0) AS carbs_g"
        " FROM meals WHERE user_id = %s AND created_at LIKE %s",
        (user_id, f"{today}%"), fetch="one",
    )
    return dict(row)


def get_week_summary(user_id: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=6)).strftime("%Y-%m-%d")
    rows = db_query(
        "SELECT SUBSTRING(created_at, 1, 10) AS day, SUM(calories) AS total,"
        " SUM(protein_g) AS protein_g, SUM(fat_g) AS fat_g,"
        " SUM(carbs_g) AS carbs_g"
        " FROM meals WHERE user_id = %s AND created_at >= %s"
        " GROUP BY day ORDER BY day",
        (user_id, since), fetch="all",
    )
    return [dict(r) for r in rows]


def get_logged_dates(user_id: int, days: int = 30) -> list[str]:
    """Return distinct dates (YYYY-MM-DD) where meals were logged, last N days."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = db_query(
        "SELECT DISTINCT SUBSTRING(created_at, 1, 10) AS day"
        " FROM meals WHERE user_id = %s AND created_at >= %s ORDER BY day DESC",
        (user_id, since), fetch="all",
    )
    return [r["day"] for r in rows]


def get_daily_calorie_avg(user_id: int, days: int = 30) -> float:
    """Return average daily calories over the last N days (only days with logs)."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    row = db_query(
        "SELECT AVG(day_total) AS avg_cal FROM ("
        "  SELECT SUM(calories) AS day_total"
        "  FROM meals WHERE user_id = %s AND created_at >= %s"
        "  GROUP BY SUBSTRING(created_at, 1, 10)"
        ") sub",
        (user_id, since), fetch="one",
    )
    return float(row["avg_cal"] or 0)


def get_water_logged_dates(user_id: int, days: int = 30) -> list[str]:
    """Return distinct dates where any water was logged, last N days."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = db_query(
        "SELECT DISTINCT SUBSTRING(created_at, 1, 10) AS day"
        " FROM water WHERE user_id = %s AND created_at >= %s ORDER BY day DESC",
        (user_id, since), fetch="all",
    )
    return [r["day"] for r in rows]


def get_targets(user_id: int) -> dict:
    row = db_query(
        "SELECT daily_limit, daily_protein_g, daily_fat_g, daily_carbs_g"
        " FROM limits WHERE user_id = %s", (user_id,), fetch="one",
    )
    if row:
        return dict(row)
    return {
        "daily_limit": DEFAULT_CALORIE_LIMIT,
        "daily_protein_g": 0, "daily_fat_g": 0, "daily_carbs_g": 0,
    }


def set_targets(user_id: int, calories: int,
                protein_g: int = 0, fat_g: int = 0, carbs_g: int = 0) -> None:
    db_query(
        "INSERT INTO limits (user_id, daily_limit, daily_protein_g, daily_fat_g,"
        " daily_carbs_g) VALUES (%s,%s,%s,%s,%s)"
        " ON CONFLICT(user_id) DO UPDATE SET daily_limit = EXCLUDED.daily_limit,"
        " daily_protein_g = EXCLUDED.daily_protein_g,"
        " daily_fat_g = EXCLUDED.daily_fat_g,"
        " daily_carbs_g = EXCLUDED.daily_carbs_g",
        (user_id, calories, protein_g, fat_g, carbs_g),
    )


def save_profile(user_id: int, height_cm: int, weight_kg: float,
                 age: int, gender: str, activity: str, rec_calories: int,
                 rec_protein_g: int, rec_fat_g: int, rec_carbs_g: int,
                 body_fat_pct: float | None = None) -> None:
    db_query(
        "INSERT INTO profiles (user_id, height_cm, weight_kg, age, gender, activity,"
        " rec_calories, rec_protein_g, rec_fat_g, rec_carbs_g, body_fat_pct)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        " ON CONFLICT(user_id) DO UPDATE SET height_cm = EXCLUDED.height_cm,"
        " weight_kg = EXCLUDED.weight_kg, age = EXCLUDED.age,"
        " gender = EXCLUDED.gender, activity = EXCLUDED.activity,"
        " rec_calories = EXCLUDED.rec_calories,"
        " rec_protein_g = EXCLUDED.rec_protein_g,"
        " rec_fat_g = EXCLUDED.rec_fat_g, rec_carbs_g = EXCLUDED.rec_carbs_g,"
        " body_fat_pct = EXCLUDED.body_fat_pct",
        (user_id, height_cm, weight_kg, age, gender, activity,
         rec_calories, rec_protein_g, rec_fat_g, rec_carbs_g, body_fat_pct),
    )


def get_profile(user_id: int) -> dict | None:
    row = db_query(
        "SELECT * FROM profiles WHERE user_id = %s", (user_id,), fetch="one",
    )
    return dict(row) if row else None


def get_water_target(user_id: int) -> int:
    """Return daily water target in ml. Weight-based if profile available (~35ml/kg)."""
    profile = get_profile(user_id)
    if profile and profile.get("weight_kg"):
        return min(int(profile["weight_kg"] * 35), 3500)
    return DAILY_WATER_TARGET_ML


def update_profile_weight(user_id: int, weight_kg: float) -> bool:
    """Update weight in profiles table. Returns False if no profile exists."""
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE profiles SET weight_kg = %s WHERE user_id = %s",
            (weight_kg, user_id),
        )
        updated = cur.rowcount > 0
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def save_weight_history(user_id: int, weight_kg: float) -> None:
    db_query(
        "INSERT INTO weight_history (user_id, weight_kg, recorded_at)"
        " VALUES (%s, %s, %s)",
        (user_id, weight_kg, datetime.now(timezone.utc).isoformat()),
    )


def get_weight_history(user_id: int, limit: int = 10) -> list[dict]:
    rows = db_query(
        "SELECT weight_kg, recorded_at FROM weight_history"
        " WHERE user_id = %s ORDER BY recorded_at DESC LIMIT %s",
        (user_id, limit), fetch="all",
    )
    return [dict(r) for r in rows]


def save_water(user_id: int, username: str, chat_id: int, amount_ml: int) -> None:
    db_query(
        "INSERT INTO water (user_id, username, chat_id, amount_ml, created_at)"
        " VALUES (%s,%s,%s,%s,%s)",
        (user_id, username, chat_id, amount_ml,
         datetime.now(timezone.utc).isoformat()),
    )


def get_today_water(user_id: int) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = db_query(
        "SELECT COALESCE(SUM(amount_ml), 0) AS total FROM water"
        " WHERE user_id = %s AND created_at LIKE %s",
        (user_id, f"{today}%"), fetch="one",
    )
    return row["total"]


def add_reminder_chat(chat_id: int) -> None:
    db_query(
        "INSERT INTO reminder_chats (chat_id) VALUES (%s)"
        " ON CONFLICT DO NOTHING", (chat_id,),
    )


def remove_reminder_chat(chat_id: int) -> None:
    db_query("DELETE FROM reminder_chats WHERE chat_id = %s", (chat_id,))


def get_reminder_chats() -> list[int]:
    rows = db_query("SELECT chat_id FROM reminder_chats", fetch="all")
    return [r["chat_id"] for r in rows]


def reset_today_meals(user_id: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db_query(
        "DELETE FROM meals WHERE user_id = %s AND created_at LIKE %s",
        (user_id, f"{today}%"),
    )


def reset_today_water(user_id: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db_query(
        "DELETE FROM water WHERE user_id = %s AND created_at LIKE %s",
        (user_id, f"{today}%"),
    )


def reset_all_data(user_id: int) -> None:
    conn = db_connect()
    cur = conn.cursor()
    for table in ("meals", "water", "limits", "profiles", "diet_prefs", "meal_plans", "weight_history"):
        cur.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
    conn.commit()
    cur.close()
    conn.close()


def save_diet_prefs(user_id: int, goal: str, schedule: list,
                    excludes: list, budget_amount: float,
                    budget_currency: str) -> None:
    db_query(
        "INSERT INTO diet_prefs (user_id, goal, schedule, excludes,"
        " budget_amount, budget_currency) VALUES (%s,%s,%s,%s,%s,%s)"
        " ON CONFLICT(user_id) DO UPDATE SET goal = EXCLUDED.goal,"
        " schedule = EXCLUDED.schedule, excludes = EXCLUDED.excludes,"
        " budget_amount = EXCLUDED.budget_amount,"
        " budget_currency = EXCLUDED.budget_currency",
        (user_id, goal, json.dumps(schedule), json.dumps(excludes),
         budget_amount, budget_currency),
    )


def update_diet_pref(user_id: int, **kwargs) -> None:
    for key in kwargs:
        if key not in DIET_PREF_COLUMNS:
            raise ValueError(f"Invalid diet_pref column: {key!r}")
    conn = db_connect()
    cur = conn.cursor()
    try:
        # Ensure row exists
        cur.execute(
            "INSERT INTO diet_prefs (user_id, goal, schedule, excludes,"
            " budget_amount, budget_currency) VALUES (%s, 'maintain', '[]', '[]', 0, 'EUR')"
            " ON CONFLICT DO NOTHING",
            (user_id,),
        )
        for key, value in kwargs.items():
            if key in ("schedule", "excludes"):
                value = json.dumps(value)
            cur.execute(f"UPDATE diet_prefs SET {key} = %s WHERE user_id = %s",
                        (value, user_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def get_diet_prefs(user_id: int) -> dict | None:
    row = db_query(
        "SELECT * FROM diet_prefs WHERE user_id = %s", (user_id,), fetch="one",
    )
    if not row:
        return None
    d = dict(row)
    d["schedule"] = json.loads(d["schedule"])
    d["excludes"] = json.loads(d["excludes"])
    return d


def save_meal_plan(user_id: int, week_start: str,
                   plan_json: str, shoplist_json: str) -> None:
    db_query(
        "INSERT INTO meal_plans (user_id, week_start, plan_json, shoplist_json,"
        " created_at) VALUES (%s,%s,%s,%s,%s)"
        " ON CONFLICT(user_id) DO UPDATE SET week_start = EXCLUDED.week_start,"
        " plan_json = EXCLUDED.plan_json, shoplist_json = EXCLUDED.shoplist_json,"
        " created_at = EXCLUDED.created_at",
        (user_id, week_start, plan_json, shoplist_json,
         datetime.now(timezone.utc).isoformat()),
    )


def get_meal_plan(user_id: int) -> dict | None:
    row = db_query(
        "SELECT * FROM meal_plans WHERE user_id = %s", (user_id,), fetch="one",
    )
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


def _validate_meal_plan_macros(plan: dict, targets: dict) -> list[str]:
    """Check each day's totals against calorie (±5%) and protein (±10%) targets.
    Returns a list of warning strings; empty = all days pass."""
    cal_target = targets["daily_limit"]
    prot_target = targets["daily_protein_g"]
    warnings = []
    for day in plan.get("days", []):
        day_name = day.get("day", "?")
        dt = day.get("day_total", {})
        cal = dt.get("calories", 0)
        prot = dt.get("protein_g", 0)
        if not (cal_target * 0.95 <= cal <= cal_target * 1.05):
            pct = round((cal / cal_target - 1) * 100, 1) if cal_target else 0
            sign = "+" if pct > 0 else ""
            warnings.append(f"{day_name}: {cal} kcal ({sign}{pct}% vs {cal_target})")
        if prot_target > 0 and not (prot_target * 0.90 <= prot <= prot_target * 1.10):
            pct = round((prot / prot_target - 1) * 100, 1) if prot_target else 0
            sign = "+" if pct > 0 else ""
            warnings.append(f"{day_name}: {prot}g protein ({sign}{pct}% vs {prot_target}g)")
    return warnings


def _progress_bar(current: int, target: int, width: int = 10) -> str:
    """ASCII progress bar. e.g. [████████░░] 80%"""
    if target <= 0:
        return ""
    pct = min(current / target, 1.0)
    filled = round(pct * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {round(pct * 100)}%"


_CONFIDENCE_ICONS = {"high": "\u2705", "medium": "\U0001f7e1", "low": "\u26a0\ufe0f"}


def _confidence_icon(confidence: str) -> str:
    return _CONFIDENCE_ICONS.get(confidence, "")


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


MAX_IMAGE_BYTES = 800_000   # 800 KB — safe margin under Groq token limits
MAX_IMAGE_DIM = 1024        # max pixels on longest side


def resize_image_if_needed(image_bytes: bytes) -> bytes:
    """Resize image to stay within Groq's token limits before base64 encoding."""
    if len(image_bytes) <= MAX_IMAGE_BYTES:
        return image_bytes
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return buf.getvalue()
    return buf.getvalue()


def estimate_calories_from_photo(image_bytes: bytes, caption: str = "") -> dict:
    """Send a meal photo to Groq vision model and return parsed JSON."""
    image_bytes = resize_image_if_needed(image_bytes)
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


_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
    "active": 1.725, "very_active": 1.9,
}


def calculate_recommendations(
    height_cm: int, weight_kg: float, age: int, gender: str, activity: str,
    body_fat_pct: float | None = None,
) -> dict:
    """
    Compute TDEE using multiple BMR formulas (OmniCalculator methodology).
    Averages all available formulas for a balanced recommendation. No API call.

    Formulas:
      - Mifflin-St Jeor (best for general population)
      - Harris-Benedict revised (Roza & Shizgal 1984)
      - Katch-McArdle (most accurate when body fat % is known)
    """
    multiplier = _ACTIVITY_MULTIPLIERS.get(activity, 1.55)

    # Formula 1: Mifflin-St Jeor
    if gender == "male":
        bmr_mifflin = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr_mifflin = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # Formula 2: Harris-Benedict revised (Roza & Shizgal, 1984)
    if gender == "male":
        bmr_harris = 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age + 88.362
    else:
        bmr_harris = 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age + 447.593

    bmr_values = [bmr_mifflin, bmr_harris]

    # Formula 3: Katch-McArdle (requires body fat %)
    bmr_katch = None
    if body_fat_pct is not None and 5 <= body_fat_pct <= 50:
        lbm = weight_kg * (1 - body_fat_pct / 100)
        bmr_katch = 370 + 21.6 * lbm
        bmr_values.append(bmr_katch)

    bmr_avg = sum(bmr_values) / len(bmr_values)
    daily_calories = round(bmr_avg * multiplier)

    # Macros: 1.8 g/kg protein, 27.5% fat, rest carbs
    daily_protein_g = round(weight_kg * 1.8)
    daily_fat_g = round((daily_calories * 0.275) / 9)
    daily_carbs_g = max(0, round(
        (daily_calories - daily_protein_g * 4 - daily_fat_g * 9) / 4
    ))

    return {
        "daily_calories": daily_calories,
        "daily_protein_g": daily_protein_g,
        "daily_fat_g": daily_fat_g,
        "daily_carbs_g": daily_carbs_g,
        "bmr_mifflin": round(bmr_mifflin),
        "bmr_harris": round(bmr_harris),
        "bmr_katch": round(bmr_katch) if bmr_katch is not None else None,
        "formulas_used": len(bmr_values),
    }


def scale_macros(base_calories: int, new_calories: int,
                 protein_g: int, fat_g: int, carbs_g: int) -> tuple[int, int, int]:
    """Scale macros proportionally when calorie limit changes."""
    if base_calories <= 0:
        return (protein_g, fat_g, carbs_g)
    ratio = new_calories / base_calories
    return (round(protein_g * ratio), round(fat_g * ratio), round(carbs_g * ratio))


def compute_streak(logged_dates: list[str]) -> tuple[int, int]:
    """Given date strings (DESC order), return (current_streak, longest_streak)."""
    if not logged_dates:
        return 0, 0
    today = datetime.now(timezone.utc).date()
    dates = sorted(
        {datetime.strptime(d, "%Y-%m-%d").date() for d in logged_dates},
        reverse=True,
    )
    # Current streak: consecutive days ending at today or yesterday
    current = 0
    if dates[0] >= today - timedelta(days=1):
        check = dates[0]
        for d in dates:
            if d == check:
                current += 1
                check = check - timedelta(days=1)
            else:
                break
    # Longest streak over all dates
    longest = 1
    run = 1
    for i in range(1, len(dates)):
        if (dates[i - 1] - dates[i]).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return current, longest


def generate_meal_plan(targets: dict, profile: dict, prefs: dict) -> dict:
    """Generate a 7-day meal plan via AI in 3 parts to stay within token limits."""
    schedule_str = ", ".join(
        f"{s['meal']} at {s['time']}" for s in prefs["schedule"]
    )
    excludes_str = ", ".join(prefs["excludes"]) if prefs["excludes"] else "none"
    goal_desc = GOALS.get(prefs["goal"], prefs["goal"])

    base_context = (
        f"IMPORTANT: Each day MUST total {targets['daily_limit']} kcal "
        f"(within 5%). Do NOT go below this.\n\n"
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

    day_groups = [
        ["Monday", "Tuesday", "Wednesday"],
        ["Thursday", "Friday"],
        ["Saturday", "Sunday"],
    ]

    days_prompt = (
        "You are a professional dietitian. Create meal plans for the specified days "
        "based on the user's profile and preferences. "
        "Return ONLY valid JSON with no extra text.\n\n"
        "CRITICAL: Each day MUST hit the daily calorie target within 5%.\n"
        "Each day MUST hit the protein target within 10%.\n"
        "Distribute calories across all meal slots.\n"
        "Respect excluded foods strictly. Vary meals - no repeats.\n"
        "For lose_weight: high-protein, high-fiber. "
        "For gain_muscle: high-protein, calorie-dense.\n\n"
        "JSON format:\n"
        '{"days": [{"day": "<name>", "meals": [{"slot": "<meal>", "time": "<HH:MM>", '
        '"name": "<dish>", "ingredients": ["<item qty>", ...], '
        '"calories": <int>, "protein_g": <int>, "fat_g": <int>, "carbs_g": <int>}], '
        '"day_total": {"calories": <int>, "protein_g": <int>, "fat_g": <int>, '
        '"carbs_g": <int>}}]}'
    )

    all_days = []
    all_ingredients = []

    for group in day_groups:
        day_names = ", ".join(group)
        user_msg = f"{base_context}\n\nGenerate meals for: {day_names}"
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.5,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": days_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        part = _parse_ai_json(response.choices[0].message.content)
        for day in part.get("days", []):
            all_days.append(day)
            for meal in day.get("meals", []):
                all_ingredients.extend(meal.get("ingredients", []))

    # Validate macro targets before generating shopping list
    macro_warnings = _validate_meal_plan_macros({"days": all_days}, targets)
    if macro_warnings:
        logger.warning("Meal plan macro deviations: %s", "; ".join(macro_warnings))

    # Generate shopping list from all ingredients
    shoplist_prompt = (
        "You are a dietitian assistant. Given this list of ingredients used across "
        "a 7-day meal plan, create a consolidated weekly shopping list. "
        "Combine duplicate items, sum quantities. Estimate realistic prices in "
        f"{prefs['budget_currency']}. Budget is {prefs['budget_amount']} "
        f"{prefs['budget_currency']}.\n"
        "Return ONLY valid JSON:\n"
        '{"shopping_list": [{"item": "<name>", "quantity": "<amount>", '
        '"estimated_cost": <float>}], "total_cost": <float>, '
        f'"currency": "{prefs["budget_currency"]}"' + "}"
    )

    ingredients_str = "\n".join(f"- {ing}" for ing in all_ingredients)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": shoplist_prompt},
            {"role": "user", "content": f"Ingredients:\n{ingredients_str}"},
        ],
    )
    shoplist_data = _parse_ai_json(response.choices[0].message.content)

    return {
        "days": all_days,
        "shopping_list": shoplist_data.get("shopping_list", []),
        "total_cost": shoplist_data.get("total_cost", 0),
        "currency": shoplist_data.get("currency", prefs["budget_currency"]),
    }


def _html(text: str) -> str:
    """Escape text for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_reply(data: dict, today: dict, targets: dict) -> tuple[str, str]:
    """Format calorie estimate with daily progress and macros.
    Returns (text, parse_mode) — parse_mode is 'HTML'."""
    lines = ["\U0001f37d <b>Calorie Estimate</b>"]
    for item in data.get("items", []):
        portion = item.get("portion", "")
        portion_str = f" ({_html(portion)})" if portion else ""
        raw = item.get("per_100g_raw", "?")
        cooked = item.get("per_100g_cooked", "?")
        p = item.get("protein_g", 0)
        f_val = item.get("fat_g", 0)
        c = item.get("carbs_g", 0)
        fib = item.get("fiber_g", 0)
        conf = _confidence_icon(item.get("confidence", ""))
        name = _html(item.get("name", ""))
        lines.append(
            f"{conf} <b>{name}</b>{portion_str}: ~{item['calories']} kcal\n"
            f"   P: {p}g | F: {f_val}g | C: {c}g | Fiber: {fib}g\n"
            f"   <i>per 100g: {raw} raw / {cooked} cooked</i>"
        )

    tp = data.get("total_protein_g", 0)
    tf = data.get("total_fat_g", 0)
    tc = data.get("total_carbs_g", 0)
    tfib = data.get("total_fiber_g", 0)
    lines.append(
        f"\n<b>Meal total:</b> ~{data.get('total', 0)} kcal"
        f" | P: {tp}g | F: {tf}g | C: {tc}g | Fiber: {tfib}g"
    )

    note = data.get("note")
    if note:
        lines.append(f"<i>{_html(note)}</i>")

    # Daily progress
    cal_limit = targets["daily_limit"]
    cal_today = today["calories"]
    remaining = cal_limit - cal_today
    bar = _progress_bar(cal_today, cal_limit)
    if remaining >= 0:
        lines.append(
            f"\n\U0001f4ca <b>Today:</b> {cal_today} / {cal_limit} kcal\n"
            f"{bar} ({remaining} remaining)"
        )
    else:
        lines.append(
            f"\n\u26a0\ufe0f <b>Over limit!</b> {cal_today} / {cal_limit} kcal\n"
            f"{bar} (+{abs(remaining)} over)"
        )

    tp_target = targets["daily_protein_g"]
    tf_target = targets["daily_fat_g"]
    tc_target = targets["daily_carbs_g"]
    if tp_target > 0:
        p_bar = _progress_bar(today["protein_g"], tp_target)
        f_bar = _progress_bar(today["fat_g"], tf_target)
        c_bar = _progress_bar(today["carbs_g"], tc_target)
        lines.append(
            f"P: {today['protein_g']}/{tp_target}g {p_bar}\n"
            f"F: {today['fat_g']}/{tf_target}g {f_bar}\n"
            f"C: {today['carbs_g']}/{tc_target}g {c_bar}"
        )

    return "\n".join(lines), "HTML"


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
        "/profile <height_cm> <weight_kg> <age> <gender> <activity> [body_fat%]\n"
        "  Calculates TDEE using Mifflin-St Jeor + Harris-Benedict (averaged)\n"
        "  Add body fat % for Katch-McArdle formula (most accurate)\n"
        "  example: /profile 180 75 28 male moderate\n"
        "  example: /profile 180 75 28 male moderate 18\n"
        "  Activity: sedentary, light, moderate, active, very_active\n\n"
        "/myprofile - show your profile and daily targets\n\n"
        "/macros - show today's macro progress (P/F/C)\n\n"
        "\U0001f4ca CALORIE TRACKING\n\n"
        "/today - show all meals logged today with totals\n"
        "/week - 7-day calorie and macro summary\n"
        "/history [YYYY-MM-DD] - view meals for a past date (default: yesterday)\n"
        "/delmeal <id> - delete a specific logged meal by ID\n"
        "/stats - logging streaks and 30-day statistics\n"
        "/setlimit <kcal> - override daily calorie limit\n"
        "  (macros scale proportionally)\n"
        "  example: /setlimit 2200\n"
        "/limit - check your current daily limit\n\n"
        "\U0001f4a7 WATER TRACKING\n\n"
        "/water <ml> - log water (e.g. /water 500)\n"
        "/watertoday - show today's water intake\n"
        "  Target is weight-based (~35ml/kg) if profile is set\n"
        "/reminders on|off - toggle water reminders\n"
        "  Schedule: 11:00, 14:00, 16:00, 18:00, 20:00 UTC\n"
        "/weight <kg> - update your weight (e.g. /weight 74.5)\n"
        "/weight - show weight history\n\n"
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
            "Usage: /profile <height_cm> <weight_kg> <age> <gender> <activity> [body_fat%]\n\n"
            f"Activity levels:\n{activity_list}\n\n"
            "Example: /profile 180 75 28 male moderate\n"
            "Example: /profile 180 75 28 male moderate 18  (with body fat % for Katch-McArdle)"
        )
        return

    try:
        height_cm = int(context.args[0])
        weight_kg = float(context.args[1])
        age = int(context.args[2])
        gender = context.args[3].lower()
        activity = context.args[4].lower()
        body_fat_pct = float(context.args[5]) if len(context.args) >= 6 else None
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
    if body_fat_pct is not None and not (5 <= body_fat_pct <= 50):
        await update.message.reply_text("Body fat % must be between 5 and 50.")
        return

    recs = calculate_recommendations(height_cm, weight_kg, age, gender, activity, body_fat_pct)
    rec_cal = recs["daily_calories"]
    rec_p = recs["daily_protein_g"]
    rec_f = recs["daily_fat_g"]
    rec_c = recs["daily_carbs_g"]

    save_profile(update.effective_user.id, height_cm, weight_kg, age, gender, activity,
                 rec_cal, rec_p, rec_f, rec_c, body_fat_pct)
    set_targets(update.effective_user.id, rec_cal, rec_p, rec_f, rec_c)

    # Build BMR breakdown
    bmr_lines = [
        f"  Mifflin-St Jeor: {recs['bmr_mifflin']} kcal/day (BMR)",
        f"  Harris-Benedict: {recs['bmr_harris']} kcal/day (BMR)",
    ]
    if recs["bmr_katch"] is not None:
        bmr_lines.append(
            f"  Katch-McArdle:   {recs['bmr_katch']} kcal/day (BMR, body fat {body_fat_pct}%)"
        )
    bmr_breakdown = "\n".join(bmr_lines)
    formula_note = (
        f"Average of {recs['formulas_used']} formulas"
        if recs["formulas_used"] > 1 else "Mifflin-St Jeor"
    )

    await update.message.reply_text(
        f"\u2705 Profile saved!\n\n"
        f"Height: {height_cm}cm | Weight: {weight_kg}kg\n"
        f"Age: {age} | Gender: {gender}\n"
        f"Activity: {activity} ({ACTIVITY_LEVELS[activity]})\n\n"
        f"\U0001f4ca BMR Estimates:\n{bmr_breakdown}\n\n"
        f"\U0001f3af Daily Targets ({formula_note}):\n"
        f"Calories: {rec_cal} kcal/day (TDEE)\n"
        f"Protein: {rec_p}g | Fat: {rec_f}g | Carbs: {rec_c}g\n\n"
        f"These are now your active daily targets.\n"
        f"Use /setlimit to adjust calories (macros will scale)."
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
        f"\U0001f3af TDEE Recommendations:\n"
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

    cal_bar = _progress_bar(today["calories"], targets["daily_limit"])
    lines = [
        "\U0001f4ca <b>Today's Macros</b>",
        f"Calories: {today['calories']} / {targets['daily_limit']} kcal {cal_bar}",
    ]

    if targets["daily_protein_g"] > 0:
        p_bar = _progress_bar(today["protein_g"], targets["daily_protein_g"])
        f_bar = _progress_bar(today["fat_g"], targets["daily_fat_g"])
        c_bar = _progress_bar(today["carbs_g"], targets["daily_carbs_g"])
        lines.append(f"Protein: {today['protein_g']} / {targets['daily_protein_g']}g {p_bar}")
        lines.append(f"Fat:     {today['fat_g']} / {targets['daily_fat_g']}g {f_bar}")
        lines.append(f"Carbs:   {today['carbs_g']} / {targets['daily_carbs_g']}g {c_bar}")
    else:
        lines.append(f"Protein: {today['protein_g']}g")
        lines.append(f"Fat:     {today['fat_g']}g")
        lines.append(f"Carbs:   {today['carbs_g']}g")
        lines.append("\n<i>Set your profile for personalized targets: /profile</i>")

    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


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
            f"- [#{m['id']}] [{t}] {m['meal_text'][:40]}: ~{m['calories']} kcal"
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
    lines.append("\nUse /delmeal <id> to remove a specific entry.")
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
    target_ml = get_water_target(user.id)
    remaining = target_ml - total

    if remaining > 0:
        await update.message.reply_text(
            f"\U0001f4a7 Logged {amount}ml.\n"
            f"Today: {total} / {target_ml}ml ({remaining}ml remaining)"
        )
    else:
        await update.message.reply_text(
            f"\U0001f4a7 Logged {amount}ml.\n"
            f"\u2705 Today: {total} / {target_ml}ml - target reached!"
        )


async def watertoday_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    total = get_today_water(user_id)
    target_ml = get_water_target(user_id)
    remaining = target_ml - total
    if remaining > 0:
        await update.message.reply_text(
            f"\U0001f4a7 Water today: {total} / {target_ml}ml"
            f" ({remaining}ml remaining)"
        )
    else:
        await update.message.reply_text(
            f"\u2705 Water today: {total} / {target_ml}ml - target reached!"
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
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Yes, delete everything",
                    callback_data=f"confirm_reset_all:{user_id}",
                ),
                InlineKeyboardButton(
                    "Cancel",
                    callback_data=f"cancel_reset_all:{user_id}",
                ),
            ]
        ])
        await update.message.reply_text(
            "Are you sure? This will permanently delete ALL your data:\n"
            "meals, water, profile, diet preferences, calorie limits, and weight history.\n\n"
            "This cannot be undone.",
            reply_markup=keyboard,
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
        "\u2699\ufe0f Generating your 7-day meal plan...\n"
        "This takes ~30 seconds (4 AI calls). Please wait."
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

        # Warn user if any days deviate >10% from calorie target
        cal_target = targets["daily_limit"]
        bad_days = [
            d.get("day", "?")
            for d in plan.get("days", [])
            if not (cal_target * 0.90 <= d.get("day_total", {}).get("calories", 0) <= cal_target * 1.10)
        ]
        if bad_days:
            await update.message.reply_text(
                f"Note: {', '.join(bad_days)} may not hit your calorie target precisely. "
                "The plan is still valid — adjust portions slightly if needed."
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
# New command handlers
# ---------------------------------------------------------------------------

async def delmeal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Usage: /delmeal <id>\n"
            "Find meal IDs in your /today list.\n"
            "Example: /delmeal 42"
        )
        return
    try:
        meal_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            "Please provide a numeric meal ID. Example: /delmeal 42"
        )
        return

    user_id = update.effective_user.id
    meal = get_meal_by_id(meal_id, user_id)
    if not meal:
        await update.message.reply_text(
            f"Meal #{meal_id} not found (or it doesn't belong to you)."
        )
        return

    deleted = delete_meal_by_id(meal_id, user_id)
    if deleted:
        await update.message.reply_text(
            f"\u2705 Deleted meal #{meal_id}: {meal['meal_text'][:50]} (~{meal['calories']} kcal)"
        )
    else:
        await update.message.reply_text(f"Could not delete meal #{meal_id}. Please try again.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if context.args:
        date_str = context.args[0]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            await update.message.reply_text(
                "Invalid date format. Use YYYY-MM-DD.\nExample: /history 2025-04-10"
            )
            return
    else:
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    meals = get_meals_for_date(user_id, date_str)
    if not meals:
        await update.message.reply_text(f"No meals logged for {date_str}.")
        return

    targets = get_targets(user_id)
    total_cal = sum(m["calories"] for m in meals)
    total_p = sum(m["protein_g"] for m in meals)
    total_f = sum(m["fat_g"] for m in meals)
    total_c = sum(m["carbs_g"] for m in meals)
    remaining = targets["daily_limit"] - total_cal

    lines = [f"\U0001f4cb Meals for {date_str}:"]
    for m in meals:
        t = m["created_at"][11:16]
        lines.append(
            f"- [#{m['id']}] [{t}] {m['meal_text'][:40]}: ~{m['calories']} kcal"
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
    await update.message.reply_text("\n".join(lines))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    targets = get_targets(user_id)

    logged_dates = get_logged_dates(user_id, days=30)
    if not logged_dates:
        await update.message.reply_text(
            "No meal data yet. Start logging meals to see your stats!"
        )
        return

    current_streak, longest_streak = compute_streak(logged_dates)
    avg_cal = get_daily_calorie_avg(user_id, days=30)

    water_dates = set(get_water_logged_dates(user_id, days=30))
    water_adherence = round(len(water_dates) / 30 * 100)

    goal_limit = targets["daily_limit"]
    low = goal_limit * 0.9
    high = goal_limit * 1.1
    since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    day_rows = db_query(
        "SELECT SUM(calories) AS total"
        " FROM meals WHERE user_id = %s AND created_at >= %s"
        " GROUP BY SUBSTRING(created_at, 1, 10)",
        (user_id, since), fetch="all",
    )
    on_goal_days = sum(1 for r in day_rows if low <= r["total"] <= high)
    goal_adherence = round(on_goal_days / len(day_rows) * 100) if day_rows else 0

    s = lambda n: "s" if n != 1 else ""
    lines = [
        "\U0001f4ca Your Stats (last 30 days):\n",
        f"Logging streak:  {current_streak} day{s(current_streak)}",
        f"Longest streak:  {longest_streak} day{s(longest_streak)}",
        f"Days logged:     {len(logged_dates)} / 30\n",
        f"Avg daily kcal:  {avg_cal:.0f} kcal  (target: {goal_limit})",
        f"Calorie goal:    {goal_adherence}% of logged days within \u00b110%",
        f"Water logging:   {water_adherence}% of days",
    ]
    await update.message.reply_text("\n".join(lines))


async def weight_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not context.args:
        history = get_weight_history(user_id, limit=10)
        if not history:
            await update.message.reply_text(
                "No weight history yet.\n"
                "Usage: /weight <kg>  (e.g. /weight 74.5)"
            )
            return
        lines = ["Weight history (most recent first):"]
        for entry in history:
            date_str = entry["recorded_at"][:10]
            lines.append(f"- {date_str}: {entry['weight_kg']} kg")
        await update.message.reply_text("\n".join(lines))
        return

    try:
        weight_kg = float(context.args[0].replace(",", "."))
    except ValueError:
        await update.message.reply_text(
            "Please provide a number. Example: /weight 74.5"
        )
        return

    if not (30 <= weight_kg <= 300):
        await update.message.reply_text("Weight must be between 30 and 300 kg.")
        return

    profile = get_profile(user_id)
    if not profile:
        await update.message.reply_text(
            "No profile found. Please set your full profile first:\n"
            "/profile <height_cm> <weight_kg> <age> <gender> <activity>"
        )
        return

    update_profile_weight(user_id, weight_kg)
    save_weight_history(user_id, weight_kg)
    new_water_target = min(int(weight_kg * 35), 3500)

    await update.message.reply_text(
        f"\u2705 Weight updated to {weight_kg} kg.\n\n"
        f"New water target: {new_water_target} ml/day\n"
        f"Run /profile to recalculate calorie & macro recommendations."
    )


async def reset_all_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    parts = query.data.split(":")
    action_part = parts[0]
    owner_id = int(parts[1])

    if owner_id != user_id:
        await query.answer("This button isn't for you.", show_alert=True)
        return

    if action_part == "confirm_reset_all":
        reset_all_data(user_id)
        await query.edit_message_text("\u2705 All your data has been permanently deleted.")
    else:
        await query.edit_message_text("Reset cancelled. Your data is safe.")


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
        reply, parse_mode = format_reply(data, today, targets)
    except json.JSONDecodeError:
        logger.exception("Failed to parse Groq response")
        reply = "Sorry, I couldn't parse the calorie data. Try rephrasing your meal."
        parse_mode = ""
    except Exception:
        logger.exception("Error estimating calories")
        reply = "Something went wrong. Please try again in a moment."
        parse_mode = ""

    await update.message.reply_text(reply, parse_mode=parse_mode or None)


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
        reply, parse_mode = format_reply(data, today, targets)
    except json.JSONDecodeError:
        logger.exception("Failed to parse Groq vision response")
        reply = "Sorry, I couldn't identify the food in this photo. Try adding a caption describing the meal."
        parse_mode = ""
    except Exception:
        logger.exception("Error estimating calories from photo")
        reply = "Something went wrong analyzing the photo. Please try again."
        parse_mode = ""

    await update.message.reply_text(reply, parse_mode=parse_mode or None)


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

    # Register bot commands for all chat types (including groups)
    async def post_init(application):
        from telegram import BotCommand, BotCommandScopeAllGroupChats, BotCommandScopeDefault
        commands = [
            BotCommand("start", "Welcome message"),
            BotCommand("help", "All commands with examples"),
            BotCommand("profile", "Set measurements"),
            BotCommand("myprofile", "Show profile and targets"),
            BotCommand("macros", "Today's macro progress"),
            BotCommand("today", "Today's meals and totals"),
            BotCommand("week", "7-day summary"),
            BotCommand("setlimit", "Override daily calorie limit"),
            BotCommand("limit", "Show current limit"),
            BotCommand("water", "Log water intake"),
            BotCommand("watertoday", "Today's water intake"),
            BotCommand("reminders", "Toggle water reminders"),
            BotCommand("goal", "Set fitness goal"),
            BotCommand("schedule", "Set eating schedule"),
            BotCommand("exclude", "Set foods to avoid"),
            BotCommand("budget", "Set weekly food budget"),
            BotCommand("mealplan", "Generate 7-day meal plan"),
            BotCommand("shoplist", "Show last shopping list"),
            BotCommand("diet", "Show diet preferences"),
            BotCommand("reset", "Reset your data"),
            BotCommand("delmeal", "Delete a logged meal by ID"),
            BotCommand("history", "View meals for a past date"),
            BotCommand("stats", "Streaks and 30-day statistics"),
            BotCommand("weight", "Log weight or view history"),
        ]
        await application.bot.set_my_commands(commands, scope=BotCommandScopeDefault())
        await application.bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

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

    # Reset command + inline keyboard callback
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CallbackQueryHandler(reset_all_callback, pattern=r"^(confirm|cancel)_reset_all:\d+$"))

    # Diet planning commands
    app.add_handler(CommandHandler("goal", goal_command))
    app.add_handler(CommandHandler("schedule", schedule_command))
    app.add_handler(CommandHandler("exclude", exclude_command))
    app.add_handler(CommandHandler("budget", budget_command))
    app.add_handler(CommandHandler("mealplan", mealplan_command))
    app.add_handler(CommandHandler("shoplist", shoplist_command))
    app.add_handler(CommandHandler("diet", diet_command))

    # New commands
    app.add_handler(CommandHandler("delmeal", delmeal_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("weight", weight_command))

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
