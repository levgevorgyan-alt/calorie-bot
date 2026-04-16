# Calorie Bot - Agent Instructions

## Project Overview
A Telegram bot that estimates meal calories and macronutrients (protein, fat,
carbs) using AI (Groq/Llama), tracks daily intake per user with personalized
targets based on body measurements, enforces configurable calorie/macro limits,
logs water intake, and sends scheduled water reminders. Supports both text
descriptions and meal photos.

## Tech Stack
- **Language:** Python 3.9+
- **Telegram SDK:** `python-telegram-bot` v20+ (async, webhook mode with job-queue)
- **AI — meal text:** Groq `llama-3.3-70b-versatile` (temp 0.3)
- **AI — meal photo:** Groq `meta-llama/llama-4-scout-17b-16e-instruct` (temp 0.3)
- **AI — meal plan days:** Groq `meta-llama/llama-4-maverick-17b-128e-instruct` (temp 0.5)
- **AI — shopping list:** Groq `llama-3.1-8b-instant` (temp 0.3, simpler task)
- **TDEE/BMR:** Pure Python — Mifflin-St Jeor + Harris-Benedict (revised) + optional Katch-McArdle (no API call)
- **Database:** PostgreSQL via Supabase (free hosted, persistent)
- **Hosting:** Render (web service, webhook-based)
- **Image processing:** Pillow (resize photos >800 KB before Groq upload)

## Project Structure
```
bot.py              # Single-file application (all logic)
requirements.txt    # Python dependencies
render.yaml         # Render deployment blueprint
.env.example        # Required environment variables
README.md           # User-facing documentation
AGENTS.md           # This file
```

## Architecture
All logic lives in `bot.py` as a single-file application. There is no module
structure. The file is organized into these sections:

1. **Config & constants** - env vars, defaults, system/vision/profile prompts, water schedule
2. **Database helpers** - `db_connect()`, `db_init()`, CRUD functions for meals,
   water, limits, profiles, diet_prefs, meal_plans, and reminder_chats tables. Includes migration logic
   for adding macro columns to older databases.
3. **AI helpers** - `estimate_calories()`, `estimate_calories_from_photo()`,
   `calculate_recommendations()`, `generate_meal_plan()`, `scale_macros()`,
   `compute_streak()`, `resize_image_if_needed()`, `_parse_ai_json()`, `format_reply()`
4. **Command handlers** - async functions for `/start`, `/help`, `/profile`,
   `/myprofile`, `/macros`, `/today`, `/week`, `/history`, `/delmeal`, `/stats`,
   `/setlimit`, `/limit`, `/water`, `/watertoday`, `/reminders`, `/weight`,
   `/reset`, `/goal`, `/schedule`, `/exclude`, `/budget`, `/mealplan`, `/shoplist`,
   `/diet`, plus `reset_all_callback` (inline keyboard handler)
5. **Meal handler** - `handle_meal()` processes any non-command text message
6. **Photo handler** - `handle_photo()` downloads the photo, sends to vision model
7. **Water reminder job** - `send_water_reminder()` runs on a daily schedule
8. **Main** - wires handlers, registers scheduled jobs, starts webhook or polling

## Database Schema
Eight tables in PostgreSQL (Supabase):

- **meals** - `id` (PK), `user_id`, `username`, `chat_id`, `meal_text`, `calories`, `protein_g`, `fat_g`, `carbs_g`, `created_at`
- **limits** - `user_id` (PK), `daily_limit`, `daily_protein_g`, `daily_fat_g`, `daily_carbs_g`
- **profiles** - `user_id` (PK), `height_cm`, `weight_kg`, `age`, `gender`, `activity`, `rec_calories`, `rec_protein_g`, `rec_fat_g`, `rec_carbs_g`
- **water** - `id` (PK), `user_id`, `username`, `chat_id`, `amount_ml`, `created_at`
- **reminder_chats** - `chat_id` (PK)
- **diet_prefs** - `user_id` (PK), `goal`, `schedule` (JSON), `excludes` (JSON), `budget_amount`, `budget_currency`
- **meal_plans** - `user_id` (PK), `week_start`, `plan_json`, `shoplist_json`, `created_at`
- **weight_history** - `id` (PK), `user_id`, `weight_kg`, `recorded_at`

All timestamps are ISO 8601 UTC. Day boundaries are midnight UTC.

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from BotFather |
| `GROQ_API_KEY` | Yes | API key from console.groq.com |
| `WEBHOOK_URL` | Yes (prod) | Public URL assigned by Render |
| `PORT` | No | Defaults to 10000; Render sets automatically |
| `DATABASE_URL` | Yes | PostgreSQL connection string from Supabase |

## Key Behaviors
- Every non-command text message is treated as a meal description and sent to the
  AI for calorie and macro estimation.
- Photo messages are analyzed using Groq's vision model
  (`meta-llama/llama-4-scout-17b-16e-instruct`) to identify food items and
  estimate calories/macros. Captions are passed as extra context. Photos >800 KB
  are resized to 1024px / quality-reduced before base64 encoding via `resize_image_if_needed()`.
- The AI returns JSON with per-item calories, macros (P/F/C), portion sizes, and
  per-100g values for both raw and cooked.
- Each meal reply includes the user's daily running total for calories and macros
  vs. their targets.
- Users are identified by Telegram numeric `user_id`. Each user's data is independent.
- `/profile` computes TDEE via `calculate_recommendations()` (pure Python, no API call)
  using the average of Mifflin-St Jeor and Harris-Benedict (revised). If a 6th arg
  `body_fat%` is provided, Katch-McArdle is also included in the average. Results are
  applied immediately via `set_targets()`. Profile response shows per-formula BMR breakdown.
- `PROFILE_PROMPT` has been removed — TDEE is now deterministic Python math.
- Users can update weight alone with `/weight <kg>` without re-running `/profile`.
  Weight is logged to `weight_history`; daily water target updates automatically.
- Water daily target is weight-based: `min(weight_kg * 35, 3500)` ml when profile
  exists, otherwise falls back to 2000 ml constant. Computed by `get_water_target()`.
- When `/setlimit` changes the calorie target, macro targets are scaled
  proportionally based on the original AI recommendations from the profile.
- Water reminders are sent at 11:00, 14:00, 16:00, 18:00, 20:00 UTC to chats
  that have opted in via `/reminders on`.
- `/reset all` shows an inline keyboard confirmation (2 buttons) before deleting.
  `callback_data` encodes the owner's `user_id` to prevent cross-user abuse in groups.
- `DIET_PREF_COLUMNS` frozenset allowlists column names in `update_diet_pref()` to
  prevent SQL injection from crafted kwargs.
- Non-food text is filtered by `_looks_like_food()` before calling Groq. Short food
  descriptions (1–3 words, no quantity/cooking method) trigger `_needs_clarification()`,
  which sends a specific question and stores the original text in `context.user_data["pending_meal"]`.
  The next message combines both and calls the AI.
- `_truncate(text, limit=40)` appends `…` when meal text is cut in `/today`, `/history`, `/delmeal`.
- `/today` uses inline keyboard buttons (one per meal) for one-tap deletion via
  `delmeal_inline_callback`. Callback pattern: `delmeal_inline:<meal_id>:<user_id>`.
- `/mealplan` uses a pre-flight checklist (✅/❌ per requirement) before generating,
  and sends output as 3 messages (Mon-Wed, Thu-Sun, Shopping list) via `_format_day_block()`.
- `/diet` shows action hints for missing fields (e.g., "not set → /schedule ...").
- `/help` sends a compact quick-start (`_HELP_QUICK_START` constant) by default;
  `/help full` sends the full reference as a second message.
- `/start` shows numbered onboarding steps with a "Quick Start Guide" inline button
  handled by `help_quickstart_callback()`.
- Meal estimation responses use Telegram HTML parse mode (`parse_mode="HTML"`). All
  user-supplied text in replies is escaped via `_html()`. `format_reply()` returns
  `(text, parse_mode)` tuple. Progress bars rendered via `_progress_bar()`.
- Meal calorie estimates now include `fiber_g`, `confidence` (high/medium/low), and
  `context_hint` (home_cooked/restaurant/branded/unknown) from the AI response.
  Restaurant meals prompt the model to assume 20-30% larger portions.
- After meal plan generation, `_validate_meal_plan_macros()` checks each day against
  ±5% calorie and ±10% protein tolerances; logs warnings, notifies user if >10% off.
- `/help` shows all commands organized by category with inline usage examples.

## Bot Commands Reference
| Command | Description | Example |
|---------|-------------|---------|
| *(plain text)* | Estimate calories and macros for a meal | `chicken salad with rice` |
| *(photo)* | Estimate from a meal photo | Send a photo, optionally with caption |
| `/start` | Welcome message | `/start` |
| `/help` | Full command list with examples | `/help` |
| `/profile <h> <w> <age> <gender> <activity> [bf%]` | Compute TDEE (multi-formula), auto-apply targets | `/profile 180 75 28 male moderate` or `/profile 180 75 28 male moderate 18` |
| `/myprofile` | Show profile and daily targets | `/myprofile` |
| `/macros` | Today's macro progress (P/F/C) | `/macros` |
| `/today` | Today's meals with IDs, calories, and macros | `/today` |
| `/week` | 7-day calorie and macro summary | `/week` |
| `/history [YYYY-MM-DD]` | Meals for a past date (default: yesterday) | `/history 2025-04-10` |
| `/delmeal <id>` | Delete a specific logged meal by ID | `/delmeal 42` |
| `/stats` | Logging streaks and 30-day statistics | `/stats` |
| `/setlimit <kcal>` | Override calorie limit; macros scale proportionally | `/setlimit 2200` |
| `/limit` | Show current calorie limit and macro targets | `/limit` |
| `/water <ml>` | Log water intake | `/water 500` |
| `/watertoday` | Today's water total vs personal target | `/watertoday` |
| `/reminders on/off` | Toggle water reminders for this chat | `/reminders on` |
| `/weight <kg>` | Update weight and log to history | `/weight 74.5` |
| `/weight` | Show weight history | `/weight` |
| `/reset meals` | Clear today's meal logs | `/reset meals` |
| `/reset water` | Clear today's water logs | `/reset water` |
| `/reset all` | Delete all user data (with inline confirmation) | `/reset all` |
| `/goal <goal>` | Set fitness goal | `/goal lose_weight` |
| `/schedule <meal time>, ...` | Set eating schedule | `/schedule breakfast 8:00, lunch 13:00` |
| `/exclude <foods>` | Set foods to avoid | `/exclude pork, shellfish` |
| `/budget <amount> <currency>` | Set weekly food budget | `/budget 80 EUR` |
| `/mealplan` | Generate 7-day meal plan with shopping list | `/mealplan` |
| `/shoplist` | Re-show last shopping list | `/shoplist` |
| `/diet` | Show current diet preferences | `/diet` |

## Running Locally
```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="..."
export GROQ_API_KEY="..."
python bot.py --poll
```
The `--poll` flag uses long polling instead of webhooks (no public URL needed).

## Deployment
Push to GitHub. Render auto-deploys from the connected repo. Env vars are set in
the Render dashboard under Environment. The bot registers its Telegram webhook
automatically on startup.

## Common Pitfalls
- **Group privacy:** BotFather's Group Privacy must be turned OFF for the bot to
  see non-command messages in group chats.
- **Render free tier:** The service spins down after 15 min idle (~30s cold start).
  Use UptimeRobot to keep it alive. Data persists in Supabase across redeploys.
- **Groq rate limits:** Free tier allows 30 req/min. No retry logic is implemented;
  errors are caught and a fallback message is sent to the user.

## Extending the Bot
- To add a new command: write an async handler function, register it with
  `app.add_handler(CommandHandler("name", handler))` in `main()` before the
  `MessageHandler` (which must stay last).
- To add a new DB table: add the CREATE TABLE statement in `db_init()`.
- To change the AI provider: modify `estimate_calories()` and update
  `requirements.txt`. The rest of the code only depends on the returned JSON shape.
- The AI JSON schema includes macros (`protein_g`, `fat_g`, `carbs_g`,
  `total_protein_g`, `total_fat_g`, `total_carbs_g`) alongside calories.
  Any provider change must preserve this schema.
- Database uses `psycopg2` with a `db_query()` helper that handles connections,
  cursors, and cleanup. All queries use `%s` parameter style.
