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
- **AI Provider:** Groq API with `llama-3.3-70b-versatile` model (text + profile)
- **AI Vision:** Groq API with `meta-llama/llama-4-scout-17b-16e-instruct` model (photo analysis)
- **Database:** PostgreSQL via Supabase (free hosted, persistent)
- **Hosting:** Render (web service, webhook-based)

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
   `_parse_ai_json()`, `format_reply()`
4. **Command handlers** - async functions for `/start`, `/help`, `/profile`,
   `/myprofile`, `/macros`, `/today`, `/week`, `/setlimit`, `/limit`,
   `/water`, `/watertoday`, `/reminders`, `/reset`, `/goal`, `/schedule`,
   `/exclude`, `/budget`, `/mealplan`, `/shoplist`, `/diet`
5. **Meal handler** - `handle_meal()` processes any non-command text message
6. **Photo handler** - `handle_photo()` downloads the photo, sends to vision model
7. **Water reminder job** - `send_water_reminder()` runs on a daily schedule
8. **Main** - wires handlers, registers scheduled jobs, starts webhook or polling

## Database Schema
Five tables in SQLite (`calories.db`):

- **meals** - `id`, `user_id`, `username`, `chat_id`, `meal_text`, `calories`, `protein_g`, `fat_g`, `carbs_g`, `created_at`
- **limits** - `user_id` (PK), `daily_limit`, `daily_protein_g`, `daily_fat_g`, `daily_carbs_g`
- **profiles** - `user_id` (PK), `height_cm`, `weight_kg`, `age`, `gender`, `activity`, `rec_calories`, `rec_protein_g`, `rec_fat_g`, `rec_carbs_g`
- **water** - `id`, `user_id`, `username`, `chat_id`, `amount_ml`, `created_at`
- **reminder_chats** - `chat_id` (PK)

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
  estimate calories/macros. Captions are passed as extra context.
- The AI returns JSON with per-item calories, macros (P/F/C), portion sizes, and
  per-100g values for both raw and cooked.
- Each meal reply includes the user's daily running total for calories and macros
  vs. their targets.
- Users are identified by Telegram numeric `user_id`. Each user's data is independent.
- Users set their body measurements via `/profile` (height cm, weight kg, age,
  gender, activity level). The AI calculates recommended daily calories and macros
  using Mifflin-St Jeor BMR with the specified activity multiplier (sedentary=1.2,
  light=1.375, moderate=1.55, active=1.725, very_active=1.9).
- When `/setlimit` changes the calorie target, macro targets are scaled
  proportionally based on the original AI recommendations from the profile.
- Water reminders are sent at 11:00, 14:00, 16:00, 18:00, 20:00 UTC to chats
  that have opted in via `/reminders on`.
- Users can reset their own data with `/reset meals`, `/reset water`, or `/reset all`.
- `/help` shows all commands organized by category with inline usage examples.

## Bot Commands Reference
| Command | Description | Example |
|---------|-------------|---------|
| *(plain text)* | Estimate calories and macros for a meal | `chicken salad with rice` |
| *(photo)* | Estimate from a meal photo | Send a photo, optionally with caption |
| `/start` | Welcome message | `/start` |
| `/help` | Full command list with examples | `/help` |
| `/profile <h> <w> <age> <gender> <activity>` | Set body measurements and activity level | `/profile 180 75 28 male moderate` |
| `/myprofile` | Show profile and daily targets | `/myprofile` |
| `/macros` | Today's macro progress (P/F/C) | `/macros` |
| `/today` | Today's meals, calories, and macros | `/today` |
| `/week` | 7-day calorie and macro summary | `/week` |
| `/setlimit <kcal>` | Override calorie limit; macros scale proportionally | `/setlimit 2200` |
| `/limit` | Show current calorie limit and macro targets | `/limit` |
| `/water <ml>` | Log water intake | `/water 500` |
| `/watertoday` | Today's water total vs 2000ml target | `/watertoday` |
| `/reminders on/off` | Toggle water reminders for this chat | `/reminders on` |
| `/reset meals` | Clear today's meal logs | `/reset meals` |
| `/reset water` | Clear today's water logs | `/reset water` |
| `/reset all` | Delete all user data (meals, water, profile, limits) | `/reset all` |

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
