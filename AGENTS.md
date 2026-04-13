# Calorie Bot - Agent Instructions

## Project Overview
A Telegram bot that estimates meal calories using AI (Groq/Llama), tracks daily
intake per user, enforces configurable calorie limits, logs water intake, and
sends scheduled water reminders.

## Tech Stack
- **Language:** Python 3.9+
- **Telegram SDK:** `python-telegram-bot` v20+ (async, webhook mode with job-queue)
- **AI Provider:** Groq API with `llama-3.3-70b-versatile` model
- **AI Vision:** Groq API with `meta-llama/llama-4-scout-17b-16e-instruct` model (photo analysis)
- **Database:** SQLite (single file `calories.db`)
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

1. **Config & constants** - env vars, defaults, system prompt, water schedule
2. **Database helpers** - `db_connect()`, `db_init()`, CRUD functions for meals,
   water, limits, and reminder_chats tables
3. **Calorie estimation** - `estimate_calories()` calls Groq, `format_reply()`
   builds the Telegram response. `estimate_calories_from_photo()` uses the
   vision model for photo-based estimation.
4. **Command handlers** - async functions for `/start`, `/help`, `/today`,
   `/week`, `/setlimit`, `/limit`, `/water`, `/watertoday`, `/reminders`, `/reset`
5. **Meal handler** - `handle_meal()` processes any non-command text message
6. **Photo handler** - `handle_photo()` downloads the photo, sends to vision model
7. **Water reminder job** - `send_water_reminder()` runs on a daily schedule
8. **Main** - wires handlers, registers scheduled jobs, starts webhook or polling

## Database Schema
Four tables in SQLite (`calories.db`):

- **meals** - `id`, `user_id`, `username`, `chat_id`, `meal_text`, `calories`, `created_at`
- **limits** - `user_id` (PK), `daily_limit` (default 1800)
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
| `DB_PATH` | No | SQLite file path; defaults to `calories.db` |

## Key Behaviors
- Every non-command text message is treated as a meal description and sent to the
  AI for calorie estimation.
- Photo messages are analyzed using Groq's vision model (`llama-3.2-90b-vision-preview`)
  to identify food items and estimate calories. Captions are passed as extra context.
- The AI returns JSON with per-item calories, portion sizes, and per-100g values
  for both raw and cooked.
- Each meal reply includes the user's daily running total and remaining calories
  vs. their limit.
- Users are identified by Telegram numeric `user_id`. Each user's data is independent.
- Water reminders are sent at 11:00, 14:00, 16:00, 18:00, 20:00 UTC to chats
  that have opted in via `/reminders on`.
- Users can reset their own data with `/reset meals`, `/reset water`, or `/reset all`.
- `/help` shows all commands organized by category with inline usage examples.

## Bot Commands Reference
| Command | Description | Example |
|---------|-------------|---------|
| *(plain text)* | Estimate calories for a meal | `chicken salad with rice` |
| *(photo)* | Estimate calories from a meal photo | Send a photo, optionally with caption |
| `/start` | Welcome message | `/start` |
| `/help` | Full command list with examples | `/help` |
| `/today` | Today's meals and calorie total | `/today` |
| `/week` | 7-day calorie summary | `/week` |
| `/setlimit <kcal>` | Set daily calorie limit (500-10000) | `/setlimit 2200` |
| `/limit` | Show current calorie limit | `/limit` |
| `/water <ml>` | Log water intake | `/water 500` |
| `/watertoday` | Today's water total vs 2000ml target | `/watertoday` |
| `/reminders on/off` | Toggle water reminders for this chat | `/reminders on` |
| `/reset meals` | Clear today's meal logs | `/reset meals` |
| `/reset water` | Clear today's water logs | `/reset water` |
| `/reset all` | Delete all user data (meals, water, limit) | `/reset all` |

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
  SQLite data is lost on redeploy.
- **Groq rate limits:** Free tier allows 30 req/min. No retry logic is implemented;
  errors are caught and a fallback message is sent to the user.
- **render.yaml** still references `OPENAI_API_KEY` in envVars (legacy); the actual
  code uses `GROQ_API_KEY`. Update render.yaml if using blueprint deploys.

## Extending the Bot
- To add a new command: write an async handler function, register it with
  `app.add_handler(CommandHandler("name", handler))` in `main()` before the
  `MessageHandler` (which must stay last).
- To add a new DB table: add the CREATE TABLE statement in `db_init()`.
- To change the AI provider: modify `estimate_calories()` and update
  `requirements.txt`. The rest of the code only depends on the returned JSON shape.
- To switch to persistent storage: replace SQLite calls with a PostgreSQL client
  (e.g. `psycopg2`). The query patterns are standard SQL and transfer directly.
