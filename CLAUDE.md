# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Telegram bot for meal tracking, nutrition estimation, and dietary planning. Estimates calories and macros from text or photos via Groq/Llama, stores user data in PostgreSQL (Supabase), and deploys to Render.com (webhook) or runs locally (polling). English-only, single-tier (no premium gating), all timestamps in UTC.

**`AGENTS.md` is the authoritative deep reference** for tables, command behaviors, callback patterns, AI models, and design notes. Read it before non-trivial changes. Keep `AGENTS.md` and `README.md` in sync when behavior or commands change (this is a standing user requirement).

## Running the Bot

```bash
pip install -r requirements.txt
# Local (no public URL needed):
python bot.py --poll
# Production (requires WEBHOOK_URL):
python bot.py
```

There is no test suite, no linter config, and no build step — `python bot.py [--poll]` is the only entry point.

## Required Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) |
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |
| `DATABASE_URL` | PostgreSQL connection string (Supabase) |
| `WEBHOOK_URL` | Public HTTPS URL (production only) |
| `PORT` | Server port (default: 10000) |

## Architecture

Single-file application: **`bot.py`** (~2850 lines). Sections in order (line ranges approximate, follow the `# ---` divider comments to reorient):

1. **Config & constants** (~1–230) — env vars, Groq client, `_groq_with_retry()`, `SYSTEM_PROMPT`, `VISION_PROMPT`, `PREFLIGHT_PROMPT`, `MEALPLAN_PROMPT`, `WATER_SCHEDULE`, `DIET_PREF_COLUMNS`, `_HELP_QUICK_START`.
2. **Database helpers** — `psycopg2` via `db_query()`. `db_init()` is idempotent and adds the `body_fat_pct` column via `ALTER TABLE IF NOT EXISTS` for existing deployments.
3. **AI helpers** — `_parse_ai_json()`, `_validate_meal_plan_macros()`, `_progress_bar()`, `preflight_meal_input()` (cheap input validator, see below), `estimate_calories()` (text), `estimate_calories_from_photo()` (vision, with `resize_image_if_needed()` for >800 KB photos), `calculate_recommendations()` (pure-Python TDEE — Mifflin-St Jeor + Harris-Benedict, optional Katch-McArdle), `generate_meal_plan()` (async; 3 day-group calls run concurrently via `asyncio.gather(asyncio.to_thread(...))`, then sequential shopping-list call), `format_reply()`.
4. **Command handlers** — `/start`, `/help`, `/profile`, `/myprofile`, `/macros`, `/today`, `/week`, `/setlimit`, `/limit`, `/water`, `/watertoday`, `/reminders`, `/reset`, `/goal`, `/schedule`, `/exclude`, `/budget`, `/diet`, `/mealplan`, `/shoplist`, `/delmeal`, `/history`, `/stats`, `/weight`, `/try`, `/export`, plus all callback handlers.
5. **Message handlers** — `handle_meal()` (text), `handle_photo()`.
6. **Water reminder job** — `JobQueue` daily task broadcasting to opted-in chats.
7. **Main** — wires handlers, webhook vs. polling, registers BotCommand list.

## Database Tables (in `db_init()`)

`meals`, `limits`, `profiles` (with `body_fat_pct` column added via ALTER), `water`, `reminder_chats`, `diet_prefs`, `meal_plans`, `weight_history`. All timestamps UTC; day boundaries midnight UTC. User isolation by Telegram numeric `user_id` everywhere.

Older deployments may still have `profiles.language`, `profiles.user_timezone`, and a `subscriptions` table from previous versions — the bot ignores them. They were intentionally left in place to avoid destructive migrations.

## Key Design Decisions Worth Knowing

- **AI input preflight** — `preflight_meal_input()` calls the cheap `llama-3.1-8b-instant` model before estimation. It returns one of `non_food` (greeting/chatter — bot replies with a hint), `needs_clarification` (typo or vague input — bot suggests a correction and asks a targeted question, e.g. "Did you mean **egg**? How was it cooked — boiled, fried, or scrambled?"), or `ok` (proceed to the heavy 70b estimation). Cached in `_PREFLIGHT_CACHE` (FIFO, max 200). On preflight failure, falls through to estimation rather than blocking the user.
- **Two-turn clarification** — when preflight asks a question, the original input is stashed in `context.user_data["pending_meal"]`. The next message is concatenated as `"original (reply)"` and sent straight to estimation, skipping preflight.
- **No retry beyond `_groq_with_retry`** — one 2-second retry on `GroqRateLimitError`, then surface error. Free Groq tier is 30 req/min.
- **Parallel mealplan** — 3 day-group Groq calls in parallel via `asyncio.gather` cuts ~30s → ~10s. Don't serialize them.
- **`scale_macros()`** — when `/setlimit` changes calories, macro targets scale proportionally from the original AI recommendations stored in `profiles`.
- **Callback data** — every pattern embeds the owner's `user_id` for cross-user abuse prevention in groups; chat IDs use `-?\d+` for negative group IDs. Stay under Telegram's 64-byte limit.
- **HTML parse mode** — `format_reply()` returns `(text, parse_mode)`. All user-supplied text in replies is escaped via `_html()`. Don't switch to Markdown without escaping all interpolations.
- **AI JSON shape is load-bearing** — meal estimates include `protein_g`, `fat_g`, `carbs_g`, `fiber_g`, `confidence`, `context_hint`, plus `total_*` and per-100g raw/cooked values. Preserve this shape across any provider change.
- **`DIET_PREF_COLUMNS` frozenset** — allowlists kwarg names in `update_diet_pref()` to block SQL injection from crafted kwargs. Add new columns to the frozenset, not just the call site.

## Adding Features

- New command: write `async def`, register with `app.add_handler(CommandHandler(...))` in `main()` **before** the catch-all `MessageHandler`. Add to BotCommand list.
- New table or column: add `CREATE TABLE` / `ALTER TABLE IF NOT EXISTS` in `db_init()` so existing deployments migrate on startup.
