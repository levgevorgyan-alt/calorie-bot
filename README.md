# Calorie Bot

A Telegram bot that estimates calories and macronutrients (protein, fat, carbs)
from meal descriptions or photos using AI. Features personalized daily targets
based on user measurements, daily/weekly tracking, water intake logging, and
scheduled water reminders.

## Prerequisites

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and save the token.
2. Get a free Groq API key from [console.groq.com](https://console.groq.com).
3. Create a free [Supabase](https://supabase.com) project and copy the database connection string.

## Local Development

```bash
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="your-token"
export GROQ_API_KEY="your-key"
export DATABASE_URL="postgresql://postgres:password@db.xxxxx.supabase.co:5432/postgres"

# Run in polling mode (no webhook needed locally)
python bot.py --poll
```

## Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) > **New** > **Web Service**.
3. Connect your GitHub repo.
4. Set **Build Command** to `pip install -r requirements.txt`.
5. Set **Start Command** to `python bot.py`.
6. Add environment variables under **Environment**:
   - `TELEGRAM_BOT_TOKEN` - your bot token from BotFather
   - `GROQ_API_KEY` - your Groq API key
   - `DATABASE_URL` - your Supabase PostgreSQL connection string
   - `WEBHOOK_URL` - the URL Render assigns (e.g. `https://calorie-bot.onrender.com`)
7. Click **Deploy**.

## Usage

Send any meal description to the bot (DM or group chat), or send a photo of your meal:

```
2 scrambled eggs, toast with butter, black coffee
```

You can also send a **photo** of your meal. Add a caption for better accuracy
(e.g. "about 200g of pasta with sauce").

If your description is vague (e.g. just "egg"), the bot will ask for more detail before estimating — cooking method, quantity, preparation.

The bot replies with calories, macros, fiber, confidence indicator, per-100g values, and daily progress with visual progress bars (HTML-formatted):

```
🍽 Calorie Estimate:
- Scrambled eggs (2 large, 100g): ~182 kcal
  P: 12g | F: 14g | C: 2g
  per 100g: 143 raw / 182 cooked
- Toast with butter (50g): ~178 kcal
  P: 4g | F: 9g | C: 21g
  per 100g: 265 raw / 265 cooked

Total: ~360 kcal | P: 16g | F: 23g | C: 23g
📊 Today: 360 / 2100 kcal (1740 remaining)
   P: 16/150g | F: 23/65g | C: 23/250g
```

## Commands

### Profile
- `/profile <height_cm> <weight_kg> <age> <gender> <activity> [body_fat%]` - Compute TDEE and auto-apply targets
  - Uses Mifflin-St Jeor + Harris-Benedict averaged; add body fat % for Katch-McArdle (most accurate)
  - Response shows per-formula BMR breakdown
  - Activity levels: `sedentary`, `light`, `moderate`, `active`, `very_active`
- `/myprofile` - Show your profile and daily targets
- `/macros` - Show today's macro progress (P/F/C)
- `/weight <kg>` - Update your weight quickly (e.g. `/weight 74.5`)
- `/weight` - Show weight history
- `/language` - Change language: 🇬🇧 English / 🇷🇺 Русский (auto-detected from Telegram settings)
- `/timezone` - Auto-detect timezone via location share, or set manually (`/timezone Europe/Berlin`, `/timezone UTC+3`)
- `/export [days]` - Export meal and water logs as CSV files (default: 30 days)

### Calories
- `/today` - Show today's meals with inline delete buttons (tap to remove a meal)
- `/week` - 7-day calorie summary
- `/history [YYYY-MM-DD]` - View meals for a date (defaults to today); includes ⬅️/➡️ day navigation buttons
- `/delmeal <id>` - Delete a specific logged meal by ID
- `/stats` - Logging streaks and 30-day statistics
- `/setlimit <kcal>` - Override daily calorie limit (macros scale proportionally)
- `/limit` - Show your current limit

### Water
- `/water <ml>` - Log water intake (e.g. `/water 500`); reply includes quick-add buttons
- `/water` - No arg: shows today's progress + 250/500/750/1000ml tap-to-log buttons
- `/watertoday` - Today's water intake with quick-add buttons
- `/reminders` - No arg: shows current ON/OFF status with toggle buttons
- `/reminders on` / `/reminders off` - Enable or disable water reminders via text command

### Water Reminder Schedule (UTC)
| Time  | Amount |
|-------|--------|
| 11:00 | 500ml  |
| 14:00 | 500ml  |
| 16:00 | 250ml  |
| 18:00 | 500ml  |
| 20:00 | 250ml  |

Daily target: weight-based (~35 ml/kg, capped at 3500 ml) when profile is set, otherwise 2000 ml.

### Reset
- `/reset meals` - Clear today's meal logs
- `/reset water` - Clear today's water logs
- `/reset all` - Delete all your data (asks for confirmation via inline button)

### Diet Planning
- `/goal` - No arg: shows current goal with 3 tap-to-select buttons (Lose Weight / Maintain / Gain Muscle)
- `/goal <lose_weight|maintain|gain_muscle>` - Set goal via text command
- `/schedule <meal time>, ...` - Set eating schedule (e.g. `/schedule breakfast 8:00, lunch 13:00, dinner 19:00`)
- `/exclude <foods>` - Set foods to avoid (e.g. `/exclude pork, shellfish`). Use `/exclude clear` to reset.
- `/budget <amount> <currency>` - Set weekly food budget (e.g. `/budget 80 EUR`)
- `/mealplan` - Generate a 7-day meal plan with shopping list
- `/shoplist` - Re-show last shopping list
- `/diet` - Show current diet preferences

### Examples

**Set diet preferences:**
```
/goal lose_weight
/schedule breakfast 8:00, lunch 13:00, snack 16:00, dinner 19:00
/exclude pork, shellfish, peanuts
/budget 80 EUR
```

**Generate a weekly meal plan:**
```
/mealplan
```

**Set up your profile:**
```
/profile 180 75 28 male moderate
/profile 180 75 28 male moderate 18    (with body fat % — adds Katch-McArdle formula)
```

**Log a meal** (just type it):
```
chicken salad with rice
2 eggs, toast with butter, black coffee
big mac, medium fries, diet coke
```

**Log a meal by photo:**
Send a photo of your meal. Optionally add a caption like "about 200g of pasta" for better accuracy.

**Override calorie limit (macros scale):**
```
/setlimit 2200
```

**Log water:**
```
/water 500
```

**Check daily progress:**
```
/today
/watertoday
/macros
```

**Weekly summary:**
```
/week
```

**Enable water reminders in a group:**
```
/reminders on
```

**Reset today's data:**
```
/reset meals
/reset water
```

## Notes

- Render free tier spins down after 15 min of inactivity. First message after idle has a ~30s cold start.
- In group chats, turn off Group Privacy via BotFather so the bot sees all messages.
- Data is stored in Supabase PostgreSQL. Persists across redeploys.
- Set your profile first with `/profile` for personalized calorie and macro recommendations.
- Activity levels: sedentary (no exercise), light (1-3 days/week), moderate (3-5 days/week), active (6-7 days/week), very_active (hard daily exercise or physical job).
