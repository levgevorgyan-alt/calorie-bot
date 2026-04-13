# Calorie Bot

A Telegram bot that estimates calories and macronutrients (protein, fat, carbs)
from meal descriptions or photos using AI. Features personalized daily targets
based on user measurements, daily/weekly tracking, water intake logging, and
scheduled water reminders.

## Prerequisites

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and save the token.
2. Get a free Groq API key from [console.groq.com](https://console.groq.com).

## Local Development

```bash
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="your-token"
export GROQ_API_KEY="your-key"

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
   - `WEBHOOK_URL` - the URL Render assigns (e.g. `https://calorie-bot.onrender.com`)
7. Click **Deploy**.

## Usage

Send any meal description to the bot (DM or group chat), or send a photo of your meal:

```
2 scrambled eggs, toast with butter, black coffee
```

You can also send a **photo** of your meal. Add a caption for better accuracy
(e.g. "about 200g of pasta with sauce").

The bot replies with calories, macros, per-100g values, and daily progress:

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
- `/profile <height_cm> <weight_kg> <age> <gender>` - Set measurements for personalized targets
- `/myprofile` - Show your profile and daily targets
- `/macros` - Show today's macro progress (P/F/C)

### Calories
- `/today` - Show today's meals and calorie total
- `/week` - 7-day calorie summary
- `/setlimit <kcal>` - Override daily calorie limit (macros scale proportionally)
- `/limit` - Show your current limit

### Water
- `/water <ml>` - Log water intake (e.g. `/water 500`)
- `/watertoday` - Show today's water intake
- `/reminders on` - Enable water reminders in this chat
- `/reminders off` - Disable water reminders

### Water Reminder Schedule (UTC)
| Time  | Amount |
|-------|--------|
| 11:00 | 500ml  |
| 14:00 | 500ml  |
| 16:00 | 250ml  |
| 18:00 | 500ml  |
| 20:00 | 250ml  |

Daily target: 2000ml

### Reset
- `/reset meals` - Clear today's meal logs
- `/reset water` - Clear today's water logs
- `/reset all` - Delete all your data (meals, water, profile, limits)

### Examples

**Set up your profile:**
```
/profile 180 75 28 male
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
- Data is stored in SQLite. On Render free tier, data resets on redeploy.
- Set your profile first with `/profile` for personalized calorie and macro recommendations.
