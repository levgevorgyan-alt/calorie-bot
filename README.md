# Calorie Bot

A Telegram bot that estimates calories from meal descriptions, tracks daily
intake per user, enforces configurable calorie limits, logs water intake,
and sends scheduled water reminders.

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

Send any meal description to the bot (DM or group chat):

```
2 scrambled eggs, toast with butter, black coffee
```

The bot replies with a calorie breakdown and daily progress:

```
🍽 Calorie Estimate:
- 2 scrambled eggs: ~180 kcal
- Toast with butter: ~120 kcal
- Black coffee: ~2 kcal

Total: ~302 kcal
📊 Today so far: 302 / 1800 kcal (1498 remaining)
```

## Commands

### Calories
- `/today` - Show today's meals and calorie total
- `/week` - 7-day calorie summary
- `/setlimit <kcal>` - Set your daily calorie limit (default: 1800)
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
- `/reset all` - Delete all your data (meals, water, limit resets to default)

### Examples

**Log a meal** (just type it):
```
chicken salad with rice
2 eggs, toast with butter, black coffee
big mac, medium fries, diet coke
```

**Set a calorie limit:**
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
