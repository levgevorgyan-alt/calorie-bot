# Calorie Bot

A Telegram bot that estimates calories from meal descriptions using OpenAI.

## Prerequisites

1. Create a Telegram bot via [@BotFather](https://t.me/BotFather) and save the token.
2. Get an OpenAI API key from [platform.openai.com](https://platform.openai.com).

## Local Development

```bash
pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="your-token"
export OPENAI_API_KEY="your-key"

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
   - `OPENAI_API_KEY` - your OpenAI key
   - `WEBHOOK_URL` - the URL Render assigns (e.g. `https://calorie-bot.onrender.com`)
7. Click **Deploy**.

The bot will automatically register its webhook with Telegram on startup.

## Usage

Send any meal description to the bot (DM or group chat):

```
2 scrambled eggs, toast with butter, black coffee
```

The bot replies with a calorie breakdown:

```
Calorie Estimate:
- 2 scrambled eggs: ~180 kcal
- 1 slice toast with butter: ~120 kcal
- Black coffee: ~2 kcal

Total: ~302 kcal
```

## Commands

- `/start` - Welcome message
- `/help` - Usage instructions

## Notes

- Render free tier spins down after 15 min of inactivity. First message after idle has a ~30s cold start.
- In group chats, configure BotFather's "Group Privacy" setting to control whether the bot sees all messages or only those mentioning it.
