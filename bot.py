import json
import logging
import os
import sys

from openai import OpenAI
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
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
PORT = int(os.environ.get("PORT", "10000"))

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are a nutrition assistant. Given a meal description, estimate calories "
    "for each item and provide a total. Return ONLY valid JSON with no extra text: "
    '{"items": [{"name": "<food>", "calories": <int>}], "total": <int>}. '
    "Use reasonable portion sizes. If a description is unclear, make your best "
    "estimate and note assumptions in a short 'note' field."
)


def estimate_calories(meal_text: str) -> dict:
    """Send meal description to OpenAI and return parsed JSON."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meal_text},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def format_reply(data: dict) -> str:
    """Turn the JSON calorie data into a readable Telegram message."""
    lines = ["\U0001f37d Calorie Estimate:"]
    for item in data.get("items", []):
        lines.append(f"- {item['name']}: ~{item['calories']} kcal")
    lines.append(f"\nTotal: ~{data['total']} kcal")
    note = data.get("note")
    if note:
        lines.append(f"\n({note})")
    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a description of what you ate and I'll estimate the calories.\n\n"
        "Example: '2 scrambled eggs, 1 toast with butter, black coffee'"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just type what you ate in plain text. I'll reply with a calorie estimate.\n\n"
        "Examples:\n"
        "- 'bowl of oatmeal with banana'\n"
        "- 'big mac, medium fries, diet coke'\n"
        "- 'chicken salad with olive oil dressing'"
    )


async def handle_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any text message as a meal description."""
    meal_text = update.message.text
    if not meal_text:
        return

    try:
        data = estimate_calories(meal_text)
        reply = format_reply(data)
    except json.JSONDecodeError:
        logger.exception("Failed to parse OpenAI response")
        reply = "Sorry, I couldn't parse the calorie data. Try rephrasing your meal."
    except Exception:
        logger.exception("Error estimating calories")
        reply = "Something went wrong. Please try again in a moment."

    await update.message.reply_text(reply)


def main() -> None:
    use_polling = "--poll" in sys.argv

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_meal))

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
