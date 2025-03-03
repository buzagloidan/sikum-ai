import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import ContextTypes
from telegram.error import BadRequest

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import constants
from sikum import (
    STATE_QUIZ_ACTIVE,
    PREMIUM_USES_PER_DAY,
    FREE_USES_PER_DAY,
    start,
    help_command,
    subscribe_command,
    support_command,
    list_saved_quizzes,
    send_question,
    start_saved_quiz,
    share_quiz
)

async def handle_button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks from navigation buttons and actions."""
    query = update.callback_query
    
    try:
        # Safety check
        if not query:
            return
            
        await query.answer()  # Acknowledge the button click
        
        # Handle navigation and common actions buttons
        if query.data == "start":
            # Return to start menu
            await query.message.delete()
            await start(update, context)
            return
            
        if query.data == "help":
            # Show help
            await query.message.delete()
            await help_command(update, context)
            return
            
        if query.data == "list_quizzes":
            # Show saved quizzes
            await query.message.delete()
            await list_saved_quizzes(update, context)
            return
            
        if query.data == "subscribe":
            # Go to subscription page
            await query.message.delete()
            await subscribe_command(update, context)
            return
            
        if query.data == "support":
            # Contact support
            await query.message.delete()
            await support_command(update, context)
            return
            
        if query.data == "start_quiz":
            # Start quiz after document processing
            await query.message.delete()
            await send_question(update, context)
            return
            
        if query.data == "cancel_quiz":
            # Cancel quiz
            context.user_data.pop("randomized_questions", None)
            context.user_data.pop("current_question_index", None)
            context.user_data.pop("correct_answers", None)
            context.user_data.pop(STATE_QUIZ_ACTIVE, None)
            
            await query.edit_message_text("❌ המבחן בוטל. שלח קובץ חדש כדי להתחיל שוב.")
            return
            
        # Handle subscription options
        if query.data == "sub_monthly":
            user_id = update.effective_user.id
            await query.edit_message_text("מעבר לתשלום...")
            
            # Create the invoice for monthly subscription
            await context.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title="מנוי חודשי פרימיום",
                description=f"מנוי חודשי לסיכום.AI עם {PREMIUM_USES_PER_DAY} ניסיונות ביום",
                payload=f"premium_monthly_{user_id}",
                provider_token="",  # Empty for Telegram Stars
                currency="XTR",    # XTR is the currency code for Telegram Stars
                prices=[LabeledPrice(label="מנוי חודשי", amount=100)],
                need_name=False,
                need_phone_number=False,
                need_email=False,
                need_shipping_address=False,
                is_flexible=False
            )
            return
            
        if query.data == "sub_yearly":
            user_id = update.effective_user.id
            await query.edit_message_text("מעבר לתשלום...")
            
            # Create the invoice for yearly subscription
            await context.bot.send_invoice(
                chat_id=update.effective_chat.id,
                title="מנוי שנתי פרימיום",
                description=f"מנוי שנתי לסיכום.AI עם {PREMIUM_USES_PER_DAY} ניסיונות ביום",
                payload=f"premium_yearly_{user_id}",
                provider_token="",  # Empty for Telegram Stars
                currency="XTR",    # XTR is the currency code for Telegram Stars
                prices=[LabeledPrice(label="מנוי שנתי", amount=900)],
                need_name=False,
                need_phone_number=False,
                need_email=False,
                need_shipping_address=False,
                is_flexible=False
            )
            return
            
        # Handle direct actions with quiz IDs
        if query.data.startswith("play_"):
            quiz_id = int(query.data.split("_")[1])
            await query.message.delete()
            await start_saved_quiz(update, context, quiz_id)
            return
            
        if query.data.startswith("share_"):
            quiz_id = int(query.data.split("_")[1])
            await share_quiz(update, context, quiz_id)
            return
        
        # If no handler matched, pass to the existing handle_answer function
        return False
        
    except Exception as e:
        logger.error(f"Error handling button click: {e}")
        await query.edit_message_text("⚠️ אירעה שגיאה. אנא נסה שוב.")
        return True 