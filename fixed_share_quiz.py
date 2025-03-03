async def share_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE, quiz_id: int) -> None:
    """Share a saved quiz with others via a shareable link."""
    user_id = update.effective_user.id
    
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # First check if quiz exists
        c.execute('SELECT 1 FROM saved_quizzes WHERE quiz_id = ?', (quiz_id,))
        if not c.fetchone():
            # Handle response based on whether it's a callback query or direct message
            if update.callback_query:
                await update.callback_query.edit_message_text("❌ מבחן זה לא נמצא.")
            else:
                await update.message.reply_text("❌ מבחן זה לא נמצא.")
            return
            
    share_link = f"https://t.me/{context.bot.username}?start=quiz_{quiz_id}"
    
    # Handle response based on whether it's a callback query or direct message
    if update.callback_query:
        await update.callback_query.edit_message_text(
            f"🔗 הנה הקישור לשיתוף המבחן:\n{share_link}",
            disable_web_page_preview=True
        )
    else:
        await update.message.reply_text(
            f"🔗 קישור לשיתוף המבחן:\n{share_link}",
            disable_web_page_preview=True
        ) 