async def save_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save the current quiz for future replay."""
    user_id = update.effective_user.id

    # Check if there's an active or completed quiz
    if not (context.user_data.get(STATE_QUIZ_ACTIVE) or context.user_data.get("quiz_completed")):
        await update.message.reply_text("אין מבחן פעיל לשמירה. צור קודם מבחן חדש!")
        return
    
    questions = context.user_data.get("randomized_questions", [])
    if not questions:
        await update.message.reply_text("לא נמצאו שאלות לשמירה!")
        return

    # If title was provided with command, save directly
    if context.args:
        title = " ".join(context.args)
        if validate_quiz_title(title):
            await save_quiz_with_title(update, context, title)
        else:
            await update.message.reply_text(
                "❌ שם המבחן אינו תקין. אנא בחר שם באורך של 1-100 תווים."
            )
    else:
        # Otherwise, ask for title interactively
        await update.message.reply_text("אנא שלח שם למבחן (או /cancel לביטול)")
        context.user_data['waiting_for_quiz_title'] = True
        context.user_data['quiz_to_save'] = questions 