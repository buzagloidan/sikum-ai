# Sikum.AI - Automated Quiz Generator Bot

Sikum.AI is a Telegram bot that automatically generates quizzes from uploaded documents, helping students learn and test their knowledge.

## Try It Now!

You can try the live version of the bot at:
**[sikum.buzagloidan.com](https://sikum.buzagloidan.com)**

## Screenshots

For screenshots of the bot in action, please visit:
**[sikum.buzagloidan.com](https://sikum.buzagloidan.com)**

## Features

- 📚 **Document Processing**: Supports PDF, DOCX, PPTX, and TXT files
- 🧠 **AI-Powered Quizzes**: Uses Google's Gemini API to generate relevant questions from document content
- 💾 **Quiz Saving**: Save your favorite quizzes for later use
- 🔄 **Quiz Sharing**: Share quizzes with friends or classmates
- 📊 **Statistics**: Track usage statistics and performance
- 🔒 **Rate Limiting**: Free and premium usage tiers

## Setup

### Prerequisites

- Python 3.10+
- Telegram Bot token (from [@BotFather](https://t.me/BotFather))
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/buzagloidan/sikum-ai.git
cd sikum-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on the example:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your credentials:
```
BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
ADMIN_IDS=your_admin_id
```

5. Create the required directories:
```bash
mkdir -p temp logs
```

### Running the Bot

```bash
python sikum.py
```

## Usage

1. Start the bot with `/start`
2. Upload a document (PDF, DOCX, PPTX, or TXT)
3. The bot will generate a quiz based on the document content
4. Answer the questions and get immediate feedback
5. Save quizzes with `/save` for future reference
6. List saved quizzes with `/list`
7. Share quizzes with `/share`

## Admin Commands

- `/stats` - View system statistics
- `/grant` - Grant premium access to users

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 