# Bible Verse Search

Find Bible verses based on topics, questions, or keywords. Ask natural language questions like "What is the golden rule?" and get the exact verse(s) with full text.

## Features

- **Semantic Search**: Uses AI to understand what you're looking for (requires OpenAI API key)
- **Fallback Search**: Works without an API key using a built-in database of common verses
- **Full Verse Display**: Fetches and displays the complete verse text
- **Multiple Results**: Returns multiple relevant verses for broader topics
- **Interactive Mode**: Run queries interactively or via command line

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MrSpace077/BibleApp.git
   cd BibleApp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up OpenAI API for semantic search:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

## Usage

### Interactive Mode

```bash
python bible_search.py
```

Then type your questions:
```
Search: What is the golden rule?
Search: Verses about love
Search: New Jerusalem in Revelation
```

### Command Line Mode

```bash
python bible_search.py "What is the golden rule?"
python bible_search.py "verses about love and patience"
python bible_search.py "new jerusalem revelation"
```

## Examples

**Finding a specific verse:**
```
Search: What is the golden rule?
```
Output:
```
📖 Matthew 7:12 (WEB)
Therefore whatever you desire for men to do to you, you shall also do to them;
for this is the law and the prophets.
```

**Finding multiple verses on a topic:**
```
Search: New Jerusalem in Revelation
```
Output:
```
📖 Revelation 21:1-4 (WEB)
I saw a new heaven and a new earth...

📖 Revelation 21:10-14 (WEB)
He carried me away in the Spirit to a great and high mountain...

📖 Revelation 21:18-21 (WEB)
The construction of its wall was jasper...
```

## Translation

The app uses the World English Bible (WEB) translation by default, which is in the public domain. The semantic search feature can identify verses from any translation and will display them in WEB.

## How It Works

1. **With OpenAI API**: Your query is processed by GPT-4o-mini to identify the most relevant Bible verses based on semantic understanding.

2. **Without OpenAI API**: The app uses a built-in database of common verses organized by topic (love, faith, hope, fear, etc.) and matches your query against keywords.

## Requirements

- Python 3.8+
- Internet connection (to fetch verse text)
- Optional: OpenAI API key for semantic search

## License

MIT License
