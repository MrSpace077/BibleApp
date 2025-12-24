#!/usr/bin/env python3
"""
Bible Verse Search - Find Bible verses based on topics, questions, or keywords.
Uses semantic search with OpenAI to understand what you're looking for.
"""

import os
import sys
import json
import requests
from typing import Optional
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

load_dotenv()

# Bible API endpoints
BIBLE_API_URL = "https://bible-api.com"


class BibleSearch:
    """Bible verse search using semantic understanding."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.openai_client = None

        # Initialize OpenAI if available and configured
        api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def print(self, text: str, style: str = None):
        """Print text with optional styling."""
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    def print_verse(self, reference: str, text: str, translation: str = "WEB"):
        """Display a verse in a formatted way."""
        if self.console:
            panel = Panel(
                text.strip(),
                title=f"[bold blue]{reference}[/bold blue]",
                subtitle=f"[dim]{translation}[/dim]",
                border_style="green"
            )
            self.console.print(panel)
        else:
            print(f"\n{'='*60}")
            print(f"📖 {reference} ({translation})")
            print(f"{'='*60}")
            print(text.strip())
            print(f"{'='*60}\n")

    def print_error(self, message: str):
        """Display an error message."""
        if self.console:
            self.console.print(f"[bold red]{message}[/bold red]")
        else:
            print(f"❌ {message}")

    def fetch_verse(self, reference: str, translation: str = "web") -> Optional[dict]:
        """
        Fetch a verse from the Bible API.

        Args:
            reference: Bible reference like "John 3:16" or "Matthew 5:1-12"
            translation: Translation code (web, kjv, etc.)

        Returns:
            Dictionary with verse data or None if not found
        """
        try:
            reference = reference.strip()
            url = f"{BIBLE_API_URL}/{reference}"
            params = {"translation": translation}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Check if verse actually exists
                if "error" not in data and data.get("text"):
                    return data
                return None
            else:
                return None
        except Exception:
            return None

    def search_single_with_ai(self, query: str) -> Optional[dict]:
        """
        Use OpenAI to find a single relevant Bible verse.

        Args:
            query: Natural language query about a Bible verse

        Returns:
            Single verse reference with explanation, or None
        """
        if not self.openai_client:
            return self._fallback_search_single(query)

        try:
            system_prompt = """You are a Bible expert assistant. When given a question or topic,
you identify the SINGLE most relevant Bible verse.

Respond ONLY with a JSON object containing:
- "reference": The Bible reference (e.g., "Matthew 7:12", "John 3:16")
- "reason": Brief explanation of why this verse is relevant

For specific verse queries (like "the golden rule"), give the exact verse.
If you cannot find a single specific verse that matches, respond with: {"reference": null, "reason": "No specific verse found"}

Example response:
{"reference": "Matthew 7:12", "reason": "This is the Golden Rule - treat others as you want to be treated"}

Respond with ONLY the JSON object, no other text."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            if result.get("reference"):
                return result
            return None

        except Exception:
            return self._fallback_search_single(query)

    def search_multiple_with_ai(self, query: str) -> list[dict]:
        """
        Use OpenAI to find multiple relevant Bible verses.

        Args:
            query: Natural language query about Bible verses

        Returns:
            List of verse references with explanations
        """
        if not self.openai_client:
            return self._fallback_search_multiple(query)

        try:
            system_prompt = """You are a Bible expert assistant. When given a question or topic,
you identify multiple relevant Bible verses.

Respond ONLY with a JSON array of objects, each containing:
- "reference": The Bible reference (e.g., "Matthew 7:12", "John 3:16-17")
- "reason": Brief explanation of why this verse is relevant

Include 2-10 relevant verses depending on the topic.
If you cannot find any relevant verses, respond with: []

Example response:
[{"reference": "Revelation 21:1-4", "reason": "New heaven and new earth"}, {"reference": "Revelation 21:10-14", "reason": "Description of New Jerusalem"}]

Respond with ONLY the JSON array, no other text."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            verses = json.loads(content)
            return verses if verses else []

        except Exception:
            return self._fallback_search_multiple(query)

    def _fallback_search_single(self, query: str) -> Optional[dict]:
        """Fallback single verse search using keyword database."""
        # Single verse database for common specific queries
        single_verse_database = {
            "golden rule": {"reference": "Matthew 7:12", "reason": "The Golden Rule - do unto others"},
            "john 3:16": {"reference": "John 3:16", "reason": "God's love for the world"},
            "god so loved": {"reference": "John 3:16", "reason": "God's love for the world"},
            "beginning": {"reference": "Genesis 1:1", "reason": "In the beginning God created"},
            "faith definition": {"reference": "Hebrews 11:1", "reason": "Definition of faith"},
            "what is faith": {"reference": "Hebrews 11:1", "reason": "Definition of faith"},
            "love is patient": {"reference": "1 Corinthians 13:4-7", "reason": "Love is patient, love is kind"},
            "lord is my shepherd": {"reference": "Psalm 23:1", "reason": "The Lord is my shepherd"},
            "i can do all things": {"reference": "Philippians 4:13", "reason": "I can do all things through Christ"},
            "fear not": {"reference": "Isaiah 41:10", "reason": "Fear not, for I am with you"},
            "plans for you": {"reference": "Jeremiah 29:11", "reason": "Plans for hope and a future"},
            "jeremiah 29:11": {"reference": "Jeremiah 29:11", "reason": "Plans for hope and a future"},
            "trust in the lord": {"reference": "Proverbs 3:5-6", "reason": "Trust in the Lord with all your heart"},
            "way truth life": {"reference": "John 14:6", "reason": "Jesus is the way, truth, and life"},
            "do not worry": {"reference": "Matthew 6:25", "reason": "Do not worry about your life"},
            "be strong": {"reference": "Joshua 1:9", "reason": "Be strong and courageous"},
            "all things work together": {"reference": "Romans 8:28", "reason": "All things work together for good"},
            "grace sufficient": {"reference": "2 Corinthians 12:9", "reason": "My grace is sufficient for you"},
            "god is love": {"reference": "1 John 4:8", "reason": "God is love"},
            "wages of sin": {"reference": "Romans 6:23", "reason": "Wages of sin is death"},
            "all have sinned": {"reference": "Romans 3:23", "reason": "All have sinned"},
            "saved by grace": {"reference": "Ephesians 2:8-9", "reason": "Saved by grace through faith"},
            "armor of god": {"reference": "Ephesians 6:10-18", "reason": "Put on the full armor of God"},
            "fruit of the spirit": {"reference": "Galatians 5:22-23", "reason": "Love, joy, peace, patience..."},
            "mustard seed": {"reference": "Matthew 17:20", "reason": "Faith like a mustard seed"},
            "lord's prayer": {"reference": "Matthew 6:9-13", "reason": "Our Father in heaven..."},
            "our father": {"reference": "Matthew 6:9-13", "reason": "The Lord's Prayer"},
            "resurrection and life": {"reference": "John 11:25-26", "reason": "I am the resurrection and the life"},
            "good shepherd": {"reference": "John 10:11", "reason": "I am the good shepherd"},
            "light of the world": {"reference": "John 8:12", "reason": "I am the light of the world"},
            "bread of life": {"reference": "John 6:35", "reason": "I am the bread of life"},
        }

        query_lower = query.lower()

        for keyword, verse in single_verse_database.items():
            if keyword in query_lower:
                return verse

        return None

    def _fallback_search_multiple(self, query: str) -> list[dict]:
        """Fallback multiple verse search using keyword database."""
        verse_database = {
            "love": [
                {"reference": "1 Corinthians 13:4-7", "reason": "Love is patient, love is kind"},
                {"reference": "John 3:16", "reason": "God's love for the world"},
                {"reference": "1 John 4:8", "reason": "God is love"},
                {"reference": "Romans 8:38-39", "reason": "Nothing can separate us from God's love"}
            ],
            "faith": [
                {"reference": "Hebrews 11:1", "reason": "Definition of faith"},
                {"reference": "Romans 10:17", "reason": "Faith comes from hearing"},
                {"reference": "James 2:17", "reason": "Faith without works is dead"},
                {"reference": "Matthew 17:20", "reason": "Faith like a mustard seed"}
            ],
            "hope": [
                {"reference": "Romans 15:13", "reason": "God of hope"},
                {"reference": "Jeremiah 29:11", "reason": "Plans for hope and a future"},
                {"reference": "Romans 8:28", "reason": "All things work together for good"}
            ],
            "strength": [
                {"reference": "Philippians 4:13", "reason": "I can do all things through Christ"},
                {"reference": "Isaiah 40:31", "reason": "Those who hope in the Lord will renew their strength"},
                {"reference": "Psalm 46:1", "reason": "God is our refuge and strength"}
            ],
            "fear": [
                {"reference": "Isaiah 41:10", "reason": "Fear not, for I am with you"},
                {"reference": "2 Timothy 1:7", "reason": "God has not given us a spirit of fear"},
                {"reference": "Psalm 23:4", "reason": "I will fear no evil"}
            ],
            "peace": [
                {"reference": "Philippians 4:6-7", "reason": "Peace that surpasses understanding"},
                {"reference": "John 14:27", "reason": "My peace I give to you"},
                {"reference": "Isaiah 26:3", "reason": "Perfect peace for those who trust"}
            ],
            "salvation": [
                {"reference": "Romans 10:9-10", "reason": "Confession and belief for salvation"},
                {"reference": "Ephesians 2:8-9", "reason": "Saved by grace through faith"},
                {"reference": "John 14:6", "reason": "Jesus is the way, truth, and life"},
                {"reference": "Acts 4:12", "reason": "No other name for salvation"}
            ],
            "new jerusalem": [
                {"reference": "Revelation 21:1-4", "reason": "New heaven, new earth, New Jerusalem"},
                {"reference": "Revelation 21:10-14", "reason": "Description of the New Jerusalem"},
                {"reference": "Revelation 21:18-21", "reason": "The walls and gates of New Jerusalem"},
                {"reference": "Revelation 21:22-27", "reason": "No temple needed, God is its light"},
                {"reference": "Revelation 22:1-5", "reason": "River of life in New Jerusalem"}
            ],
            "revelation": [
                {"reference": "Revelation 1:1-3", "reason": "Introduction to Revelation"},
                {"reference": "Revelation 21:1-4", "reason": "New heaven and new earth"},
                {"reference": "Revelation 22:12-13", "reason": "Jesus is coming soon"}
            ],
            "creation": [
                {"reference": "Genesis 1:1", "reason": "In the beginning God created"},
                {"reference": "John 1:1-3", "reason": "All things made through the Word"},
                {"reference": "Colossians 1:16", "reason": "All things created by and for Christ"}
            ],
            "prayer": [
                {"reference": "Matthew 6:9-13", "reason": "The Lord's Prayer"},
                {"reference": "Philippians 4:6-7", "reason": "Prayer with thanksgiving"},
                {"reference": "1 Thessalonians 5:17", "reason": "Pray without ceasing"},
                {"reference": "James 5:16", "reason": "Prayer of a righteous person"}
            ],
            "forgiveness": [
                {"reference": "1 John 1:9", "reason": "God is faithful to forgive"},
                {"reference": "Ephesians 4:32", "reason": "Forgive as God forgave you"},
                {"reference": "Matthew 6:14-15", "reason": "Forgive others to be forgiven"},
                {"reference": "Colossians 3:13", "reason": "Bear with and forgive one another"}
            ],
            "wisdom": [
                {"reference": "James 1:5", "reason": "Ask God for wisdom"},
                {"reference": "Proverbs 9:10", "reason": "Fear of the Lord is beginning of wisdom"},
                {"reference": "Proverbs 3:5-6", "reason": "Trust in the Lord, don't lean on own understanding"}
            ],
            "comfort": [
                {"reference": "Psalm 23:4", "reason": "You are with me in the valley"},
                {"reference": "Matthew 5:4", "reason": "Blessed are those who mourn"},
                {"reference": "2 Corinthians 1:3-4", "reason": "God of all comfort"},
                {"reference": "Revelation 21:4", "reason": "God will wipe away every tear"}
            ],
            "trust": [
                {"reference": "Proverbs 3:5-6", "reason": "Trust in the Lord with all your heart"},
                {"reference": "Psalm 56:3", "reason": "When I am afraid, I put my trust in you"},
                {"reference": "Isaiah 26:3-4", "reason": "Trust in the Lord forever"}
            ],
            "children": [
                {"reference": "Proverbs 22:6", "reason": "Train up a child in the way"},
                {"reference": "Matthew 19:14", "reason": "Let the little children come to me"},
                {"reference": "Psalm 127:3", "reason": "Children are a heritage from the Lord"}
            ],
            "marriage": [
                {"reference": "Genesis 2:24", "reason": "A man shall leave and cleave to his wife"},
                {"reference": "Ephesians 5:25", "reason": "Husbands love your wives"},
                {"reference": "1 Corinthians 13:4-7", "reason": "Love is patient, love is kind"}
            ],
            "death": [
                {"reference": "John 11:25-26", "reason": "I am the resurrection and the life"},
                {"reference": "1 Thessalonians 4:13-14", "reason": "Hope for those who have died"},
                {"reference": "Philippians 1:21", "reason": "To live is Christ, to die is gain"},
                {"reference": "Revelation 21:4", "reason": "No more death or mourning"}
            ],
            "heaven": [
                {"reference": "John 14:2-3", "reason": "In my Father's house are many rooms"},
                {"reference": "Revelation 21:1-4", "reason": "New heaven and new earth"},
                {"reference": "Philippians 3:20", "reason": "Our citizenship is in heaven"}
            ],
            "sin": [
                {"reference": "Romans 3:23", "reason": "All have sinned"},
                {"reference": "Romans 6:23", "reason": "Wages of sin is death"},
                {"reference": "1 John 1:8-9", "reason": "If we confess our sins"}
            ],
            "grace": [
                {"reference": "Ephesians 2:8-9", "reason": "Saved by grace through faith"},
                {"reference": "2 Corinthians 12:9", "reason": "My grace is sufficient for you"},
                {"reference": "Romans 5:8", "reason": "God demonstrates His love while we were sinners"}
            ],
            "healing": [
                {"reference": "James 5:14-15", "reason": "Prayer of faith will heal the sick"},
                {"reference": "Isaiah 53:5", "reason": "By his wounds we are healed"},
                {"reference": "Psalm 103:2-3", "reason": "He heals all your diseases"},
                {"reference": "Jeremiah 17:14", "reason": "Heal me, Lord, and I will be healed"}
            ],
            "anxiety": [
                {"reference": "Philippians 4:6-7", "reason": "Do not be anxious about anything"},
                {"reference": "1 Peter 5:7", "reason": "Cast all your anxiety on him"},
                {"reference": "Matthew 6:25-34", "reason": "Do not worry about tomorrow"}
            ],
            "anger": [
                {"reference": "Ephesians 4:26-27", "reason": "Be angry and do not sin"},
                {"reference": "James 1:19-20", "reason": "Slow to anger"},
                {"reference": "Proverbs 15:1", "reason": "A gentle answer turns away wrath"}
            ],
            "temptation": [
                {"reference": "1 Corinthians 10:13", "reason": "God provides a way out"},
                {"reference": "James 1:12-15", "reason": "Blessed is the one who perseveres"},
                {"reference": "Matthew 26:41", "reason": "Watch and pray so you don't fall"}
            ],
            "patience": [
                {"reference": "James 5:7-8", "reason": "Be patient until the Lord's coming"},
                {"reference": "Romans 12:12", "reason": "Be patient in tribulation"},
                {"reference": "Galatians 5:22-23", "reason": "Fruit of the Spirit includes patience"}
            ],
            "joy": [
                {"reference": "Nehemiah 8:10", "reason": "The joy of the Lord is your strength"},
                {"reference": "Philippians 4:4", "reason": "Rejoice in the Lord always"},
                {"reference": "Psalm 16:11", "reason": "Fullness of joy in God's presence"}
            ],
            "beatitudes": [
                {"reference": "Matthew 5:3-12", "reason": "Blessed are the poor in spirit..."}
            ],
            "ten commandments": [
                {"reference": "Exodus 20:1-17", "reason": "The Ten Commandments"},
                {"reference": "Deuteronomy 5:6-21", "reason": "Ten Commandments restated"}
            ],
            "christmas": [
                {"reference": "Luke 2:1-20", "reason": "Birth of Jesus narrative"},
                {"reference": "Matthew 1:18-25", "reason": "Joseph's dream, Jesus' birth"},
                {"reference": "Isaiah 9:6", "reason": "For unto us a child is born"}
            ],
            "easter": [
                {"reference": "Matthew 28:1-10", "reason": "The resurrection of Jesus"},
                {"reference": "1 Corinthians 15:3-8", "reason": "Christ died and was raised"},
                {"reference": "John 20:1-18", "reason": "Mary at the empty tomb"}
            ],
            "resurrection": [
                {"reference": "John 11:25-26", "reason": "I am the resurrection and the life"},
                {"reference": "1 Corinthians 15:20-22", "reason": "Christ the firstfruits"},
                {"reference": "Romans 6:5", "reason": "United with him in resurrection"}
            ]
        }

        query_lower = query.lower()

        for keyword, verses in verse_database.items():
            if keyword in query_lower:
                return verses

        return []

    def search_single(self, query: str) -> bool:
        """
        Search for a single verse. If not found, falls back to multiple verses.

        Args:
            query: Natural language query

        Returns:
            True if verse(s) found, False otherwise
        """
        self.print(f"\n🔍 Searching for single verse: [bold cyan]{query}[/bold cyan]\n" if RICH_AVAILABLE else f"\n🔍 Searching for single verse: {query}\n")

        # Try to find a single verse
        result = self.search_single_with_ai(query)

        if result:
            reference = result.get("reference")
            reason = result.get("reason", "")

            verse_data = self.fetch_verse(reference)

            if verse_data:
                text = verse_data.get("text", "Verse text not available")
                translation = verse_data.get("translation_name", "WEB")

                self.print(f"[dim]({reason})[/dim]" if RICH_AVAILABLE else f"({reason})")
                self.print_verse(reference, text, translation)
                return True
            else:
                # Verse reference found but couldn't fetch - still show reference
                self.print(f"[dim]({reason})[/dim]" if RICH_AVAILABLE else f"({reason})")
                self.print(f"\n📖 [bold]{reference}[/bold]" if RICH_AVAILABLE else f"\n📖 {reference}")
                self.print("[yellow](Could not fetch verse text - check your internet connection)[/yellow]" if RICH_AVAILABLE else "(Could not fetch verse text)")
                return True

        # No single verse found, try multiple verses
        self.print("[yellow]No single verse found. Searching for related verses...[/yellow]\n" if RICH_AVAILABLE else "No single verse found. Searching for related verses...\n")
        return self.search_multiple(query, from_fallback=True)

    def search_multiple(self, query: str, from_fallback: bool = False) -> bool:
        """
        Search for multiple verses on a topic.

        Args:
            query: Natural language query
            from_fallback: Whether this was called as fallback from single search

        Returns:
            True if verses found, False otherwise
        """
        if not from_fallback:
            self.print(f"\n🔍 Searching for verses about: [bold cyan]{query}[/bold cyan]\n" if RICH_AVAILABLE else f"\n🔍 Searching for verses about: {query}\n")

        # Find multiple verses
        results = self.search_multiple_with_ai(query)

        if not results:
            self.print_error("Verse not found, please try again later")
            return False

        self.print(f"Found [green]{len(results)}[/green] relevant verse(s):\n" if RICH_AVAILABLE else f"Found {len(results)} relevant verse(s):\n")

        for result in results:
            reference = result.get("reference", "Unknown")
            reason = result.get("reason", "")

            verse_data = self.fetch_verse(reference)

            if verse_data:
                text = verse_data.get("text", "Verse text not available")
                translation = verse_data.get("translation_name", "WEB")

                self.print(f"[dim]({reason})[/dim]" if RICH_AVAILABLE else f"({reason})")
                self.print_verse(reference, text, translation)
            else:
                self.print(f"\n📖 [bold]{reference}[/bold]" if RICH_AVAILABLE else f"\n📖 {reference}")
                self.print(f"   [dim]{reason}[/dim]" if RICH_AVAILABLE else f"   {reason}")
                self.print("[yellow](Could not fetch verse text)[/yellow]" if RICH_AVAILABLE else "(Could not fetch verse text)")

        return True


def main():
    """Main entry point for the Bible search CLI."""
    bible = BibleSearch()

    # Check for command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ("-s", "--single") and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            bible.search_single(query)
            return
        elif mode in ("-m", "--multiple") and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            bible.search_multiple(query)
            return
        else:
            # Default to single search with fallback
            query = " ".join(sys.argv[1:])
            bible.search_single(query)
            return

    # Interactive mode
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel(
            "[bold]Bible Verse Search[/bold]\n\n"
            "Find Bible verses by asking questions or describing topics.\n\n"
            "[bold cyan]Two Search Modes:[/bold cyan]\n"
            "  [green]1[/green] - Single Verse Search (finds one specific verse)\n"
            "  [green]2[/green] - Multiple Verses Search (finds a group of verses)\n\n"
            "[bold]Examples:[/bold]\n"
            "  • What is the golden rule? [dim](single)[/dim]\n"
            "  • New Jerusalem in Revelation [dim](multiple)[/dim]\n"
            "  • Verses about love [dim](multiple)[/dim]\n\n"
            "Type [cyan]'quit'[/cyan] or [cyan]'exit'[/cyan] to stop.",
            title="📖 Welcome",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print("📖 Bible Verse Search")
        print("="*60)
        print("\nFind Bible verses by asking questions or describing topics.")
        print("\nTwo Search Modes:")
        print("  1 - Single Verse Search (finds one specific verse)")
        print("  2 - Multiple Verses Search (finds a group of verses)")
        print("\nExamples:")
        print("  • What is the golden rule? (single)")
        print("  • New Jerusalem in Revelation (multiple)")
        print("  • Verses about love (multiple)")
        print("\nType 'quit' or 'exit' to stop.")
        print("="*60)

    while True:
        try:
            # Get search mode
            print()
            if RICH_AVAILABLE:
                mode = console.input("[bold yellow]Select mode ([green]1[/green]=Single, [green]2[/green]=Multiple):[/bold yellow] ")
            else:
                mode = input("Select mode (1=Single, 2=Multiple): ")

            mode = mode.strip()

            if mode.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 📖\n")
                break

            if mode not in ("1", "2"):
                if RICH_AVAILABLE:
                    console.print("[red]Please enter 1 or 2[/red]")
                else:
                    print("Please enter 1 or 2")
                continue

            # Get search query
            if RICH_AVAILABLE:
                if mode == "1":
                    query = console.input("[bold green]Enter your search (single verse):[/bold green] ")
                else:
                    query = console.input("[bold green]Enter your search (multiple verses):[/bold green] ")
            else:
                if mode == "1":
                    query = input("Enter your search (single verse): ")
                else:
                    query = input("Enter your search (multiple verses): ")

            query = query.strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 📖\n")
                break

            # Perform search based on mode
            if mode == "1":
                bible.search_single(query)
            else:
                bible.search_multiple(query)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 📖\n")
            break
        except EOFError:
            print("\n\nGoodbye! 📖\n")
            break


if __name__ == "__main__":
    main()
