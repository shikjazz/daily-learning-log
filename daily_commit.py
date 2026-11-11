#!/usr/bin/env python3
"""
Daily GitHub Contribution Script
---------------------------------
Makes a small, meaningful daily commit to keep the contribution graph active.
Designed to run via GitHub Actions cron job.

What it does:
- Updates a contributions log with a daily entry
- Rotates tips/quotes related to CS/NLP/ML
- Keeps a streak counter
"""

import json
import random
import datetime
import os
from pathlib import Path

# --- Configuration ---
LOG_FILE = "contributions/log.json"
README_FILE = "README.md"

TOPICS = [
    "Reviewed NLP pipeline optimization techniques",
    "Explored transformer attention mechanisms",
    "Studied reinforcement learning reward shaping",
    "Practiced Python async patterns",
    "Read about knowledge graph embeddings",
    "Explored multilingual NLP approaches",
    "Reviewed machine translation evaluation metrics",
    "Studied computational linguistics formalisms",
    "Practiced data structure implementations",
    "Explored graph neural network architectures",
    "Reviewed information retrieval techniques",
    "Studied text summarization approaches",
    "Practiced algorithm complexity analysis",
    "Explored sentiment analysis methods",
    "Reviewed neural network regularization",
    "Studied word embedding techniques",
    "Practiced SQL query optimization",
    "Explored named entity recognition",
    "Reviewed sequence-to-sequence models",
    "Studied cross-lingual transfer learning",
    "Practiced REST API design patterns",
    "Explored topic modeling algorithms",
    "Reviewed dependency parsing methods",
    "Studied generative language models",
    "Practiced Docker containerization",
    "Explored speech recognition systems",
    "Reviewed text classification pipelines",
    "Studied question answering systems",
    "Practiced Git branching strategies",
    "Explored dialogue system architectures",
    "Reviewed coreference resolution techniques",
]

TIPS = [
    "Tip: Use `torch.no_grad()` during inference to save memory.",
    "Tip: Pre-tokenize large datasets offline for faster training.",
    "Tip: Profile your code with `cProfile` before optimizing.",
    "Tip: Use type hints -- future you will thank present you.",
    "Tip: Write tests before refactoring legacy code.",
    "Tip: `functools.lru_cache` is free performance for pure functions.",
    "Tip: Keep functions under 20 lines when possible.",
    "Tip: Use `pathlib.Path` instead of `os.path` for cleaner code.",
    "Tip: Batch your GPU operations to maximize throughput.",
    "Tip: Document your data pipeline assumptions explicitly.",
    "Tip: Use virtual environments for every project.",
    "Tip: Read the error message before Googling it.",
    "Tip: `git stash` is your friend when switching contexts.",
    "Tip: Use f-strings over .format() for readability.",
    "Tip: Set random seeds for reproducible ML experiments.",
]


def load_log() -> dict:
    """Load or initialize the contributions log."""
    path = Path(LOG_FILE)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"entries": [], "streak": 0, "total_commits": 0}


def save_log(log: dict) -> None:
    """Save the contributions log."""
    path = Path(LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def update_readme(log: dict, today: str, topic: str, tip: str) -> None:
    """Update README with current stats and today's entry."""
    streak = log["streak"]
    total = log["total_commits"]

    readme_content = f"""# Daily Learning Log

A personal log tracking my daily CS/NLP/ML learning journey.

## Stats

| Metric | Value |
|--------|-------|
| Current Streak | {streak} days |
| Total Entries | {total} |
| Last Updated | {today} |

## Today's Focus

> {topic}

## Daily Tip

{tip}

## About

This repo tracks my daily learning and coding activity.
Each day, I log what I studied or worked on, building
a consistent habit of learning and growth.

Topics: NLP, Machine Learning, Computational Linguistics,
Software Engineering, Data Structures & Algorithms.

---

*Updated daily via GitHub Actions*
"""
    with open(README_FILE, "w") as f:
        f.write(readme_content)


def main():
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    log = load_log()

    # Check if already committed today
    if log["entries"] and log["entries"][-1]["date"] == today:
        print(f"Already committed today ({today}). Skipping.")
        return

    # Pick random topic and tip
    topic = random.choice(TOPICS)
    tip = random.choice(TIPS)

    # Update streak
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    if log["entries"] and log["entries"][-1]["date"] == yesterday:
        log["streak"] += 1
    elif not log["entries"]:
        log["streak"] = 1
    else:
        log["streak"] = 1  # Reset streak

    log["total_commits"] += 1

    # Add entry
    entry = {
        "date": today,
        "timestamp": now,
        "topic": topic,
        "tip": tip,
        "streak": log["streak"],
    }
    log["entries"].append(entry)

    # Keep only last 90 days to avoid bloat
    if len(log["entries"]) > 90:
        log["entries"] = log["entries"][-90:]

    # Save everything
    save_log(log)
    update_readme(log, today, topic, tip)

    print(f"Committed for {today}")
    print(f"  Topic: {topic}")
    print(f"  Streak: {log['streak']} days")
    print(f"  Total: {log['total_commits']} entries")


if __name__ == "__main__":
    main()
