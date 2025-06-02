import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from timeline_generator import (
    summarize_commit_message,
    pick_other_dev,
    analyze_repo,
    LARGE_COMMIT_LINE_CHANGE,
    DEVELOPER_INACTIVITY_DAYS,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tests for summarize_commit_message

def test_summarize_commit_message_short():
    msg = "Fix typo"
    summary = summarize_commit_message(msg)
    logger.info(f"test_summarize_commit_message_short: input='{msg}', output='{summary}'")
    assert summary == msg  # Short message returns as-is

def test_summarize_commit_message_long_no_llm(monkeypatch):
    # Turn off LLM summarizer for fallback truncation
    monkeypatch.setattr("timeline_generator.USE_LLM_SUMMARIZER", False)
    long_msg = "a" * 100
    summary = summarize_commit_message(long_msg)
    logger.info(f"test_summarize_commit_message_long_no_llm: input length={len(long_msg)}, output='{summary}'")
    assert summary.endswith("...") and len(summary) <= 53  # truncated

def test_summarize_commit_message_long_with_llm(monkeypatch):
    # Mock Gemini LLM generate_content method
    class FakeResponse:
        text = "Short summary"

    class FakeModel:
        def generate_content(self, prompt):
            return FakeResponse()

    monkeypatch.setattr("timeline_generator.USE_LLM_SUMMARIZER", True)
    monkeypatch.setattr("timeline_generator.gemini_model", FakeModel())

    long_msg = "a" * 100
    summary = summarize_commit_message(long_msg)
    logger.info(f"test_summarize_commit_message_long_with_llm: input length={len(long_msg)}, output='{summary}'")
    assert summary == "Short summary"

def test_summarize_commit_message_llm_error(monkeypatch):
    # Simulate LLM throwing an exception (e.g., quota exceeded)
    class FakeModel:
        def generate_content(self, prompt):
            raise Exception("Quota exceeded")

    monkeypatch.setattr("timeline_generator.USE_LLM_SUMMARIZER", True)
    monkeypatch.setattr("timeline_generator.gemini_model", FakeModel())

    long_msg = "a" * 100
    summary = summarize_commit_message(long_msg)
    logger.info(f"test_summarize_commit_message_llm_error: input length={len(long_msg)}, output='{summary}'")
    assert summary.endswith("...")

# Tests for pick_other_dev 

def test_pick_other_dev_basic():
    devs = ["Alice", "Bob", "Charlie"]
    exclude = ["Alice"]
    chosen = pick_other_dev(devs, exclude)
    logger.info(f"test_pick_other_dev_basic: devs={devs}, exclude={exclude}, chosen='{chosen}'")
    assert chosen in ["Bob", "Charlie"]

def test_pick_other_dev_all_excluded():
    devs = ["Alice"]
    exclude = ["Alice"]
    chosen = pick_other_dev(devs, exclude)
    logger.info(f"test_pick_other_dev_all_excluded: devs={devs}, exclude={exclude}, chosen='{chosen}'")
    assert chosen == "Alice"

def test_pick_other_dev_no_exclusions():
    devs = ["Alice", "Bob"]
    exclude = []
    chosen = pick_other_dev(devs, exclude)
    logger.info(f"test_pick_other_dev_no_exclusions: devs={devs}, exclude={exclude}, chosen='{chosen}'")
    assert chosen in devs

# Tests for analyze_repo

def make_commit(author_name, date, lines_changed, message, hexsha):
    commit = MagicMock()
    commit.author.name = author_name
    commit.committed_date = date.timestamp()
    commit.stats.total = {"lines": lines_changed}
    commit.message = message
    commit.hexsha = hexsha
    return commit

def test_analyze_repo_small_commit_creates_bug_events():
    commit_date = datetime.now()
    commit = make_commit("Dev1", commit_date, 5, "Bug fix commit", "1234567abcdef")

    repo = MagicMock()
    repo.iter_commits.return_value = [commit]

    events = analyze_repo(repo, "main", max_commits=1)

    event_names = [e["event"] for e in events]
    logger.info(f"test_analyze_repo_small_commit_creates_bug_events: events={event_names}")
    assert "DEV_JOIN" in event_names
    assert "BUG_OPEN" in event_names
    assert any(e["event"] == "DEV_LEAVE" for e in events)

def test_analyze_repo_large_commit_creates_feature_events():
    commit_date = datetime.now()
    commit = make_commit("Dev1", commit_date, LARGE_COMMIT_LINE_CHANGE + 10, "Add big feature", "abcdef1234567")

    repo = MagicMock()
    repo.iter_commits.return_value = [commit]

    events = analyze_repo(repo, "main", max_commits=1)

    event_names = [e["event"] for e in events]
    logger.info(f"test_analyze_repo_large_commit_creates_feature_events: events={event_names}")
    assert "DEV_JOIN" in event_names
    assert "FEATURE_PROPOSE" in event_names
    assert any(e["event"] == "DEV_LEAVE" for e in events)

def test_analyze_repo_multiple_commits_multiple_devs():
    now = datetime.now()
    commits = [
        make_commit("Alice", now - timedelta(days=10), 5, "Fix bug A", "aaa111"),
        make_commit("Bob", now - timedelta(days=5), LARGE_COMMIT_LINE_CHANGE + 1, "Add feature B", "bbb222"),
        make_commit("Alice", now - timedelta(days=1), 3, "Fix bug C", "aaa333"),
    ]

    repo = MagicMock()
    repo.iter_commits.return_value = commits

    events = analyze_repo(repo, "main")

    devs = set(e.get("developer") or e.get("author") for e in events if e["event"] in ["DEV_JOIN", "FEATURE_PROPOSE", "BUG_OPEN"])
    event_types = set(e["event"] for e in events)
    logger.info(f"test_analyze_repo_multiple_commits_multiple_devs: devs={devs}, event_types={event_types}")

    assert devs == {"Alice", "Bob"}
    assert "FEATURE_PROPOSE" in event_types
    assert "BUG_OPEN" in event_types

def test_analyze_repo_developer_leave_date_correct():
    commit_date = datetime.now()
    commit = make_commit("DevLeave", commit_date, 5, "Fix bug", "deadbeef")

    repo = MagicMock()
    repo.iter_commits.return_value = [commit]

    events = analyze_repo(repo, "main", max_commits=1)

    leave_events = [e for e in events if e["event"] == "DEV_LEAVE"]
    logger.info(f"test_analyze_repo_developer_leave_date_correct: leave_events={leave_events}")
    assert leave_events, "Expected a DEV_LEAVE event"
    leave_event = leave_events[0]
    expected_leave_date = commit_date + timedelta(days=DEVELOPER_INACTIVITY_DAYS)
    actual_leave_date = datetime.fromisoformat(leave_event["date"])
    assert actual_leave_date == expected_leave_date

def test_analyze_repo_commit_with_no_author():
    commit = MagicMock()
    commit.author.name = ""
    commit.committed_date = datetime.now().timestamp()
    commit.stats.total = {"lines": 10}
    commit.message = "Some commit"
    commit.hexsha = "deadbeef"

    repo = MagicMock()
    repo.iter_commits.return_value = [commit]

    events = analyze_repo(repo, "main", max_commits=1)
    logger.info(f"test_analyze_repo_commit_with_no_author: events={events}")
    assert any(e["event"] == "DEV_JOIN" for e in events)

def test_analyze_repo_empty_commit_list():
    repo = MagicMock()
    repo.iter_commits.return_value = []
    events = analyze_repo(repo, "main", max_commits=5)
    logger.info(f"test_analyze_repo_empty_commit_list: events={events}")
    assert events == []  # No commits, no events

if __name__ == "__main__":
    pytest.main()
