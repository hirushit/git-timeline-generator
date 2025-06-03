import os
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict
from git import Repo, GitCommandError
from dotenv import load_dotenv
import google.generativeai as genai

# Load configuration from .env and config.json
CONFIG_FILE = "config.json"
load_dotenv()

# Default configuration
default_config = {
    "repo_url": "https://github.com/redis/redis.git",
    "remote_name": "origin",
    "branch": "unstable",
    "local_path": "source_repo",
    "max_commits": 100,
    "large_commit_line_threshold": 100,
    "commit_message_summary_threshold": 50,
    "use_llm_summarizer": False,
    "developer_leave_inactivity_days": 30,
    "feature_event_offsets": {
        "FEATURE_PLAN": -5,
        "FEATURE_DESIGN": -4,
        "FEATURE_DESIGN_REVIEW": -3,
        "FEATURE_PR_OPEN": -2,
        "FEATURE_PR_REVIEW": -1,
        "FEATURE_PR_CLOSE": 0
    },
    "bug_event_offsets": {
        "BUG_PR_OPEN": -3,
        "BUG_PR_REVIEW": -2,
        "BUG_PR_CLOSE": -1,
        "BUG_CLOSE": 0
    }
}

# Merge utility
def merge_configs(default, custom):
    for key, value in default.items():
        if isinstance(value, dict) and isinstance(custom.get(key), dict):
            default[key] = merge_configs(value, custom[key])
        else:
            default[key] = custom.get(key, value)
    return default

# Load and merge config
try:
    with open(CONFIG_FILE) as f:
        user_config = json.load(f)
        config = merge_configs(default_config.copy(), user_config)
except (FileNotFoundError, json.JSONDecodeError):
    print(f"[Warning] Missing or invalid config file '{CONFIG_FILE}'. Using default values.")
    config = default_config.copy()

SOURCE_REPO_URL = config.get("repo_url")
REMOTE_NAME = config.get("remote_name", "origin")
BRANCH = config.get("branch")
LOCAL_PATH = config.get("local_path")
MAX_COMMITS = config.get("max_commits")
LARGE_COMMIT_LINE_CHANGE = config.get("large_commit_line_threshold")
COMMIT_MESSAGE_SUMMARY_THRESHOLD = config.get("commit_message_summary_threshold")
DEVELOPER_INACTIVITY_DAYS = config.get("developer_leave_inactivity_days")
USE_LLM_SUMMARIZER = config.get("use_llm_summarizer")
FEATURE_OFFSETS = config.get("feature_event_offsets", {})
BUG_OFFSETS = config.get("bug_event_offsets", {})

print(f"Max commits to analyze: {'ALL' if MAX_COMMITS is None else MAX_COMMITS}")

# Set up Gemini LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Summarize long commit messages using Gemini or fallback
def summarize_commit_message(commit_message: str) -> str:
    length = len(commit_message.strip())
    print(f"Commit message length: {length} characters")

    if length <= COMMIT_MESSAGE_SUMMARY_THRESHOLD:
        return commit_message.strip()

    if USE_LLM_SUMMARIZER:
        try:
            prompt = f"Summarize the following commit message into a short feature or bugfix title. Please do not use any Markdown formatting like **bold** or *italic*:\n\n\"{commit_message.strip()}\""
            response = gemini_model.generate_content(prompt)
            summary = response.text.strip()
            print("LLM summary used.")
            return summary if summary else commit_message[:COMMIT_MESSAGE_SUMMARY_THRESHOLD] + "..."
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("[Gemini LLM] API quota exceeded. Proceeding with fallback summarization.")
            else:
                print(f"[Gemini] Error during summarization: {e}. Using fallback.")
            return commit_message[:COMMIT_MESSAGE_SUMMARY_THRESHOLD] + "..."
    else:
        print("LLM off — using fallback truncation.")
        return commit_message[:COMMIT_MESSAGE_SUMMARY_THRESHOLD] + "..."

def is_active(dev_name: str, as_of_day: datetime, activity_log, threshold_days: int):
    if dev_name not in activity_log or activity_log[dev_name]["last_commit"] is None:
        return False  
    
    last_commit = activity_log[dev_name]["last_commit"]
    return (as_of_day - last_commit).days <= threshold_days

def pick_other_dev(devs, exclude, as_of_day, dev_activity):
    active_devs = [d for d in devs if d not in exclude and is_active(d, as_of_day, dev_activity, DEVELOPER_INACTIVITY_DAYS)]
    if not active_devs:
        print(f"[Warning] No active devs found on {as_of_day.date()}. Falling back to excluded dev: {exclude[0]}")
    return random.choice(active_devs) if active_devs else exclude[0]

def prepare_repo(source_repo_url: str, branch: str, local_path: str = "source_repo", remote_name: str = "origin") -> Repo:
    if os.path.exists(local_path):
        print(f"Repository already exists at '{local_path}'. Verifying remote URL and branch...")
        repo = Repo(local_path)

        if remote_name not in repo.remotes:
            raise ValueError(f"Remote '{remote_name}' not found in local repository.")

        remote = repo.remotes[remote_name]
        remote_url = next(remote.urls)

        if remote_url != source_repo_url:
            raise ValueError(f"Remote URL mismatch: expected '{source_repo_url}', found '{remote_url}'")

        try:
            repo.git.checkout(branch)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to checkout branch '{branch}': {e}")

        try:
            remote.fetch()
        except GitCommandError as e:
            print(f"[Warning] Failed to fetch from remote: {e}")

        try:
            status = repo.git.rev_list('--left-right', '--count', f"{branch}...{remote_name}/{branch}")
        except GitCommandError as e:
            print(f"[Warning] Could not compare with remote branch: {e}")

        print("Fetching latest changes from remote...")
        remote.fetch()

        status = repo.git.rev_list('--left-right', '--count', f"{branch}...{remote_name}/{branch}")
        behind, ahead = map(int, status.strip().split('\t'))

        if behind and not ahead:
            print(f"Local branch '{branch}' is behind by {behind} commits. Pulling latest changes...")
            remote.pull(branch)
        elif ahead and not behind:
            raise RuntimeError(f"Local branch '{branch}' is ahead of '{remote_name}/{branch}' by {ahead} commits. Push or reset required.")
        elif ahead and behind:
            raise RuntimeError(f"Local branch '{branch}' has diverged from '{remote_name}/{branch}' (ahead {ahead}, behind {behind}). Manual resolution required.")
        else:
            print(f"Branch '{branch}' is up to date with '{remote_name}/{branch}'.")

        print("Repository verified and updated.")
        return repo
    else:
        print(f"Cloning repository from '{source_repo_url}' into '{local_path}'...")
        repo = Repo.clone_from(source_repo_url, local_path, branch=branch)
        print("Repository cloned successfully.")
        return repo

# Analyze the Git history and simulate a project timeline
def analyze_repo(repo: Repo, branch: str, max_commits=None):
    print(f"Analyzing branch '{branch}'...")
    commits = list(repo.iter_commits(branch))
    print(f"Total commits found: {len(commits)}")
    commits.reverse()
    if max_commits:
        commits = commits[:max_commits]
        print(f"Processing first {max_commits} commits")

    dev_activity = defaultdict(lambda: {
        "first_commit": None,
        "last_commit": None,
        "last_commit_id": None
    })
    timeline_events = []

    for idx, commit in enumerate(commits, 1):
        author = commit.author.name
        commit_date = datetime.fromtimestamp(commit.committed_date)
        lines_changed = commit.stats.total['lines']
        commit_message = commit.message.strip()

        print(f"[{idx}/{len(commits)}] Commit {commit.hexsha[:7]} by {author} on {commit_date.isoformat()} — {lines_changed} lines changed")

        if dev_activity[author]["first_commit"] is None:
            dev_activity[author]["first_commit"] = commit_date
            print(f"Developer joined: {author}")
            timeline_events.append({
                "event": "DEV_JOIN",
                "date": commit_date.isoformat(),
                "developer": author
            })

        dev_activity[author]["last_commit"] = commit_date
        dev_activity[author]["last_commit_id"] = commit.hexsha[:7]

        all_devs = list(dev_activity.keys())
        summary = summarize_commit_message(commit_message)

        if lines_changed >= LARGE_COMMIT_LINE_CHANGE:
            print("Large commit — treating as FEATURE")
            feature_id = summary
            proposer = author
            planner = pick_other_dev(all_devs, [proposer], commit_date, dev_activity)
            designer = pick_other_dev(all_devs, [proposer, planner], commit_date, dev_activity)
            reviewer = pick_other_dev(all_devs, [proposer, planner, designer], commit_date, dev_activity)

            role_map = {
                "FEATURE_PLAN": planner,
                "FEATURE_DESIGN": designer,
                "FEATURE_DESIGN_REVIEW": reviewer,
                "FEATURE_PR_OPEN": proposer,
                "FEATURE_PR_REVIEW": reviewer,
                "FEATURE_PR_CLOSE": proposer,
            }

            for event_name, offset in (FEATURE_OFFSETS or {}).items():
                event_day = commit_date + timedelta(days=offset)
                timeline_events.append({
                    "event": event_name,
                    "date": event_day.isoformat(),
                    "developer": role_map.get(event_name, proposer),
                    "feature": feature_id,
                    "source_commit_id": commit.hexsha[:7]
                })

            timeline_events.append({
                "event": "FEATURE_PROPOSE",
                "date": (commit_date + timedelta(days=FEATURE_OFFSETS.get("FEATURE_PR_OPEN", 0))).isoformat(),
                "feature": feature_id,
                "author": author,
                "commit_id": commit.hexsha[:7],
                "source_commit_id": commit.hexsha[:7]
            })
        else:
            print("Small commit — treating as BUG")
            bug_id = summary
            fixer = author
            reviewer = pick_other_dev(all_devs, [fixer], commit_date, dev_activity)

            timeline_events.append({
                "event": "BUG_OPEN",
                "date": commit_date.isoformat(),
                "bug": bug_id,
                "author": author,
                "source_commit_id": commit.hexsha[:7]
            })

            for event_name, offset in (BUG_OFFSETS or {}).items():
                event_day = commit_date + timedelta(days=offset)
                timeline_events.append({
                    "event": event_name,
                    "date": event_day.isoformat(),
                    "developer": fixer if "REVIEW" not in event_name else reviewer,
                    "bug": bug_id,
                    "source_commit_id": commit.hexsha[:7]
                })

    print("Adding developer leave events...")
    for dev, activity in dev_activity.items():
        leave_date = activity["last_commit"] + timedelta(days=DEVELOPER_INACTIVITY_DAYS)
        print(f"Developer left: {dev} on {leave_date.isoformat()}")
        timeline_events.append({
            "event": "DEV_LEAVE",
            "date": leave_date.isoformat(),
            "developer": dev,
            "last_commit_date": activity["last_commit"].isoformat(),
            "last_commit_id": activity["last_commit_id"]
        })

    print("Analysis complete.")
    return timeline_events

# Save timeline to JSON file
def save_timeline(events, filename="timeline.json"):
    print(f"Saving timeline to '{filename}'...")
    events.sort(key=lambda e: e["date"])
    with open(filename, "w") as f:
        json.dump(events, f, indent=2)
    print("Timeline saved.")

# Main entry point
if __name__ == "__main__":
    try:
        repo = prepare_repo(SOURCE_REPO_URL, BRANCH, LOCAL_PATH, REMOTE_NAME)
        timeline = analyze_repo(repo, BRANCH, max_commits=MAX_COMMITS)
        save_timeline(timeline)
    except Exception as e:
        print(f"Error: {e}")
