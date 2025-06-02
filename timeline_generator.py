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

# Load settings from config.json
with open(CONFIG_FILE) as f:
    config = json.load(f)

SOURCE_REPO_URL = config["repo_url"]
BRANCH = config["branch"]
LOCAL_PATH = config["local_path"]
MAX_COMMITS = config.get("max_commits") or None  # Can be None for all commits
LARGE_COMMIT_LINE_CHANGE = config["large_commit_line_threshold"]
COMMIT_MESSAGE_SUMMARY_THRESHOLD = config.get("commit_message_summary_threshold", 50)
DEVELOPER_INACTIVITY_DAYS = config["developer_leave_inactivity_days"]
USE_LLM_SUMMARIZER = config["use_llm_summarizer"]
FEATURE_OFFSETS = config["feature_event_offsets"]  # Dict of feature event name -> days offset
BUG_OFFSETS = config["bug_event_offsets"]          # Dict of bug event name -> days offset

print(f"Max commits to analyze: {'ALL' if MAX_COMMITS is None else MAX_COMMITS}")

# Set up Gemini LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Summarize long commit messages using Gemini or fallback
def summarize_commit_message(commit_message: str) -> str:
    length = len(commit_message.strip())
    print(f"Commit message length: {length} characters")

    if length <= COMMIT_MESSAGE_SUMMARY_THRESHOLD:
        return commit_message.strip()

    if USE_LLM_SUMMARIZER:
        try:
            prompt = f"Summarize the following commit message into a short feature or bugfix title:\n\n\"{commit_message.strip()}\""
            response = gemini_model.generate_content(prompt)
            summary = response.text.strip()
            print("LLM summary used.")
            return summary if summary else commit_message[:50] + "..."
        except Exception as e:
            # Handle API errors or quota limits
            if "quota" in str(e).lower() or "429" in str(e):
                print("[Gemini LLM] API quota exceeded. Proceeding with fallback summarization.")
            else:
                print(f"[Gemini] Error during summarization: {e}. Using fallback.")
            return commit_message[:COMMIT_MESSAGE_SUMMARY_THRESHOLD] + "..."
    else:
        print("LLM off — using fallback truncation.")
        return commit_message[:COMMIT_MESSAGE_SUMMARY_THRESHOLD] + "..."

# Pick a developer not in the excluded list
def pick_other_dev(devs, exclude):
    choices = [d for d in devs if d not in exclude]
    return random.choice(choices) if choices else exclude[0]

# Clone or validate the local repository
def prepare_repo(source_repo_url: str, branch: str, local_path: str = "source_repo") -> Repo:
    if os.path.exists(local_path):
        print(f"Repository already exists at '{local_path}'. Verifying remote URL and branch...")
        repo = Repo(local_path)
        origin_url = next(repo.remote("origin").urls)

        # Make sure it's the same remote URL
        if origin_url != source_repo_url:
            raise ValueError(f"Local repo remote '{origin_url}' doesn't match expected '{source_repo_url}'.")

        # Try to checkout the correct branch
        try:
            repo.git.checkout(branch)
        except GitCommandError as e:
            raise RuntimeError(f"Failed to checkout branch '{branch}': {e}")

        print("Repository verified and branch checked out.")
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
    commits.reverse()  # Start from oldest to newest

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

        # Track first appearance of a developer
        if dev_activity[author]["first_commit"] is None:
            dev_activity[author]["first_commit"] = commit_date
            print(f"Developer joined: {author}")
            timeline_events.append({
                "event": "DEV_JOIN",
                "date": commit_date.isoformat(),
                "developer": author
            })

        # Update last activity of developer
        dev_activity[author]["last_commit"] = commit_date
        dev_activity[author]["last_commit_id"] = commit.hexsha[:7]

        all_devs = list(dev_activity.keys())
        summary = summarize_commit_message(commit_message)

        if lines_changed >= LARGE_COMMIT_LINE_CHANGE:
            # Simulate a feature event flow
            print("Large commit — treating as FEATURE")
            feature_id = f"{summary} (feature-{commit.hexsha[:7]})"
            print(f"Feature Summary: {feature_id}")

            proposer = author
            planner = pick_other_dev(all_devs, [proposer])
            designer = pick_other_dev(all_devs, [proposer, planner])
            reviewer = pick_other_dev(all_devs, [proposer, planner, designer])

            timeline_events.append({
                "event": "FEATURE_PROPOSE",
                "date": commit_date.isoformat(),
                "feature": feature_id,
                "author": author,
                "commit_id": commit.hexsha[:7]
            })

            # Create a mapping from event names to the correct developer roles
            role_map = {
                "FEATURE_PLAN": planner,
                "FEATURE_DESIGN": designer,
                "FEATURE_DESIGN_REVIEW": reviewer,
                "FEATURE_PR_OPEN": proposer,
                "FEATURE_PR_REVIEW": reviewer,
                "FEATURE_PR_CLOSE": proposer,
            }

            timeline_events.extend([
                {
                    "event": event_name,
                    "date": (commit_date + timedelta(days=FEATURE_OFFSETS[event_name])).isoformat(),
                    "developer": role_map.get(event_name, proposer),
                    "feature": feature_id
                }
                for event_name in FEATURE_OFFSETS
            ])
        else:
            # Simulate a bugfix event flow
            print("Small commit — treating as BUG")
            bug_id = f"{summary} (bug-{commit.hexsha[:7]})"
            print(f"Bug Summary: {bug_id}")

            fixer = author
            reviewer = pick_other_dev(all_devs, [fixer])

            timeline_events.append({
                "event": "BUG_OPEN",
                "date": commit_date.isoformat(),
                "bug": bug_id,
                "author": author
            })

            timeline_events.extend([
                {
                    "event": event_name,
                    "date": (commit_date + timedelta(days=BUG_OFFSETS[event_name])).isoformat(),
                    "developer": fixer if "REVIEW" not in event_name else reviewer,
                    "bug": bug_id
                }
                for event_name in BUG_OFFSETS
            ])

    # Add developer leave events based on inactivity threshold
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
        repo = prepare_repo(SOURCE_REPO_URL, BRANCH, LOCAL_PATH)
        timeline = analyze_repo(repo, BRANCH, max_commits=MAX_COMMITS)
        save_timeline(timeline)
    except Exception as e:
        print(f"Error: {e}")
