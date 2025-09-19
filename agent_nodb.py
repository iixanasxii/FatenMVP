"""
agent_nodb.py

An agent that:
1) Reads .txt files from 'docs/' without using a vectorstore.
2) Passes the text to a local LLM in JSON mode to extract tasks.
3) If the LLM output is a dictionary with multiple arrays of tasks,
   we unify them into a single array of tasks.
4) Writes tasks to 'tasks.json', each with a title and steps.

Usage:
  python agent_nodb.py
"""

import os
import json

# -----------------------------------------------------------------------------
# 1) ENV SETUP
# -----------------------------------------------------------------------------

os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------------------------------------------------------------
# 2) LOCAL LLM
# -----------------------------------------------------------------------------
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Adjust to the model name/path you have installed
local_llm = "llama3.2:3b-instruct-fp16"

# We want JSON output, so use format="json"
# You can also set max_tokens if the model truncates
llm = ChatOllama(model=local_llm, temperature=0)
llm_json = ChatOllama(model=local_llm, temperature=0, format="json")

# -----------------------------------------------------------------------------
# 3) LOADER FOR DOCS
# -----------------------------------------------------------------------------
from langchain_community.document_loaders import DirectoryLoader
# If you have PDF or other file types, you'd need additional logic or loaders.

# -----------------------------------------------------------------------------
# 4) TASKS STORAGE
# -----------------------------------------------------------------------------
TASKS_FILE = "tasks.json"

def load_tasks():
    """Load existing tasks from tasks.json, or return empty structure if none."""
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"active": [], "completed": []}

def save_tasks(tasks_data):
    """Save tasks to tasks.json."""
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks_data, f, indent=2, ensure_ascii=False)

# -----------------------------------------------------------------------------
# 5) READ RAW DOCS TEXT (NO DB)
# -----------------------------------------------------------------------------
def load_docs_text():
    """
    Directly read text from all .txt files in 'docs/' folder.
    Return a concatenated string of all text.
    """
    docs_text = ""
    docs_folder = "docs"
    if not os.path.isdir(docs_folder):
        print(f"No 'docs' folder found at {os.path.abspath(docs_folder)}. Creating it.")
        os.makedirs(docs_folder, exist_ok=True)
        return docs_text  # empty

    loader = DirectoryLoader(docs_folder, glob="*.txt")
    doc_list = loader.load()
    if not doc_list:
        print(f"No .txt files found in '{docs_folder}'.")
        return docs_text

    # Concatenate all doc contents
    for d in doc_list:
        docs_text += d.page_content + "\n\n"
    return docs_text

# -----------------------------------------------------------------------------
# 6) LLM PARSING FOR TASKS
# -----------------------------------------------------------------------------
def parse_llm_response_to_tasklist(raw_content):
    """
    Takes the raw string from the LLM response (which should be JSON),
    and tries to produce a list of tasks:
       [ { "title": "...", "steps": [ "step1", "step2"] }, ... ]
    We handle the cases:
    (A) Top-level is an array
    (B) Top-level is an object with keys like "steps" or "meetingNotes" that contain arrays
    """
    try:
        parsed = json.loads(raw_content)
    except:
        return []

    # Case A: If it's already a list, return it
    if isinstance(parsed, list):
        return parsed  # We assume each element is {title, steps}

    # Case B: If it's a dict, we look for arrays of tasks
    if isinstance(parsed, dict):
        # We'll unify any arrays we find that might contain tasks
        result = []

        # We'll check known keys that might contain tasks
        # e.g. "steps", "meetingNotes", "tasks", etc.
        # Adjust as needed based on your local LLM's structure

        # 1) If top-level "steps" is an array
        if "steps" in parsed and isinstance(parsed["steps"], list):
            # This might be an array of objects or strings
            arr = parsed["steps"]
            for item in arr:
                # If item is an object with "title" / "steps"
                if isinstance(item, dict) and "title" in item and "steps" in item:
                    result.append({
                        "title": item["title"],
                        "steps": item["steps"] if isinstance(item["steps"], list) else []
                    })
                # If item is a string, that might mean there's a single "title" and these are "steps" only
                elif isinstance(item, str):
                    # We might unify them under the parent's "title" if it exists
                    # But your example didn't have a separate parent's title
                    # For now, we'll treat them as a single task with "title": "Untitled"
                    result.append({"title": "Untitled", "steps": [item]})
                # else skip
        # 2) If "meetingNotes" in parsed
        if "meetingNotes" in parsed and isinstance(parsed["meetingNotes"], list):
            for item in parsed["meetingNotes"]:
                # Each item is presumably a { title, steps } object
                if isinstance(item, dict) and "title" in item and "steps" in item:
                    result.append({
                        "title": item["title"],
                        "steps": item["steps"] if isinstance(item["steps"], list) else []
                    })
        # 3) If there's any other key that might hold tasks, add it similarly
        # e.g. if "tasks" in parsed, parse that

        return result

    # If neither a list nor a dict with known keys, return empty
    return []

def find_tasks_in_docs(docs_text):
    """
    Use the LLM to parse tasks from docs_text.
    Return a list of tasks, each: { "title": "...", "steps": ["...", "..."] }
    """
    system_prompt = """You are an AI assistant that extracts tasks or action items from the text provided.

Please strictly return valid JSON.

If there's more than one set of tasks or "sections," unify them in an array. Each element should look like:
{
  "title": "...",
  "steps": ["...", "..."]
}

For example, if you see multiple teams with tasks, place them as multiple objects. 
Your top-level output MUST be an array. If your data is in keys like "title", "steps", "meetingNotes", unify them into one array.

Example final output if there's just one team:
[
  {
    "title": "Engineering Team",
    "steps": ["Step1", "Step2"]
  }
]

If multiple:
[
  {
    "title": "Marketing Team",
    "steps": ["Draft emails", "Launch campaign"]
  },
  {
    "title": "Engineering Team",
    "steps": ["Refactor code", "Set up CI"]
  }
]

No extra text or explanation, just the JSON. 
"""
    user_message = f"Here is the text to parse:\n\n{docs_text}\n\n"

    resp = llm_json.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    print("DEBUG LLM raw response:", resp.content)

    # parse the raw JSON string into a Python structure
    tasks_list = parse_llm_response_to_tasklist(resp.content)
    return tasks_list

# -----------------------------------------------------------------------------
# 7) MAIN: RUN AGENT WITHOUT DB
# -----------------------------------------------------------------------------
def run_agent_no_db():
    # 1) read docs text
    docs_text = load_docs_text()
    if not docs_text.strip():
        print("No text found in 'docs/' folder. Aborting.")
        return

    # 2) parse tasks
    tasks_found = find_tasks_in_docs(docs_text)
    if not tasks_found:
        print("No tasks found in the docs (LLM returned empty or invalid JSON).")
        return

    # 3) load existing tasks
    tasks_data = load_tasks()

    # 4) append new tasks
    for t in tasks_found:
        title = t.get("title", "Untitled Task")
        steps_list = t.get("steps", [])
        # Each step is turned into an object with "description", "done": false
        # in tasks.json
        new_task = {
            "title": title,
            "steps": [{"description": s, "done": False} for s in steps_list]
        }
        tasks_data["active"].append(new_task)

    # 5) save
    save_tasks(tasks_data)
    print("Tasks added to tasks.json successfully!")

if __name__ == "__main__":
    run_agent_no_db()
