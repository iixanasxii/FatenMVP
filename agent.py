"""
agent.py

Agent code that:
1) Loads the local vectorstore from 'db_store/'.
2) Retrieves the latest 'meeting minutes' or relevant docs.
3) Uses an LLM to parse tasks/action items from those docs.
4) Breaks each task into smaller steps.
5) Saves tasks to 'tasks.json' for the web interface to display.

Run:  python agent.py
"""

import os
import json

# 1) ENV SETUP
os.environ["TAVILY_API_KEY"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 2) LLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json = ChatOllama(model=local_llm, temperature=0, format="json")

# 3) LOAD VECTORSTORE
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_nomic.embeddings import NomicEmbeddings

PERSIST_DIRECTORY = "db_store"
embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

if not (os.path.isdir(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY)):
    print(f"WARNING: No vectorstore found in '{PERSIST_DIRECTORY}'. Please create it first.")
    vectordb = None
else:
    print(f"Loading vectorstore from: {PERSIST_DIRECTORY}")
    vectordb = Chroma(
        collection_name="my_collection",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function,
    )

# 4) Where tasks will be stored
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

# 5) Agent logic to find tasks in docs
def find_tasks_in_docs(docs_text):
    """
    Use the LLM to parse tasks from the text of the docs.
    We return a list of tasks, each with a 'title' and 'steps' (list of strings).
    """
    system_prompt = """You are an AI assistant that extracts tasks from the text provided look for any task in text.
    Please return a JSON array of tasks, where each task has "title" and "steps".
    Example format:
    [
    {
        "title": "Some Task",
        "steps": ["Step one", "Step two"]
    },
    ...
    ]
    """
    user_message = f"Please find tasks or action items in the following text:\n\n{docs_text}\n\n"

    resp = llm_json.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])
    # Attempt to parse JSON
    try:
        tasks_parsed = json.loads(resp.content)
        if isinstance(tasks_parsed, list):
            return tasks_parsed
        else:
            return []
    except:
        return []

def run_agent():
    """
    Main function:
    1) Retrieve 'meeting minutes' docs from vectorstore
    2) Parse tasks
    3) Save them to tasks.json
    """
    if vectordb is None:
        print("No vectorstore found, cannot retrieve docs.")
        return

    # Step 1: Retrieve docs about 'meeting minutes' or 'tasks' 
    # You can adjust the query as needed
    query = "meeting minutes OR tasks OR action items"
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    docs_text = "\n\n".join([doc.page_content for doc in docs])

    if not docs_text.strip():
        print("No relevant docs found for 'meeting minutes'.")
        return

    # Step 2: Parse tasks from docs
    tasks_found = find_tasks_in_docs(docs_text)
    if not tasks_found:
        print("No tasks found in the docs.")
        return

    # Step 3: Load existing tasks
    tasks_data = load_tasks()

    # Step 4: Append new tasks to the 'active' tasks
    for t in tasks_found:
        # Make sure each task has "title" and "steps"
        task_title = t.get("title", "Untitled Task")
        steps = t.get("steps", [])
        new_task = {
            "title": task_title,
            "steps": [{"description": step, "done": False} for step in steps]
        }
        tasks_data["active"].append(new_task)

    # Step 5: Save
    save_tasks(tasks_data)
    print("Tasks added to tasks.json successfully!")

if __name__ == "__main__":
    run_agent()
