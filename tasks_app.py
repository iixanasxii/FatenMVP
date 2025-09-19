"""
tasks_app.py

A Flask web app that:
- Loads tasks from tasks.json (with 'active' and 'completed' lists).
- Displays them in a nice UI (gradient background, single column style).
- Allows marking tasks or steps as completed (moving them to 'completed' list).
"""

import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder="static")

TASKS_FILE = "tasks.json"

def load_tasks():
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"active": [], "completed": []}

def save_tasks(tasks_data):
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks_data, f, indent=2, ensure_ascii=False)

@app.route("/")
def home():
    return render_template("tasks_index.html")

@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    data = load_tasks()
    return jsonify(data)

@app.route("/api/complete_step", methods=["POST"])
def complete_step():
    """
    Mark a specific step as done.
    If all steps in a task are done => move entire task to 'completed'.
    """
    payload = request.json
    task_index = payload.get("taskIndex")
    step_index = payload.get("stepIndex")

    tasks_data = load_tasks()
    active_tasks = tasks_data["active"]

    if task_index is not None and step_index is not None:
        # Mark the step
        if 0 <= task_index < len(active_tasks):
            steps = active_tasks[task_index]["steps"]
            if 0 <= step_index < len(steps):
                steps[step_index]["done"] = True

                # Check if all steps are done
                all_done = all(s["done"] for s in steps)
                if all_done:
                    # Move the entire task to completed
                    completed_task = active_tasks.pop(task_index)
                    tasks_data["completed"].append(completed_task)

                save_tasks(tasks_data)
                return jsonify({"status": "ok", "message": "Step marked done."})
    return jsonify({"status": "error", "message": "Invalid task or step index."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
