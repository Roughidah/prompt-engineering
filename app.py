import json
import os
import threading
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# -----------------------------
# In-memory storage + lock
# -----------------------------
tasks_by_id = {}
lock = threading.Lock()

def get_next_id():
    """Compute next available ID dynamically."""
    return max(tasks_by_id.keys(), default=0) + 1

# -----------------------------
# AI config
# -----------------------------
OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_ERROR_RESPONSE = "Sorry, I couldn't understand your request."

functions = [
    {
        "name": "addTask",
        "description": "Add a new task to the to-do list",
        "parameters": {
            "type": "object",
            "properties": {"description": {"type": "string", "description": "The task description"}},
            "required": ["description"]
        }
    },
    {
        "name": "viewTasks",
        "description": "View all tasks in the to-do list",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "completeTask",
        "description": "Mark a task as complete",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "The ID of the task to complete"},
                "completed": {"type": "boolean", "description": "Whether the task is completed (defaults to true)"}
            }
        }
    },
    {
        "name": "deleteTask",
        "description": "Delete a task from the to-do list",
        "parameters": {
            "type": "object",
            "properties": {"task_id": {"type": "integer", "description": "The ID of the task to delete"}},
            "required": ["task_id"]
        }
    }
]

# -----------------------------
# Helpers
# -----------------------------
def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

def error_response(message, status):
    return jsonify({"error": message}), status

# -----------------------------
# Core task functions
# -----------------------------
def addTask(description: str) -> dict:
    description = description.strip()
    with lock:
        task_id = get_next_id()
        task = {
            "id": task_id,
            "description": description,
            "completed": False,
            "created_at": now_iso_utc()
        }
        tasks_by_id[task_id] = task
    return task

def viewTasks() -> list:
    with lock:
        return [tasks_by_id[k] for k in sorted(tasks_by_id.keys())]

def completeTask(task_id: int, completed: bool = True):
    with lock:
        task = tasks_by_id.get(task_id)
        if not task:
            return None
        task["completed"] = completed
        return task

def deleteTask(task_id: int):
    with lock:
        return tasks_by_id.pop(task_id, None)

# -----------------------------
# AI call
# -----------------------------
def call_ai_model(system_prompt, user_message):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            functions=functions,
            function_call="auto"
        )
        message = response["choices"][0]["message"]
        if "function_call" in message:
            return message["function_call"]
        parsed = json.loads(message.get("content", "{}") or "{}")
        if "function" in parsed and "parameters" in parsed:
            return {"name": parsed["function"], "arguments": json.dumps(parsed["parameters"])}
    except Exception as e:
        print(f"[AI Error] {e}")
    return None

# -----------------------------
# CORS
# -----------------------------
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
    return resp

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.get_json(silent=True) or {}
    description = data.get('description', '').strip()
    if not isinstance(description, str) or not description:
        return error_response("description is required and must be a string", 400)
    task = addTask(description)
    return jsonify(task), 201

@app.route('/tasks', methods=['GET'])
def list_tasks():
    return jsonify(viewTasks()), 200

@app.route('/tasks/<int:task_id>/complete', methods=['PATCH'])
def mark_complete(task_id):
    data = request.get_json(silent=True) or {}
    completed = data.get("completed", True)
    if not isinstance(completed, bool):
        return error_response("completed must be a boolean", 400)
    task = completeTask(task_id, completed)
    if not task:
        return error_response("task not found", 404)
    return jsonify(task), 200

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def remove_task(task_id):
    task = deleteTask(task_id)
    if not task:
        return error_response("task not found", 404)
    return jsonify({"deleted": task}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    system_prompt = """
    You are an AI assistant for a To-Do List application.
    Respond ONLY with function_call objects (no plain text).
    You have access to:
    • addTask(description: string)
    • viewTasks()
    • completeTask(task_id: int, completed?: bool)
    • deleteTask(task_id: int)
    """
    result = call_ai_model(system_prompt, user_message)
    if result is None:
        return jsonify({'response': DEFAULT_ERROR_RESPONSE})

    func = result.get("name")
    params = json.loads(result.get("arguments", "{}"))

    try:
        if func == "addTask":
            desc = params.get("description", "").strip()
            if not desc:
                return jsonify({'response': "Invalid description"})
            task = addTask(desc)
            return jsonify({'response': f"Task added: '{task['description']}'"})

        elif func == "viewTasks":
            tasks = viewTasks()
            task_list = [f"{t['id']}: {t['description']} {'✅' if t['completed'] else ''}" for t in tasks]
            return jsonify({'response': "Your tasks:\n" + "\n".join(task_list)})

        elif func == "completeTask":
            task_id = params.get("task_id")
            try:
                task_id = int(task_id)
            except (TypeError, ValueError):
                return jsonify({'response': "Invalid task ID"})
            completed = params.get("completed", True)
            task = completeTask(task_id, completed)
            if not task:
                return jsonify({'response': "Task not found"})
            return jsonify({'response': f"Task {task_id} marked as {'complete' if completed else 'incomplete'}"})

        elif func == "deleteTask":
            task_id = params.get("task_id")
            try:
                task_id = int(task_id)
            except (TypeError, ValueError):
                return jsonify({'response': "Invalid task ID"})
            task = deleteTask(task_id)
            if not task:
                return jsonify({'response': "Task not found"})
            return jsonify({'response': f"Task {task_id} deleted"})
    except Exception as e:
        print(f"[Chat Error] {e}")
        return jsonify({'response': DEFAULT_ERROR_RESPONSE})

    return jsonify({'response': DEFAULT_ERROR_RESPONSE})

if __name__ == '__main__':
    app.run(debug=True)
