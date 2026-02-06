from flask import Flask, render_template, request, Response, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # change to "mixtral-8x7b-32768" if needed
    temperature=0.7,
    streaming=True,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Real-time web search tool (free, no key needed)
search = DuckDuckGoSearchRun()

# In-memory conversations: session_id → list of messages
conversations = {}

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
Current real-world date and time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}

You are GenChat, a helpful, concise, friendly and truthful AI assistant.
You have access to real-time web search via DuckDuckGo.
Use web search automatically for:
- Current events
- News
- Dates (e.g. "what year is it")
- Facts after your training cutoff
- Anything that requires up-to-date information

When you search, include "Searching..." in your response, then provide the answer based on search results.
Answer naturally and accurately.
Use markdown for code blocks, lists, bold/italics when useful.
Do not hallucinate or make things up.
Be useful, witty when appropriate, but always professional.
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "No message"}), 400

    if session_id not in conversations:
        conversations[session_id] = []

    # Add user message
    conversations[session_id].append({"role": "user", "content": message})

    def generate():
        full_response = ""
        try:
            # Decide if search is needed
            needs_search = any(word in message.lower() for word in [
                "current", "now", "today", "president", "news", "latest", "who is",
                "what year", "what time", "happening", "recent", "2026", "olympics"
            ])

            search_results = ""
            if needs_search:
                try:
                    search_results = search.run(message)
                    full_response += "Searching... \n\n" + search_results + "\n\n"
                except Exception as e:
                    full_response += f"Search failed: {str(e)}\n\n"

            # Build messages with history + search if any
            messages = [
                {"role": "system", "content": f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}. Search results if relevant:\n{search_results}"},
                *conversations[session_id],
                {"role": "user", "content": message}
            ]

            for chunk in llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    full_response += content
                    yield content

            # Save assistant response
            conversations[session_id].append({"role": "assistant", "content": full_response})

        except Exception as e:
            yield f"\nError: {str(e)}"

    return Response(generate(), mimetype="text/plain")

@app.route("/clear", methods=["POST"])
def clear():
    session_id = request.json.get("session_id", "default")
    if session_id in conversations:
        del conversations[session_id]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)