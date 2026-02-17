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
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    streaming=True,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

search = DuckDuckGoSearchRun()

conversations = {}

system_prompt = """
You are GenChat, a concise, helpful, friendly and **truthful** AI assistant powered by Groq + Llama 3.3.

=== STRICT RULES YOU MUST FOLLOW EVERY TIME ===

1. Current date/time: Always use the "Current real-world date and time" value provided at the top.
2. For ANY question that could require up-to-date info (president, time, news, weather, sports scores, stock prices, recent events, elections, who won something recently, latest version of software, current year/month/day, etc.) → YOU MUST rely on the search results provided below or say you couldn't get current info.
3. If search results are provided → base your answer **primarily** on them. Do NOT fall back to your internal/training knowledge when it conflicts with search results.
4. If no useful search results or search failed → say honestly: "I couldn't get reliable real-time information right now — my search tool didn't return useful results."
5. When performing a search, start your streamed response with "Searching..." on its own line.
6. Be concise, accurate, professional. Use markdown for formatting (bold, lists, code blocks) when helpful.
7. Never hallucinate, never make up facts, never pretend to know current events without evidence from search or the current datetime line.
8. For simple time/date questions → prefer the injected current datetime over search unless the user asks for timezone conversion or something more complex.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "No message"}), 400

    # ─── DEBUG PRINTS ────────────────────────────────────────────────────────
    print("DEBUG: Received message (raw repr):", repr(message))
    print("DEBUG: message after .strip():", repr(message.strip()))

    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({"role": "user", "content": message})

    def generate():
        full_response = ""
        search_results = ""

        lower_msg = message.lower().strip()

        # ───────────────────────────────────────────────────────────────
        # VERY forgiving fast-path for time questions (debug mode)
        # ───────────────────────────────────────────────────────────────
        lower_clean = ''.join(c for c in message.lower() if c.isalnum() or c.isspace())
        has_what   = any(w in lower_clean for w in ["what", "whats", "what's"])
        has_time   = "time" in lower_clean
        is_short   = len(message.strip()) < 40
        no_tz      = all(t not in lower_clean for t in ["in", "timezone", "time zone", "convert", "london", "york", "est", "pst"])

        if has_what and has_time and is_short and no_tz:
            now_str = datetime.now().strftime("%I:%M %p %Z on %A, %B %d, %Y")
            response = f"The current time is **{now_str}**.\n\n"
            yield response
            full_response += response
            print("DEBUG: FAST-PATH TRIGGERED for:", repr(message))
            return  # ← stop here — instant response

        # If fast-path didn't trigger, show why (temporary debug help)
        yield "DEBUG: Fast-path did NOT trigger\n"
        yield f"DEBUG: has_what={has_what}, has_time={has_time}, is_short={is_short}, no_tz={no_tz}\n\n"

        # ───────────────────────────────────────────────────────────────
        # Normal path: search + LLM
        # ───────────────────────────────────────────────────────────────
        try:
            trigger_keywords = [
                "current", "now", "today", "right now", "latest", "who is", "who's",
                "president", "time", "date", "year", "month", "day", "news", "recent",
                "happening", "update", "202", "election", "olympics", "stock", "price",
                "weather", "war", "score", "result"
            ]

            needs_search = any(kw in lower_msg for kw in trigger_keywords)

            if needs_search:
                yield "Searching...\n\n"
                try:
                    search_results = search.run(message)
                    if len(search_results) > 3500:
                        search_results = search_results[:3500] + "\n[...truncated]"
                except Exception as e:
                    search_results = f"(Search failed: {str(e)})"

            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M %Z')

            system_content = f"""Current real-world date and time: {current_time_str}

Most recent search results (use these for current/factual answers — they override older knowledge):
{search_results if search_results else "No search was performed for this query."}
"""

            messages = [
                {"role": "system", "content": system_content},
                *conversations[session_id][:-1],
                {"role": "user", "content": message}
            ]

            for chunk in llm.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    full_response += content
                    yield content

            conversations[session_id].append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_msg = f"\n\nError: {str(e)}"
            yield error_msg

    return Response(generate(), mimetype="text/plain")

@app.route("/clear", methods=["POST"])
def clear():
    session_id = request.json.get("session_id", "default")
    if session_id in conversations:
        del conversations[session_id]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)