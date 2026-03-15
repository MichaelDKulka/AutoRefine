"""Flask chatbot with AutoRefine feedback widget embedded in responses.

Shows how to embed the feedback widget in a real web app. Each bot
response includes thumbs up/down buttons that POST feedback back to
the AutoRefine dashboard.

Run::

    pip install flask autorefine
    python examples/flask_app_with_widget.py

Then open http://localhost:5000 in your browser.
The AutoRefine dashboard runs on port 8787 in the background.
"""

from __future__ import annotations

from flask import Flask, request, render_template_string
from autorefine import AutoRefine

# ── Setup ────────────────────────────────────────────────────────────

SYSTEM = "You are a concise, helpful assistant. Answer in 2-3 sentences."

app = Flask(__name__)

client = AutoRefine(
    api_key="sk-your-openai-key",
    model="gpt-4o",
    refiner_key="sk-ant-your-claude-key",
    auto_learn=True,
    refine_threshold=5,
)
client.set_system_prompt(SYSTEM)

# Start the dashboard in the background (serves the widget endpoint)
client.start_dashboard(port=8787)

# ── HTML template ────────────────────────────────────────────────────

TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<title>AutoRefine Chat</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; }
h1 { color: #0969da; font-size: 22px; }
.msg { margin: 16px 0; padding: 12px 16px; border-radius: 8px; }
.user { background: #ddf4ff; }
.bot { background: #f6f8fa; border: 1px solid #d0d7de; }
.bot-text { margin-bottom: 8px; }
form { display: flex; gap: 8px; }
input[type=text] { flex: 1; padding: 8px 12px; border: 1px solid #d0d7de; border-radius: 6px; font-size: 14px; }
button[type=submit] { padding: 8px 16px; border: none; border-radius: 6px; background: #0969da; color: #fff; cursor: pointer; }
.footer { margin-top: 24px; font-size: 12px; color: #656d76; }
.footer a { color: #0969da; }
</style>
</head>
<body>
<h1>AutoRefine Chat Demo</h1>
<p style="color:#656d76;font-size:13px">
  Each response has feedback buttons powered by AutoRefine.
  Your feedback improves the bot automatically!
</p>

{% for msg in messages %}
  {% if msg.role == 'user' %}
    <div class="msg user"><b>You:</b> {{ msg.text }}</div>
  {% else %}
    <div class="msg bot">
      <div class="bot-text"><b>Bot:</b> {{ msg.text }}</div>
      {{ msg.widget | safe }}
    </div>
  {% endif %}
{% endfor %}

<form method="POST" action="/">
  <input type="text" name="message" placeholder="Type a message..." autofocus>
  <button type="submit">Send</button>
</form>

<div class="footer">
  Powered by <a href="https://upwelldigitalsolutions.com" target="_blank">AutoRefine</a>
  | <a href="http://localhost:8787" target="_blank">Dashboard</a>
</div>
</body>
</html>
"""

# ── Routes ───────────────────────────────────────────────────────────

conversation: list[dict] = []


@app.route("/", methods=["GET", "POST"])
def index():
    global conversation

    if request.method == "POST":
        user_msg = request.form.get("message", "").strip()
        if user_msg:
            conversation.append({"role": "user", "text": user_msg})

            # Call AutoRefine
            chat_msgs = [{"role": "user", "content": m["text"]}
                         for m in conversation if m["role"] == "user"]
            resp = client.chat(SYSTEM, chat_msgs)

            # Get the feedback widget HTML for this response
            widget_html = client.get_widget_html(resp.id, style="standard")

            conversation.append({
                "role": "bot",
                "text": resp.text,
                "widget": widget_html,
            })

    return render_template_string(TEMPLATE, messages=conversation)


if __name__ == "__main__":
    print("Starting Flask app on http://localhost:5000")
    print("Dashboard running on http://localhost:8787")
    app.run(port=5000, debug=False)
