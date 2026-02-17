# GenChat - AI Chat Assistant

A real-time chat app powered by Groq (Llama 3.3), Flask, and DuckDuckGo search.

<img width="1911" height="970" alt="image" src="https://github.com/user-attachments/assets/261ff2c7-c9ec-4d5d-989e-58d314742148" />


## Features
- Live streaming responses
- Web search for current info
- Basic user sessions

## Live Demo
(Coming soon – deploying to Render/PythonAnywhere)

## Tech Stack
- Backend: Flask + LangChain + Groq API
- Search: DuckDuckGo
- Frontend: Simple HTML/CSS/JS chat UI

## Setup (local)
```bash
git clone https://github.com/ShawnM20/genchat-assistant-.git
cd genchat-assistant-
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
# Add .env with GROQ_API_KEY=your_key
python app.py
