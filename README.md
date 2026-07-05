# SHL Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL assessments through natural dialogue.

**Live API:** https://shl-recommender-xbot.onrender.com/health

---

## What it does

Takes a hiring manager from a vague intent ("I need to hire a Java developer") to a grounded shortlist of SHL assessments through multi-turn conversation. The agent:

- **Clarifies** vague queries before recommending anything
- **Recommends** 1–10 assessments with real catalog URLs once it has enough context
- **Refines** the shortlist when constraints change mid-conversation
- **Compares** assessments when asked, grounded in catalog data only
- **Refuses** off-topic requests and prompt injection attempts

---

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 3 assessments for a mid-level Java developer with stakeholder skills.",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

The API is **stateless** — every call carries the full conversation history.

---

## Architecture

```
POST /chat
    │
    ├── Keyword retrieval (catalog_search.py)
    │     └── Token overlap + synonym expansion → top 15 candidates
    │
    ├── LLM call (Groq llama-3.3-70b-versatile)
    │     └── System prompt encodes 4 behaviors + scope rules
    │     └── Candidates injected as grounding context
    │
    └── Hallucination guard
          └── Every recommended name verified against real catalog
          └── Hallucinated names silently dropped before response
```

**Stack:** FastAPI · Uvicorn · Groq API · Python 3.11 · Render (free tier)

**Retrieval:** Lightweight keyword + synonym scoring over 84 scraped SHL Individual Test Solutions. No vector DB needed at this catalog size — defensible, zero latency, zero embedding API cost.

---

## Local Setup

```bash
git clone https://github.com/amn-00/shl-recommender.git
cd shl-recommender

pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY to .env

uvicorn main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API docs.

**Run smoke tests:**
```bash
python test_local.py
```

---

## Project Structure

```
├── main.py            # FastAPI app, /health and /chat endpoints
├── agent.py           # Agent logic, LLM call, hallucination guard
├── catalog_search.py  # Keyword retrieval over catalog
├── catalog.json       # 84 scraped SHL Individual Test Solutions
├── requirements.txt
├── .env.example
└── test_local.py      # 5 smoke tests (clarify, recommend, refine, refuse, injection)
```

---

## Author

**Aman Chaudhary** — M.Tech CS (AI & Robotics), Gautam Buddha University 2026  
Built as a take-home assignment for SHL Labs AI Intern role.
