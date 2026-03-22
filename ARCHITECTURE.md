# Product Finder AI — Architecture & Design

## Overview

**Product Finder AI** is a conversational Streamlit web application that guides users through discovering and exploring products. It uses a locally-hosted LLM (`qwen2.5vl` via Ollama) as its AI backbone and integrates with the [EscuelaJS Fake Store API](https://api.escuelajs.co/) to retrieve real product data.

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io) |
| LLM Runtime | [Ollama](https://ollama.com) (`qwen2.5vl`) |
| LLM Orchestration | [LangChain](https://www.langchain.com) (`langchain_core`, `langchain_ollama`) |
| Product Data Source | [EscuelaJS API](https://api.escuelajs.co/api/v1/products) |
| Language | Python 3 |

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User (Browser)                        │
└────────────────────────┬─────────────────────────────────┘
                         │ Chat Input / Messages
┌────────────────────────▼─────────────────────────────────┐
│               Streamlit Application (main.py)            │
│                                                          │
│  ┌─────────────────┐      ┌──────────────────────────┐  │
│  │  LangChain       │      │   EscuelaJS API Client   │  │
│  │  Prompt Chains   │      │   (requests library)     │  │
│  │  - Router        │      │   /api/v1/products       │  │
│  │  - QA Chain      │      │                          │  │
│  │  - Category      │      └──────────────┬───────────┘  │
│  │  - Intent        │                     │              │
│  └────────┬────────┘                      │              │
│           │                               │              │
│  ┌────────▼────────┐                      │              │
│  │  Ollama LLM     │                      │              │
│  │  (qwen2.5vl)    │                      │              │
│  └─────────────────┘                      │              │
│                                           │              │
│  ┌────────────────────────────────────────▼──────────┐  │
│  │           Streamlit Session State                  │  │
│  │  step | messages | category | products | selected  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Components

### 1. LLM Setup and Cache

The app uses `OllamaLLM` to connect to a locally running `qwen2.5vl` model, cached with `@st.cache_resource`. The product catalog is fetched from the EscuelaJS API and cached globally via `@st.cache_data` to ensure fast Q&A routing without hitting the external API on every prompt.

```python
@st.cache_resource
def get_llm():
    return OllamaLLM(model="qwen2.5vl")

@st.cache_data(ttl=300)
def fetch_all_products_cached():
    # Fetches all products and caches them for 5 minutes
    ...
```

---

### 2. LangChain Prompt Chains

Four prompt chains process different parts of the conversational flow:

#### `router_chain`
- **Purpose:** Identifies if the user is asking a general product question (`QUESTION`) or engaging in standard shopping flow (`FLOW`).
- **Output:** Exactly `QUESTION` or `FLOW`.

#### `qa_chain`
- **Purpose:** Answers general user questions using the cached product catalog mapping and the last 4 chat messages as context.

#### `category_chain`
- **Purpose:** Extracts a product category from free-form user input.
- **Valid Categories:** `clothes`, `electronics`, `furniture`, `shoes`, `miscellaneous`
- **Fallback:** Returns `electronics` if intent is ambiguous.
- **Output:** Exactly one category string.

#### `intent_chain`
- **Purpose:** Determines user intent after viewing a product — whether they want to see another product or start over.
- **Output:** Exactly `ANOTHER` or `START_OVER`.

---

### 3. Session State

Streamlit's `st.session_state` persists state across reruns within a session:

| Key | Type | Purpose |
|---|---|---|
| `step` | `int` | Current step in the conversational flow (1, 2, or 3) |
| `messages` | `list[dict]` | Full chat history, including optional `product` payloads |
| `category` | `str` | Currently selected product category |
| `products` | `list[dict]` | Products fetched and filtered for the selected category |
| `selected_product` | `dict` | The last product the user selected |

---

### 4. Conversational Flow

The app combines an **Intent Router** with a **3-step state machine** driven by `st.session_state.step`. On every Streamlit rerun, the app reads the user's message, routes it appropriately, and explicitly handles branch logic.

#### ▶ Global Intent Routing (The Q&A Escape Hatch)

**Trigger:** Any new message submitted by the user.

**What happens:**
1. The message is checked against a regex: `r'(?i)\bid\s*(\d+)\b'`.
   - If a match is found (e.g., *"id 20"*), the app fetches the catalog, matches the ID, and immediately jumps to the **Step 2/3 Product View**. 
   - If the ID does not exist, the app notifies the user and resets to **Step 1**.
2. If no direct ID is found, the message is passed to `router_chain`.
   - **STRICT RULE**: "Do you have..." or "Is X available?" questions are strictly routed to `QUESTION`.
   - If the user is asking a general question:
     - The app fetches `fetch_all_products_cached()` and formats it into a compressed string.
     - The app passes the catalog string and chat history to `qa_chain` (which uses Fuzzy Matching for items like "smartwatch" vs "watch").
     - The answer is printed and saved to history, and the script calls `st.rerun()`.
3. If the user asks to "browse shoes" or generic category switches, the intent is `CATEGORY`, and the app resets `step` to `1`.

---

#### ▶ Step 1 — Category Discovery
**Trigger:** Start of the flow, assuming the router detected `FLOW` or a `CATEGORY` switch.
- User types a free-form product request (e.g. *"I'm looking for a phone"*).
- `category_chain` extracts the matching category (clothes, electronics, furniture, shoes, miscellaneous).
- The app fetches `GET https://api.escuelajs.co/api/v1/products` and locally filters for products where `category.name` matches.
- The list of product IDs and titles is displayed to the user.
- Transitions to **Step 2**.

#### ▶ Step 2 — Product ID Selection
**Trigger:** After Step 1 displays the product list and the user types a numeric ID.
- The app attempts to parse input as an integer.
- The app fetches the **complete product catalogue** from `GET https://api.escuelajs.co/api/v1/products`.
- It iterating through the full dataset to find a matching `product['id']`.
- If matched, fields are extracted (utilizing the `images` array), saved to `st.session_state.selected_product`, and rendered in chat alongside an Add to Cart button.
- Transitions to **Step 3**.

#### ▶ Step 3 — Post-Viewing Intent
**Trigger:** After rendering the product card, user responds with an action.
- `intent_chain` classifies intent.
- **`ANOTHER` path:** `step` resets to `2`, current category product list re-displayed.
- **`START_OVER` path:** All session state resets to step `1`.

---

### 5. Chat Message Rendering

The chat history (`st.session_state.messages`) is rendered on every rerun. Each message is a dictionary with:
- `role`: `"user"` or `"assistant"`
- `content`: Text of the message
- `product` *(optional)*: Triggers rich product card rendering (Title, Price, First Image, Description, Add to Cart Button).

---

## API Integration

### EscuelaJS API Endpoints Used

| Endpoint | Usage |
|---|---|
| `GET /api/v1/products` | Cached globally to supply context to the Global Q&A chain. Also used to locally filter for categories in Step 1, and match by ID in Step 2. |

---

## File Structure

```
ProductFinder/
├── myenv/
│   └── main.py          # Main application (routing, LLM, and flow logic)
├── requirements.txt     # Python dependencies
└── ARCHITECTURE.md      # This document
```

---

## Key Design Decisions

- **Global Routing Interception:** Employs an LLM intent router at the very top of the submission block. If the message is a `QUESTION`, it handles the QA and calls `st.rerun()`. This elegant pattern gives conversational flexibility while preserving the stateful Step Machine underneath.
- **Local Catalog Cache:** Fetches `https://api.escuelajs.co/api/v1/products` into an `st.cache_data` singleton so questions can be answered instantaneously without repeated expensive external API hits.
- **Local LLM:** Using Ollama keeps all AI inference on-device, meaning no API keys or cloud costs.
- **Step Machine:** A simple integer `step` drives the structured conversation flow, making the "happy path" logic easy to follow and extend.
