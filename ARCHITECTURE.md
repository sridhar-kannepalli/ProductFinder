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

**Trigger:** Any new message submitted by the user OR clicking a "Select ID" button in the chat.

**What happens:**
1. **Button Interception**: In the history rendering loop, if a message contains `selectable_products`, the app renders a set of **Streamlit Buttons** (e.g., *"Select ID: 20"*). Clicking these immediately updates `st.session_state` and jumps to the **Step 3 Product View**.
2. **Flexible ID Matching**: A helper function `get_id_from_text` uses a regex `r'(?i)\b(?:id\s*)?(\d+)\b'` to extract IDs from strings.
3. **Regex Interception**: The user's chat input is checked for an explicit "Id [number]" command. If found, it jumps to the product details.
4. **Router Chain**: Classifies the intent (QUESTION, CATEGORY, or FLOW).
5. **QA Chain with Auto-Linking**: Handles availability and fuzzy matching. 
   - **Interactive Q&A**: If the AI's response contains product IDs (e.g., *"ID: 15"*), the app automatically extracts them and provides **"Select ID"** buttons below the answer, allowing the user to jump straight to product details from a question.

---

#### ▶ Global Interactions (Persistent Footer)
**Trigger:** Always rendered after the chat history loop.
- **🔄 Search different product**: Resets the app state to **Step 1** (Category Selection), clears conversation context, and allows browsing a different category.
- **❓ I have a question**: Prompts the user to ask catalog-related questions (features, pricing, availability).

---

#### ▶ Product Display & Interactions
**Trigger:** Any message with a `product` payload (Step 2/3).
- **Rich Product Card**: Displays title, price, image, and description.
- **Add to cart**: Confirms the purchase (specific to the product card).

---

#### ▶ Step 1 — Category Discovery
**Trigger:** Start of the flow OR clicking "Search different product."
- **Automated Category Filtration**: To ensure a clean experience, the app filters categories from the raw API response using three criteria:
  1. **Usage Check**: Only categories actually associated with at least one product in the current catalog are included.
  2. **Sanitization**: Categories with names containing "test", "junk", or "category" are excluded.
  3. **Data Quality**: Explicitly excludes known misspelled or placeholder items (e.g., "grosery").
- **Interactive Buttons**: Instead of text prompts, the bot renders a **3-column grid of buttons** representing each valid category.
- **One-Click Filtering**: Clicking a button immediately:
  1. Sets `st.session_state.category`.
  2. Pre-fetches the matching products from the full catalog.
  3. Transitions to **Step 2** (Product List).
- **Manual Input**: If the user types a category manually, the `category_chain` still extracts it as a fallback.

#### ▶ Step 2 — Product ID Selection
**Trigger:** After Step 1 displays the product list.
- User enters a numeric ID or "Id [number]", or clicks a "Select ID" button.
- The app uses `get_id_from_text` to parse the input or handles the button click event.
- If match found, transitions to **Step 3**.


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
├── main.py              # Main application (routing, LLM, and flow logic)
├── myenv/               # Project virtual environment (conda)
├── requirements.txt     # Python dependencies
└── ARCHITECTURE.md      # This document
```

---

## Key Design Decisions

- **Global Routing Interception:** Employs an LLM intent router at the very top of the submission block. If the message is a `QUESTION`, it handles the QA and calls `st.rerun()`. This elegant pattern gives conversational flexibility while preserving the stateful Step Machine underneath.
- **Local Catalog Cache:** Fetches `https://api.escuelajs.co/api/v1/products` into an `st.cache_data` singleton so questions can be answered instantaneously without repeated expensive external API hits.
- **Local LLM:** Using Ollama keeps all AI inference on-device, meaning no API keys or cloud costs.
- **Step Machine:** A simple integer `step` drives the structured conversation flow, making the "happy path" logic easy to follow and extend.
