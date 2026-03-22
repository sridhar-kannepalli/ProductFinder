# Product Finder AI — Architecture & Design

## Overview

**Product Finder AI** is a conversational Streamlit web application that guides users through discovering and exploring products. It uses a locally-hosted LLM (`qwen2.5-vl` via Ollama) as its AI backbone and integrates with the [Fake Store API](https://fakestoreapi.com) to retrieve real product data.

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io) |
| LLM Runtime | [Ollama](https://ollama.com) (`qwen2.5-vl`) |
| LLM Orchestration | [LangChain](https://www.langchain.com) (`langchain_core`, `langchain_ollama`) |
| Product Data Source | [Fake Store API](https://fakestoreapi.com) |
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
│  │  LangChain       │      │   Fake Store API Client  │  │
│  │  Prompt Chains   │      │   (requests library)     │  │
│  │  - Category      │      │   /products/category/:c  │  │
│  │  - Intent        │      │   /products              │  │
│  └────────┬────────┘      └──────────────┬───────────┘  │
│           │                              │               │
│  ┌────────▼────────┐                     │               │
│  │  Ollama LLM     │                     │               │
│  │  (qwen2.5-vl)   │                     │               │
│  └─────────────────┘                     │               │
│                                          │               │
│  ┌───────────────────────────────────────▼───────────┐  │
│  │           Streamlit Session State                  │  │
│  │  step | messages | category | products | selected  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Components

### 1. LLM Setup (`get_llm`)

The app uses `OllamaLLM` to connect to a locally running `qwen2.5-vl` model. It is cached with `@st.cache_resource` to avoid re-loading the model on every Streamlit rerun.

```python
@st.cache_resource
def get_llm():
    return OllamaLLM(model="qwen2.5-vl")
```

---

### 2. LangChain Prompt Chains

Two prompt chains are defined using `PromptTemplate` and piped to the LLM:

#### `category_chain`
- **Purpose:** Extracts a product category from free-form user input.
- **Valid Categories:** `electronics`, `jewelery`, `men's clothing`, `women's clothing`
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
| `products` | `list[dict]` | Products fetched for the selected category |
| `selected_product` | `dict` | The last product the user selected |

---

### 4. Conversational Flow (Step Machine)

The app operates as a **3-step state machine** driven by `st.session_state.step`. On every Streamlit rerun, the app reads the current step value and branches its logic accordingly.

```
  Step 1: Category Discovery
       │
       ▼
  Step 2: Product ID Selection
       │
       ▼
  Step 3: Post-Viewing Intent
       │
       ├──► "Another" ──► Step 2
       └──► "Start over" ──► Step 1
```

---

#### ▶ Step 1 — Category Discovery

**Trigger:** Application start, or when the user chooses to start over.

**What happens:**
1. The user submits a free-form natural language request (e.g. *"I'm looking for a phone"* or *"show me some rings"*).
2. The user's message is appended to `st.session_state.messages` and displayed in the chat.
3. A spinner is shown while the `category_chain` LangChain chain is invoked:
   - The prompt instructs the LLM to extract the closest matching category from the fixed list: `electronics`, `jewelery`, `men's clothing`, `women's clothing`.
   - The LLM responds with exactly one category name.
4. The raw LLM response is lowercased and checked against the valid category list. If no match is found, the app defaults to `electronics` as a safe fallback.
5. The app calls `GET https://fakestoreapi.com/products/category/{category}`:
   - On success (HTTP 200 with results): products are stored in `st.session_state.products`, the category is saved to `st.session_state.category`, and `step` is set to `2`.
   - On failure: an error message is shown and the step remains at `1`.
6. The product list is formatted as `ID | Title` and sent as an assistant message.
7. `st.rerun()` is called to refresh the UI with the updated chat history.

**State changes:** `step` → `2`, `category` set, `products` populated.

---

#### ▶ Step 2 — Product ID Selection

**Trigger:** After Step 1 displays the product list and prompts the user for an ID.

**What happens:**
1. The user types a numeric product ID shown in the list.
2. The app attempts to parse the input as an integer:
   - If the input is not a valid integer, a `ValueError` is caught and the user is prompted to enter a numeric ID.
3. The app fetches the **complete product catalogue** from `GET https://fakestoreapi.com/products` (all products, not just the current category).
4. It iterates through the full dataset to find a product where `product['id'] == prod_id`.
5. If a match is found:
   - The product's `title`, `price`, `description`, and `image` fields are extracted.
   - The product dict is stored in `st.session_state.selected_product`.
   - A message is appended to `st.session_state.messages` with both a text reply and the `product` payload embedded directly.
   - This payload is used during rendering to display the rich product card (image, price, description, Add to Cart button).
   - `step` is set to `3`.
6. If no matching product is found, the user is told to enter a valid ID and the step remains at `2`.
7. `st.rerun()` refreshes the UI.

**State changes:** `step` → `3`, `selected_product` set, product message appended with embedded payload.

---

#### ▶ Step 3 — Post-Viewing Intent

**Trigger:** After Step 2 renders the product card and asks the user what they'd like to do next.

**What happens:**
1. The user types a follow-up response (e.g. *"show me another one"*, *"I want to start over"*, *"let me look at something else"*).
2. A spinner is shown while the `intent_chain` is invoked:
   - The prompt instructs the LLM to classify the user's intent as exactly `ANOTHER` or `START_OVER`.
3. The LLM's response is uppercased and checked:
   - **`ANOTHER` path:** `step` is set back to `2`. The existing product list for the current category is re-displayed, and `selected_product` is cleared. The user can pick a different ID.
   - **`START_OVER` path:** All session state is reset — `step` → `1`, `category` → `None`, `products` → `[]`, `selected_product` → `None`. The welcome prompt is re-sent.
4. `st.rerun()` refreshes the UI with the new state.

> **Note:** The app also checks explicitly for phrases like *"start over"* or *"all over"* directly in the raw user input as an extra safeguard, in case the LLM response is ambiguous.

**State changes (ANOTHER):** `step` → `2`, `selected_product` cleared.  
**State changes (START_OVER):** All state reset, `step` → `1`.

---

### 5. Chat Message Rendering

The chat history (`st.session_state.messages`) is rendered on every rerun. Each message is a dictionary with:

- `role`: `"user"` or `"assistant"`
- `content`: Text of the message
- `product` *(optional)*: A product dict, which triggers rich product card rendering:
  - Product title (`###` header)
  - Price (bold)
  - Product image (`st.image`)
  - Description
  - **Add to Cart** button (with unique key `cart_{id}_{idx}` to avoid Streamlit key conflicts)

---

## API Integration

### Fake Store API Endpoints Used

| Endpoint | Usage |
|---|---|
| `GET /products/category/{category}` | Fetch products by category in Step 1 |
| `GET /products` | Fetch full product dataset for ID matching in Step 2 |

---

## Data Flow Diagram

```
User Input
    │
    ▼
[Step 1] category_chain (LLM)
    │ → category string
    ▼
GET /products/category/{category}
    │ → product list (id + title)
    ▼
Displayed in chat

User enters ID
    │
    ▼
[Step 2] GET /products (full dataset)
    │ → match by ID
    ▼
Product card (title, price, image, description)
    │ + Add to Cart button
    ▼

User responds
    │
    ▼
[Step 3] intent_chain (LLM)
    │ → ANOTHER or START_OVER
    ▼
Transition accordingly
```

---

## File Structure

```
ProductFinder/
├── myenv/
│   └── main.py          # Main application (all logic lives here)
├── requirements.txt     # Python dependencies
└── ARCHITECTURE.md      # This document
```

---

## Key Design Decisions

- **Local LLM:** Using Ollama keeps all AI inference on-device, meaning no API keys or cloud costs.
- **Step Machine:** A simple integer `step` drives the entire conversation flow, making the logic easy to follow and extend.
- **Full Dataset Fetch for ID Matching:** In Step 2, the app fetches `GET /products` (the full catalogue) rather than relying on the previously fetched category subset. This makes the ID lookup robust and consistent with the complete product catalogue.
- **Message-embedded Products:** Product data is stored directly inside the `messages` list rather than in a separate state key, ensuring product cards are re-rendered correctly for every past interaction in the chat history.
- **Unique Button Keys:** `Add to Cart` buttons use `cart_{product_id}_{message_index}` as their key to avoid Streamlit duplicate widget key errors across reruns.
