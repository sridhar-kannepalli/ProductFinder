import streamlit as st
import requests
import re
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 1. Page Config
st.set_page_config(page_title="Product Finder Chatbot", page_icon="🛒")

st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Product Finder AI")

# 2. Local LLM Setup
@st.cache_resource
def get_llm():
    return OllamaLLM(model="qwen2.5vl")

@st.cache_data(ttl=300)
def fetch_all_products_cached():
    res = requests.get("https://api.escuelajs.co/api/v1/products")
    if res.status_code == 200:
        return res.json()
    return []

@st.cache_data(ttl=3600)
def fetch_categories_cached():
    # 1. Fetch all products to see which categories are actually in use
    products = fetch_all_products_cached()
    used_category_ids = set(p.get('category', {}).get('id') for p in products if p.get('category'))
    
    # 2. Fetch all categories
    res = requests.get("https://api.escuelajs.co/api/v1/categories")
    if res.status_code == 200:
        all_cats = res.json()
        
        # 3. Filter criteria:
        # - Must have active products (ID must be in used_category_ids)
        # - Exclude "test", "grosery", "junk", etc.
        # - Priority for standard names: Clothes, Electronics, Furniture, Shoes, Miscellaneous
        
        filtered = []
        exclude_terms = ["test", "grosery", "junk", "category"]
        
        for c in all_cats:
            name = c.get('name', '').lower().strip()
            cid = c.get('id')
            
            # Check if used and not junk
            is_used = cid in used_category_ids
            is_junk = any(term in name for term in exclude_terms)
            
            if is_used and not is_junk:
                filtered.append(c)
        
        return filtered
    return []

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Error loading local LLM: {e}")
    st.stop()

# Helper for flexible ID matching
def get_id_from_text(text):
    if not text:
        return None
    # Matches just numeric "12" or "Id 12" / "ID 12"
    match = re.search(r'(?i)\b(?:id\s*)?(\d+)\b', text.strip())
    if match:
        return int(match.group(1))
    return None
    
# Chains
category_template = """You are an intent extraction AI. Find the product category from the user request.
Possible categories: 'clothes', 'electronics', 'furniture', 'shoes', 'miscellaneous'
If the user's intent doesn't strictly match one, pick the closest one from the list. 
If it is ambiguous, pick 'electronics'.
Reply ONLY with the exact category name. Nothing else.

User Request: {user_request}

Category:"""
category_prompt = PromptTemplate(template=category_template, input_variables=["user_request"])
category_chain = category_prompt | llm

intent_template = """You are an intent extraction AI. The user was just viewing a product.
They can either choose to view another product in the same category, or start all over again.
Reply exactly 'ANOTHER' if they want to view another product.
Reply exactly 'START_OVER' if they want to start all over again.
Reply ONLY with 'ANOTHER' or 'START_OVER'. Nothing else.

User Request: {user_request}

Intent:"""
intent_prompt = PromptTemplate(template=intent_template, input_variables=["user_request"])
intent_chain = intent_prompt | llm

router_template = """You are an intent router for a shopping assistant.
Determine the user's intent from the following options:
1. 'QUESTION': Asking "Do you have...", "Do you sell...", "Is X available?", or any product-specific inquiry like "What is the price of the laptop?".
2. 'CATEGORY': Asking to "browse clothes", "show me shoes", "see electronics", or "list furniture".
3. 'FLOW': Making a numeric selection (like an ID), or navigating (e.g., 'start over', 'another').

STRICT RULE: If the user message is "Do you have [item]?", you MUST respond with 'QUESTION'.

Reply ONLY with 'QUESTION', 'CATEGORY', or 'FLOW'.
User Message: {user_request}
Intent:"""
router_prompt = PromptTemplate(template=router_template, input_variables=["user_request"])
router_chain = router_prompt | llm

qa_template = """You are a focused product assistant. Use the following product catalog to answer the user's question about product availability.

# PROCESS:
1. Scan the catalog TITLES and DESCRIPTIONS for the item requested in the "User Question".
2. Use FUZZY MATCHING: If the user asks for "watch", items like "Smartwatch" are POSITIVE hits.
3. If the user asks for "camera", items with "camera" in the title or describing a built-in camera are hits.

# NEGATIVE CONSTRAINTS (CRITICAL):
- **STRICTLY PROHIBITED**: DO NOT list every item in the same category. For example, if asked for "watches", DO NOT list "laptops", "headphones", or "televisions" just because they are in the same 'Electronics' category.
- **ZERO MATCHES**: If NO specific product in the catalog is a match (even partially), say "I am sorry, but that product (camera/watch/etc.) is not available in our catalog." DO NOT show ANY other products.

# RESPONSE FORMAT:
If there are hits:
"Yes, I found the following matching items:
- ID: [ID] | Title: [TITLE] | Price: $[PRICE]"

Product Catalog:
{catalog}

Chat History:
{history}

User Question: {user_request}
Answer:"""
qa_prompt = PromptTemplate(template=qa_template, input_variables=["catalog", "history", "user_request"])
qa_chain = qa_prompt | llm


# State Init
if "step" not in st.session_state:
    st.session_state.step = 1
if "messages" not in st.session_state:
    cats = fetch_categories_cached()
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome! What kind of product are you looking for? Please select a category below:",
            "selectable_categories": cats
        }
    ]
if "category" not in st.session_state:
    st.session_state.category = None
if "products" not in st.session_state:
    st.session_state.products = []
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None


# Render history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # New: Render selectable product buttons if available
        if "selectable_products" in msg:
            for p in msg["selectable_products"]:
                # Render a button for each product
                if st.button(f"Select ID: {p['id']} | {p['title']}", key=f"sel_{p['id']}_{idx}"):
                    st.session_state.selected_product = p
                    st.session_state.step = 3
                    
                    # Add confirmation message to chat
                    reply = f"You selected **{p['title']}** (ID: {p['id']}). Here are the details. Would you like to view another product, or start all over again?"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "product": p
                    })
                    st.rerun()

        # New: Render selectable category buttons if available
        if "selectable_categories" in msg:
            cols = st.columns(3)
            for cidx, cat in enumerate(msg["selectable_categories"]):
                with cols[cidx % 3]:
                    if st.button(cat['name'], key=f"catbtn_{cat['id']}_{idx}"):
                        st.session_state.category = cat['name']
                        st.session_state.step = 2
                        
                        # Pre-fetch products
                        all_products = fetch_all_products_cached()
                        st.session_state.products = [p for p in all_products if str((p.get('category') or {}).get('name') or '').lower().strip() == cat['name'].lower().strip()]
                        
                        # Add confirmation to history
                        reply = f"I found **{len(st.session_state.products)}** products in '{cat['name']}'. Please click a button below or enter the ID number to see details."
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": reply,
                            "selectable_products": st.session_state.products
                        })
                        st.rerun()

        # If this is a message where we showed a product (appended specially as dict with product details), we can render it.
        if "product" in msg:
            p = msg["product"]
            st.write(f"### {p['title']}")
            st.write(f"**Price:** ${p['price']}")
            if p.get('images') and len(p['images']) > 0:
                img_url = p['images'][0]
                
                # Workaround for EscuelaJS occasional double-stringified arrays
                if isinstance(img_url, str) and img_url.startswith('["'):
                    import json
                    try:
                        img_url = json.loads(img_url)[0]
                    except:
                        pass
                
                st.image(img_url, width=200)
            st.write(f"**Description:** {p['description']}")
            
            if st.button("Add to cart", key=f"cart_{p['id']}_{idx}"):
                st.success("Thank you for your purchase!")


# Always show global actions at the bottom of the current chat state
if len(st.session_state.messages) > 0:
    st.write("---")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        if st.button("🔄 Search different product", key="global_reset"):
            st.session_state.step = 1
            st.session_state.category = None
            st.session_state.products = []
            st.session_state.selected_product = None
            cats = fetch_categories_cached()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Starting over! What kind of product are you looking for? Please select a category:",
                "selectable_categories": cats
            })
            st.rerun()
    with gcol2:
        if st.button("❓ I have a question", key="global_question"):
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm here to help! Please ask any question about the products I sell (e.g., about features, prices, or availability)."
            })
            st.rerun()

if user_input := st.chat_input("Type your message here..."):
    # Append user's text
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    prod_id = get_id_from_text(user_input)
    # Check if this was a standalone ID or "Id X" command
    # Only trigger if it's explicitly "Id X" or if we are in Step 2
    is_explicit_id = re.search(r'(?i)\bid\s*\d+\b', user_input)
    
    if is_explicit_id:
        catalog = fetch_all_products_cached()
        product = next((p for p in catalog if p['id'] == prod_id), None)
        
        if product:
            st.session_state.selected_product = product
            st.session_state.step = 3
            
            reply = "Here are the product details. Would you like to view another product, or start all over again?"
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "product": product
            })
            st.rerun()
        else:
            with st.chat_message("assistant"):
                reply = f"There is no product with ID {prod_id}. Please start over and select a category again."
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            
            st.session_state.step = 1
            st.session_state.category = None
            st.session_state.products = []
            st.session_state.selected_product = None
            st.rerun()
            
    route = "FLOW"
    try:
        route = router_chain.invoke({"user_request": user_input}).strip().upper()
    except Exception:
        pass
        
    if "CATEGORY" in route:
        st.session_state.step = 1
        
    if "QUESTION" in route:
        with st.chat_message("assistant"):
            with st.spinner("Searching catalog..."):
                catalog = fetch_all_products_cached()
                catalog_str = "\n".join([f"ID: {p['id']} | Title: {p['title']} | Price: ${p['price']} | Category: {(p.get('category') or {}).get('name', '')} | Desc: {p.get('description', '')[:100]}..." for p in catalog])
                
                # Build history (limit to last 4 messages, exclude current)
                hist_items = []
                for m in st.session_state.messages[-5:-1]:
                    if m.get("content"):
                        hist_items.append(f"{m['role'].capitalize()}: {m['content']}")
                hist_str = "\n".join(hist_items)
                
                answer = str(qa_chain.invoke({
                    "catalog": catalog_str,
                    "history": hist_str,
                    "user_request": user_input
                }))
                
                # Extract IDs for selectable buttons
                found_ids = re.findall(r'(?i)ID:\s*(\d+)', answer)
                selectable = []
                if found_ids:
                    found_ids = [int(i) for i in found_ids]
                    selectable = [p for p in catalog if p['id'] in found_ids]
                
                st.write(answer)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "selectable_products": selectable
                })
        st.rerun()
        
    if st.session_state.step == 1:
        with st.chat_message("assistant"):
            with st.spinner("Finding Category..."):
                response_text = category_chain.invoke({"user_request": user_input})
                category_raw = response_text.strip().lower()
                
                valid_cat = None
                for vc in ['clothes', 'electronics', 'furniture', 'shoes', 'miscellaneous']:
                    if vc in category_raw:
                        valid_cat = vc
                        break
                if not valid_cat:
                    valid_cat = "electronics" # fallback
                    
                res = requests.get("https://api.escuelajs.co/api/v1/products")
                if res.status_code == 200:
                    all_products = res.json()
                    st.session_state.products = [p for p in all_products if str((p.get('category') or {}).get('name') or '').lower().strip() == str(valid_cat or '').lower().strip()]
                
                if len(st.session_state.products) > 0:
                    st.session_state.category = valid_cat
                    st.session_state.step = 2
                    
                    reply = f"I found **{len(st.session_state.products)}** products in '{valid_cat}'. Please click a button below or enter the ID number to see details."
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply,
                        "selectable_products": st.session_state.products
                    })
                    st.rerun()
                else:
                    err_reply = f"Sorry, I couldn't fetch products for '{valid_cat}'. Could you try again?"
                    st.session_state.messages.append({"role": "assistant", "content": err_reply})
                    st.rerun()
                    
    elif st.session_state.step == 2:
        prod_id = get_id_from_text(user_input)
        if prod_id is not None:
            res_products = requests.get("https://api.escuelajs.co/api/v1/products")
            product = None
            if res_products.status_code == 200:
                dataset = res_products.json()
                product = next((p for p in dataset if p['id'] == prod_id), None)
            
            if product:
                st.session_state.selected_product = product
                st.session_state.step = 3
                
                reply = "Here are the product details. Would you like to view another product, or start all over again?"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "product": product
                })
                st.rerun()
            else:
                with st.chat_message("assistant"):
                    reply = "Invalid ID. Please enter a valid product ID from the list."
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
        else:
            with st.chat_message("assistant"):
                reply = "Please enter a valid numeric ID (e.g. 12) or 'Id 12'."
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
    elif st.session_state.step == 3:
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response_text = intent_chain.invoke({"user_request": user_input})
                intent = response_text.strip().upper()
                
                if "START_OVER" in intent or "START OVER" in intent or "ALL OVER" in user_input.upper():
                    st.session_state.step = 1
                    st.session_state.category = None
                    st.session_state.products = []
                    st.session_state.selected_product = None
                    
                    cats = fetch_categories_cached()
                    reply = "Starting over! What kind of product are you looking for? Please select a category:"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply,
                        "selectable_categories": cats
                    })
                    st.rerun()
                else:
                    st.session_state.step = 2
                    reply = f"Here are the products in '{st.session_state.category}' again. Please click a button below or enter the ID."
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply,
                        "selectable_products": st.session_state.products
                    })
                    st.session_state.selected_product = None
                    st.rerun()
