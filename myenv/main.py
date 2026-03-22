import streamlit as st
import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
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
    return OllamaLLM(model="qwen2.5-vl")

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Error loading local LLM: {e}")
    st.stop()
    
# Chains
category_template = """You are an intent extraction AI. Find the product category from the user request.
Possible categories: 'electronics', 'jewelery', 'men's clothing', 'women's clothing'
If the user's intent doesn't strictly match one, pick the closest one from the list. 
If it is ambiguous, pick 'electronics'.
Reply ONLY with the exact category name. Nothing else.

User Request: {user_request}

Category:"""
category_prompt = PromptTemplate(template=category_template, input_variables=["user_request"])
category_chain = LLMChain(llm=llm, prompt=category_prompt)

intent_template = """You are an intent extraction AI. The user was just viewing a product.
They can either choose to view another product in the same category, or start all over again.
Reply exactly 'ANOTHER' if they want to view another product.
Reply exactly 'START_OVER' if they want to start all over again.
Reply ONLY with 'ANOTHER' or 'START_OVER'. Nothing else.

User Request: {user_request}

Intent:"""
intent_prompt = PromptTemplate(template=intent_template, input_variables=["user_request"])
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)


# State Init
if "step" not in st.session_state:
    st.session_state.step = 1
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! What kind of product are you looking for? (e.g., electronics, jewelery, men's clothing, women's clothing)"}
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
        
        # If this is a message where we showed a product (appended specially as dict with product details), we can render it.
        if "product" in msg:
            p = msg["product"]
            st.write(f"### {p['title']}")
            st.write(f"**Price:** ${p['price']}")
            st.image(p['image'], width=200)
            st.write(f"**Description:** {p['description']}")
            
            # Since buttons rerender, we should only enable the button if it's the latest product interaction?
            # Or just render the button. 
            # Give it a unique key based on idx and product id.
            if st.button("Add to cart", key=f"cart_{p['id']}_{idx}"):
                st.success("Thank you for your purchase!")


if user_input := st.chat_input("Type your message here..."):
    # Append user's text
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    if st.session_state.step == 1:
        with st.chat_message("assistant"):
            with st.spinner("Finding Category..."):
                response = category_chain.invoke({"user_request": user_input})
                category_raw = response['text'].strip().lower()
                
                valid_cat = None
                for vc in ['electronics', 'jewelery', "men's clothing", "women's clothing"]:
                    if vc in category_raw:
                        valid_cat = vc
                        break
                if not valid_cat:
                    valid_cat = "electronics" # fallback
                    
                res = requests.get(f"https://fakestoreapi.com/products/category/{valid_cat}")
                if res.status_code == 200 and len(res.json()) > 0:
                    st.session_state.products = res.json()
                    st.session_state.category = valid_cat
                    st.session_state.step = 2
                    
                    product_list = "\n".join([f"- **ID:** {p['id']} | {p['title']}" for p in st.session_state.products])
                    reply = f"I found these products in '{valid_cat}':\n\n{product_list}\n\nPlease enter the ID of the product to know more."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                else:
                    err_reply = f"Sorry, I couldn't fetch products for '{valid_cat}'. Could you try again?"
                    st.session_state.messages.append({"role": "assistant", "content": err_reply})
                    st.rerun()
                    
    elif st.session_state.step == 2:
        try:
            prod_id = int(user_input.strip())
            product = next((p for p in st.session_state.products if p['id'] == prod_id), None)
            
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
                    
        except ValueError:
            with st.chat_message("assistant"):
                reply = "Please enter a valid numeric ID."
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
    elif st.session_state.step == 3:
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = intent_chain.invoke({"user_request": user_input})
                intent = response['text'].strip().upper()
                
                if "START_OVER" in intent or "START OVER" in intent or "ALL OVER" in user_input.upper():
                    st.session_state.step = 1
                    st.session_state.category = None
                    st.session_state.products = []
                    st.session_state.selected_product = None
                    
                    reply = "Starting over! What kind of product are you looking for? (e.g., electronics, jewelery, men's clothing, women's clothing)"
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                else:
                    st.session_state.step = 2
                    product_list = "\n".join([f"- **ID:** {p['id']} | {p['title']}" for p in st.session_state.products])
                    reply = f"Here are the products in '{st.session_state.category}' again:\n\n{product_list}\n\nPlease enter the ID of the product to know more."
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.session_state.selected_product = None
                    st.rerun()
