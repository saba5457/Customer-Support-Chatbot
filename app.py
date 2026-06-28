from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv("../LangChain/.env")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly customer support agent for WOW Meal N Deal restaurant in Lahore.

RESTAURANT INFO:
- Name: WOW Meal N Deal
- Rating: 4.5/5 (2,039 reviews)
- Location: 22 R2 Johar Town Rd, Block R2, Johar Town, Lahore
- Phone: 0307 0001200
- Hours: Open until 1:30 AM
- Price range: Rs 1,000-5,000 per person
- Website: wowmealndeal.pk

POPULAR MENU ITEMS:
- Stuffed Chicken (Popular)
- Cocktail Sandwich (Popular)
- Loaded Fries (Popular)
- Moroccan Steak (Popular)
- Stuffed Chicken Breast (Popular)
- Halal Beer (Popular)
- Fillet Burger (Popular)
- Chicken Fingers Combo
- Blue Lagoon Cocktail
- Halal Malt Drink
- Mushroom Steak
- Cheesy Mushroom Sandwich
- Stuffed Cheese Chicken
- Cordon Bleu

SERVICES:
- Dine-in available
- Order pickup available
- Order delivery available

YOUR JOB:
- Answer questions about menu, location, hours, delivery
- Help customers with orders
- Be friendly and welcoming
- Keep responses short and helpful
- If asked something you don't know, say: 'For more details please call 0307 0001200'
- Always respond in the same language the customer uses (Urdu or English)"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{message}")
])

chain = prompt | llm

app = Flask(__name__)
chat_histories = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    message = data.get("message", "")
    
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    history = chat_histories[session_id]
    
    response = chain.invoke({
        "history": history,
        "message": message
    })
    import time
    time.sleep(5)
    reply = response.content
    
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=reply))
    
    return jsonify({"reply": reply})

@app.route("/clear", methods=["POST"])
def clear():
    data = request.json
    session_id = data.get("session_id", "default")
    chat_histories[session_id] = []
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)