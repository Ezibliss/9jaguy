import asyncio
import os
os.environ["USER_AGENT"] = "naijaguy-app/1.0" # Set a user agent for requests

from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("API_KEY")
from datetime import datetime
import openai
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urlparse
import requests
from langchain.docstore.document import Document
import streamlit as st
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_community.retrievers import WebResearchRetriever
#from langchain_community.tools.ddg_search import DuckDuckGoSearchResults
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
#from prompts import qa_system_prompt, contextualize_qa_system_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, session, redirect
from bs4 import BeautifulSoup
import re
from flask_session import Session
import requests
import os
import time





xai_api_key = os.getenv('XAI_API_KEY')
#print("XAI KEY:", xai_api_key)

HTTP_API_KEY = os.getenv('HTTP_API_KEY')
HTTP_API_BASE = os.getenv('HTTP_API_BASE')


PORT = int(os.environ.get("PORT", 5000))

flask_app = Flask(__name__)
flask_app.secret_key = 'your-secret-key'
flask_app.config['SESSION_TYPE'] = 'filesystem'
Session(flask_app)
flask_app.secret_key = os.getenv("flask_app.secret_key")

#@flask_app.route('/', methods=['GET', 'POST'])
#def home():
#    user_input = ""
#    response = ""

#    if request.method == "POST":
#        user_input = request.form.get("user_input")
#        response = pidgin_prompt_chatbot(user_input)

#    return render_template("index.html", user_input=user_input, response=response)
    #return "This na the home page of the Naija Guy chatbot."

@flask_app.route('/', methods=['GET', 'POST'])
def home():
    if 'history' not in session:
        session['history'] = []

    if request.method == "POST":
        user_input = request.form.get("user_input")
        response = pidgin_prompt_chatbot(user_input)

        # Save both user message and bot reply to history
        session['history'].append(('user', user_input))
        session['history'].append(('9jaguy', response))
        session.modified = True  # Mark session as changed

    return render_template("index.html", history=session['history'])

@flask_app.route('/clear')
def clear_chat():
    session.clear()
    return redirect('/')


# Text Scraping and Dictionary Logic
USER_AGENT = "naijaguy-app/1.0"
os.environ["USER_AGENT"] = USER_AGENT
    

file_path = "New Text Document.txt"
lists = []
with open(file_path, 'r') as file:
    for line in file:
        url = line.strip()
        lists.append(url)

# Function to get plain text from a URL
"""def fetch_and_parse(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}" """

def clean_text(text):
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove URLs (just in case)
    text = re.sub(r'http\S+', '', text)
    # Remove unwanted characters, keeping basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\']+', '', text)
    return text
# To remove special characters

def fetch_and_parse(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        raw_text = soup.get_text()
        cleaned_text = clean_text(raw_text)  # Clean the extracted text
        return cleaned_text
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"

# Loop through URLs and extract text
for url in lists:
    text = fetch_and_parse(url)
    #print(f"Content from {url}:\n{text[:100]}...\n")  # Print first 500 characters
#Creating pidgin dictionary
pidgin_dict = {}

for url in lists:
    text = fetch_and_parse(url)
    lines = text.split("\n")
    for line in lines:
        if "-" in line:
            parts = line.split("-", 1)
            pidgin = parts[0].strip()
            english = parts[1].strip()
            pidgin_dict[pidgin] = english
import json
with open("pidgin_dict.json", "w", encoding="utf-8") as f:
    json.dump(pidgin_dict, f, ensure_ascii=False, indent=2)
# Loading the JSON directory
with open("pidgin_dict.json", "r", encoding="utf-8") as f:
    pidgin_dict = json.load(f)

english_to_pidgin_dict = {v.lower(): k.lower() for k, v in pidgin_dict.items()}
# Define translation function

"""def pidgin_to_english(text, dictionary):
    for pidgin, english in dictionary.items():
        text = text.replace(pidgin.lower(), english.lower())
    return text

def english_to_pidgin(text, dictionary):
    for pidgin, english in dictionary.items():
        text = text.replace(english.lower(), pidgin.lower())
    return text"""


def pidgin_to_english(text, dictionary):
    words = text.lower().split()
    translated = [dictionary.get(word, word) for word in words]
    return ' '.join(translated)

def english_to_pidgin(text, dictionary):
    words = text.lower().split()
    translated = [dictionary.get(word, word) for word in words]
    return ' '.join(translated)

from langchain_xai import ChatXAI

llm = ChatXAI(
    model="grok-3-mini-fast",
    xai_api_key=xai_api_key,  # Replace with your real API key
    temperature=0
)

def translate_prompt(input_text, direction="pidgin-to-english"):
    if direction == "pidgin-to-english":
        return pidgin_to_english(input_text, pidgin_dict)
    else:
        return english_to_pidgin(input_text, english_to_pidgin_dict)
        
"""CONDENCE_QUESTION_PROMPT = PromptTemplate.from_template('''Given the following conversation and a follow up question, rephrase the following question to be a standalone question.

Chat History:
{chat_history}
Follow up Input:
{question}
Standalone questions: ''')

qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm#,
            #retriever=db.as_retriever()
        )"""

"""esponse = llm.invoke(prompt)
    #print("🗣️ Bot (Pidgin):", pidgin_response)
    return pidgin_response"""

"""def pidgin_prompt_chatbot(user_input):
    prompt = f"Answer this question in Nigerian Pidgin English: {user_input}"
    response = llm.invoke(prompt)
    
    # ✅ Only return the actual message text (not internal reasoning)
    if hasattr(response, "content"):
        print("🗣️ Bot (Pidgin):", response.content)
        return response.content
    else:
        print("❌ Error: Unexpected response format")
        return "" """


'''def pidgin_prompt_chatbot(user_input):
    today = datetime.now().strftime("%A, %d %B %Y")
    prompt = f"""
You be Nigerian person wey sabi correct Pidgin English well-well.
When person ask you question, just reply like normal street Pidgin guy — no dey explain like robot or AI.

Talk like person wey dey gist or respond for WhatsApp or junction gist.

User ask you:
"{user_input}"

Reply with short, real, funny or serious answer for only Naija Pidgin. No add AI or robot yarns.
"""
    response = llm.invoke(prompt)
    
    if hasattr(response, "content"):
        output = response.content.strip()
        print("🗣️ Bot (Pidgin):", output)
        return output''' 



from datetime import datetime

'''def pidgin_prompt_chatbot(user_input):
    today = datetime.now().strftime("%A, %d %B %Y")  # e.g. Thursday, 03 July 2025

    prompt = f"""
You be Nigerian person wey sabi correct Pidgin English well-well.
Today na {today}.

When person ask you question, just reply like normal street Pidgin guy — no dey explain like robot or AI.
Talk like person wey dey gist or respond for WhatsApp or junction gist.

User ask you:
"{user_input}"

Reply with short, real answer for only Naija Pidgin. No add AI or robot yarns.
"""
    response = llm.invoke(prompt)
    
    if hasattr(response, "content"):
        output = response.content.strip()
        print("🗣️ Bot (Pidgin):", output)
        return output'''
import feedparser

def fetch_top_headlines():
    # List of RSS feed URLs
    rss_urls = [
        "https://www.premiumtimesng.com/feed",
        "https://dailypost.ng/feed",
        "https://www.legit.ng/rss/all.rss",
        "https://guardian.ng/feed/",
        "https://nairametrics.com/feed",
        "https://rss.app/feeds/9ca4YFZ8uMFPJY8N.xml",
        "https://rss.cnn.com/rss/cnn_world.rss",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://feeds.reuters.com/reuters/topNews",
        "http://feeds.reuters.com/Reuters/worldNews",
        "https://www.theguardian.com/world/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.npr.org/1004/rss.xml",
        "https://rss.dw.com/rdf/rss-en-all",
        "https://www.goal.com/en-in/news/rss.xml",
        "https://rss.app/feeds/eurosport-uk-rss-feed.xml",
        "https://www.supersport.com/rss/video"
    ]
    
    headlines = []
    
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:  # Limit to top 5 headlines
            headlines.append(f"• {entry.title} ({entry.link})")
    
    return "\n".join(headlines) if headlines else "No fresh headlines available."

telegram_app = ApplicationBuilder().token(HTTP_API_KEY).build()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("How you dey? I be your Naija Guy chatbot")


async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action = "typing")
    time.sleep(1.0)

    reply_text = pidgin_prompt_chatbot(user_message)

    await update.message.reply_text(reply_text)

#def run_telegram_bot():
    #asyncio.set_event_loop(asyncio.new_event_loop())
    #loop = asyncio.get_event_loop()
    
    #telegram_app = (
     #   ApplicationBuilder()
     #    .token(HTTP_API_KEY)
     #   .build()
    #)

    #telegram_app.add_handler(CommandHandler("start", start))
    #telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

    #loop.run_until_complete(telegram_app.run_polling())
@flask_app.route('/webhook', methods=['POST'])
def telegram_webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), telegram_app.bot)
        telegram_app.update_queue.put_nowait(update)
        return "OK"
    
def pidgin_news_summary():
    headlines = fetch_top_headlines()
    
    if not headlines or "Error" in headlines[0]:
        return "I no fit fetch the latest news now. Try again later or check BBC Pidgin."

    combined = " • ".join(headlines)
    prompt = f"""
These na the latest headlines: {combined}

Abeg summarize the main gist for me in Naija Pidgin English.
Talk like person wey dey give street gist. No talk like AI.
"""
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()
    else:
        return "I get wahala to summarize the news now, try again later."

def pidgin_prompt_chatbot(user_input):
    if "news" in user_input.lower():
        return pidgin_news_summary()  #Real-time news in Pidgin
    
    # Else do normal Pidgin chat
    today = datetime.now().strftime("%A, %d %B %Y")
    prompt = f"""
You be Nigerian person wey sabi correct Pidgin English well-well.
Today na {today}.

When person ask you question, just reply like normal street Pidgin guy — no dey explain like robot or AI.
Talk like person wey dey gist or respond for WhatsApp or junction gist.

If person ask wetin be you name, reply I be your Naija Guy.

If asked who create you reply na my guy Ezichi Bliss Abel create me.

You can also have a normal conversation with the user, but always reply in Pidgin.

If possible, scrap the web for any relevant information to answer the user's question, but always respond in pidgin.

If user want a roast, give the user a proper fun roast
User ask you:
'{user_input}'

Reply with short, real answer for only Naija Pidgin.
"""
    response = llm.invoke(prompt)
    
    if hasattr(response, "content"):
        output = response.content.strip()
        print("🗣️ Bot (Pidgin):", output)
        return output


#if __name__ == "__main__":
#    app = ApplicationBuilder().token(HTTP_API_KEY).build()
#    app.add_handler(CommandHandler("start", start))
#    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

    # Start Telegram in background
    import threading
    #threading.Thread(target=telegram_app.run_polling, daemon=True).start()

    #def run_telegram_bot():
    #    telegram_app.run_polling()

#if __name__ == "__main__":
#    bot_thread = threading.Thread(target=run_telegram_bot)
#    bot_thread.start()

#if __name__ == "__main__":
    # Start Telegram Bot in background
    #telegram_thread = threading.Thread(target=run_telegram_bot)
    #telegram_thread.start()

if __name__ == "__main__":
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))


    # Start Flask app for Render
    flask_app.run(host="0.0.0.0", port=PORT)

# Run the bot
#app.run_polling()
