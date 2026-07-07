import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import json
import urllib.parse
import requests
import numpy as np

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Food Supply Intel", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def setup_nltk():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nltk()

try:
    api_key = st.secrets["GROQ_API_KEY"]
    groq_client = Groq(api_key=api_key)
except Exception:
    groq_client = None

# --- TELEGRAM BOT SETUP ---
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
        return True
    except:
        return False

# --- LIVE FOREX ---
@st.cache_data(ttl=3600)
def get_myr_rate():
    try:
        return yf.Ticker("MYR=X").history(period="1d")['Close'].iloc[-1]
    except:
        return 4.70
USD_TO_MYR = get_myr_rate()

# --- THE COMMODITY DATABASE (Now with GPS Coordinates for Weather) ---
BASE_COMMODITIES = {
    "🌾 Grains & Cereals": {
        "Wheat": {"ticker": "ZW=F", "search": "wheat", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 27.2155, "lat": 38.5, "lon": -98.0, "region": "Kansas, USA"},
        "Corn": {"ticker": "ZC=F", "search": "corn", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 25.4012, "lat": 42.0, "lon": -93.0, "region": "Iowa, USA"},
        "Soybeans": {"ticker": "ZS=F", "search": "soybeans", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 27.2155, "lat": -12.9, "lon": -56.0, "region": "Mato Grosso, Brazil"},
        "Rough Rice": {"ticker": "ZR=F", "search": "rice", "multiplier": 0.01, "unit": "Hundredweight", "kg_per_unit": 45.3592, "lat": 30.9, "lon": 75.8, "region": "Punjab, India"},
    },
    "☕ Softs & Cash Crops": {
        "Cocoa": {"ticker": "CC=F", "search": "cocoa", "multiplier": 1.0, "unit": "Metric Ton", "kg_per_unit": 1000.0, "lat": 7.5, "lon": -5.5, "region": "Ivory Coast, Africa"},
        "Coffee": {"ticker": "KC=F", "search": "coffee", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592, "lat": -19.9, "lon": -43.9, "region": "Minas Gerais, Brazil"},
        "Sugar": {"ticker": "SB=F", "search": "sugar", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592, "lat": -22.9, "lon": -47.0, "region": "São Paulo, Brazil"},
    }
}

if 'custom_foods' not in st.session_state:
    st.session_state.custom_foods = {}
if 'deleted_foods' not in st.session_state:
    st.session_state.deleted_foods = []

COMMODITIES = {}
for cat, foods in BASE_COMMODITIES.items():
    for comm, data in foods.items():
        if (cat, comm) not in st.session_state.deleted_foods:
            if cat not in COMMODITIES: COMMODITIES[cat] = {}
            COMMODITIES[cat][comm] = data

for cat, foods in st.session_state.custom_foods.items():
    for comm, data in foods.items():
        if cat not in COMMODITIES: COMMODITIES[cat] = {}
        COMMODITIES[cat][comm] = data

# --- INTELLIGENCE FUNCTIONS ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_financial_data(ticker, multiplier):
    if ticker == "NONE": return 0.0, 0.0, 0.0, 50.0, pd.DataFrame() # OSINT Only Mode
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="6mo")
        if hist.empty or len(hist) < 15: return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()
        
        hist['50_MA'] = hist['Close'].rolling(window=50).mean() 
        hist['RSI'] = calculate_rsi(hist)
        
        raw_today = hist['Close'].iloc[-1]
        raw_yesterday = hist['Close'].iloc[-2]
        today_usd = raw_today * multiplier
        percent_change = ((raw_today - raw_yesterday) / raw_yesterday) * 100
        trend_50_ma = hist['50_MA'].iloc[-1] * multiplier
        rsi_today = hist['RSI'].iloc[-1]
        
        return today_usd, percent_change, trend_50_ma, rsi_today, hist.tail(90)
    except:
        return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()

def get_weather_data(lat, lon):
    if not lat or not lon: return None
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&timezone=auto"
        res = requests.get(url).json()
        temp = res['current_weather']['temperature']
        rain_7d = sum(res['daily']['precipitation_sum'][:7])
        return {"temp": temp, "rain": rain_7d}
    except:
        return None

def get_news_data(search_term):
    try:
        query = f'"{search_term}" (shortage OR supply OR price OR export)'
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = []
        total_sentiment = 0
        for entry in feed.entries[:10]:
            sentiment = sia.polarity_scores(entry.title)['compound']
            total_sentiment += sentiment
            articles.append({"Headline": entry.title, "Threat Score": sentiment})
        avg_sentiment = total_sentiment / len(articles) if articles else 0
        return avg_sentiment, articles
    except:
        return 0.0, []

def get_ai_brief(commodity, articles, price_change):
    if not groq_client: return "⚠️ AI Offline."
    if not articles: return "⚠️ No recent news."
    headlines = [art['Headline'] for art in articles[:5]]
    prompt = f"Act as a CIA analyst for food security. Target: {commodity}. Price change: {price_change:.2f}%. Headlines: {headlines}. Write a 2-sentence tactical 'BLUF' summarizing the threat."
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        return chat.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def ai_auto_discover(food_name, existing_categories):
    if not groq_client: return None, "AI is offline."
    prompt = f"""
    Find global data for: "{food_name}". Return ONLY valid JSON.
    {{
        "category": "MUST be one of: {existing_categories}. Or invent a new one with emoji.",
        "ticker": "Yahoo Finance futures ticker (e.g. CPO=F). If it DOES NOT TRADE on futures (like Matcha or Salt), return 'NONE'",
        "search": "1 word search term for news",
        "unit": "Trading unit (e.g. Metric Ton). If NONE, put 'Kg'",
        "kg_per_unit": Float kg in unit. If NONE, put 1.0,
        "is_cents": true if US Cents, false if USD. If NONE, put false
    }}
    """
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        clean_json = chat.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json), "Success"
    except:
        return None, "Failed"

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/US_Department_of_Agriculture_seal.svg/1024px-US_Department_of_Agriculture_seal.svg.png", width=100)
st.sidebar.title("Command Center")

if not COMMODITIES:
    st.stop()

selected_category = st.sidebar.selectbox("1. Select Sector", list(COMMODITIES.keys()))
selected_commodity = st.sidebar.selectbox("2. Select Target", list(COMMODITIES[selected_category].keys()))
details = COMMODITIES[selected_category][selected_commodity]
is_osint_only = details.get("ticker") == "NONE"

st.sidebar.divider()

# --- MACRO LOGISTICS TRACKER ---
st.sidebar.markdown("### 🚢 Macro Inputs (Logistics)")
with st.sidebar.expander("View Global Fertilizer & Shipping"):
    try:
        fert = yf.Ticker("NTR").history(period="2d")['Close']
        ship = yf.Ticker("BDRY").history(period="2d")['Close']
        fert_pct = ((fert.iloc[-1] - fert.iloc[-2])/fert.iloc[-2])*100
        ship_pct = ((ship.iloc[-1] - ship.iloc[-2])/ship.iloc[-2])*100
        st.metric("Global Fertilizer (NTR)", f"${fert.iloc[-1]:.2f}", f"{fert_pct:.2f}%")
        st.metric("Dry Bulk Shipping (BDRY)", f"${ship.iloc[-1]:.2f}", f"{ship_pct:.2f}%")
        st.caption("If these rise, food prices rise 6 months later.")
    except:
        st.caption("Macro data offline.")

st.sidebar.divider()

# --- AI AUTO-DISCOVER ---
st.sidebar.markdown("### 🤖 AI Auto-Discover")
new_food_name = st.sidebar.text_input("Enter Target (e.g., Matcha, Palm Oil)")
if st.sidebar.button("Auto-Detect & Deploy"):
    if new_food_name:
        with st.sidebar.status("AI is hunting..."):
            ai_data, status = ai_auto_discover(new_food_name, list(BASE_COMMODITIES.keys()))
            if ai_data:
                cat = ai_data["category"]
                if cat not in st.session_state.custom_foods: st.session_state.custom_foods[cat] = {}
                st.session_state.custom_foods[cat][new_food_name.title()] = {
                    "ticker": ai_data["ticker"], "search": ai_data["search"],
                    "multiplier": 0.01 if ai_data["is_cents"] else 1.0,
                    "unit": ai_data["unit"], "kg_per_unit": ai_data["kg_per_unit"],
                    "lat": None, "lon": None # Custom foods don't get weather yet
                }
                st.rerun()

# --- MAIN DASHBOARD UI ---
st.title("🌍 Global Food Supply Threat Matrix")

# Fetch Data
price_usd, price_change, trend_ma, rsi, price_history = get_financial_data(details["ticker"], details.get("multiplier", 1.0))
avg_sentiment, news_articles = get_news_data(details["search"])
weather = get_weather_data(details.get("lat"), details.get("lon"))

# Header & Delete
col_head1, col_head2 = st.columns([5, 1])
with col_head1: st.header(f"🎯 Target Acquired: {selected_commodity}")
with col_head2:
    st.write("")
    if st.button("🗑️ Remove Target", type="tertiary"):
        if selected_category in st.session_state.custom_foods and selected_commodity in st.session_state.custom_foods[selected_category]:
            del st.session_state.custom_foods[selected_category][selected_commodity]
        else:
            st.session_state.deleted_foods.append((selected_category, selected_commodity))
        st.rerun()

# Threat Logic
threat_level = "🔴 DEFCON 3 (HIGH RISK)" if price_change > 2.0 or avg_sentiment < -0.25 else "🟢 NORMAL"

# Alert Button
if st.button("🚨 Push Alert to Telegram"):
    msg = f"🚨 *FOOD THREAT ALERT*\n*Target:* {selected_commodity}\n*Status:* {threat_level}\n*Price Change:* {price_change:.2f}%\n*News Sentiment:* {avg_sentiment:.2f}"
    if send_telegram_alert(msg): st.success("Alert sent to Secure Channel!")
    else: st.error("Telegram not configured. See instructions.")

st.divider()

if is_osint_only:
    st.warning("🕵️ **OSINT-ONLY MODE:** This commodity does not trade on global futures markets (No Ticker). Financial charts are disabled. Tracking via Global News Sentiment only.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
if not is_osint_only:
    with col1: st.metric(label=f"Market Price ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%")
    with col2: 
        rsi_label = "🔥 OVERBOUGHT" if rsi > 70 else "❄️ OVERSOLD" if rsi < 30 else "⚖️ NEUTRAL"
        st.metric(label="Technical RSI (14-Day)", value=f"{rsi:.1f}", delta=rsi_label, delta_color="off")
with col3: st.metric(label="OSINT Sentiment", value=f"{avg_sentiment:.2f}", delta="Negative = Threat", delta_color="inverse")
with col4: st.metric(label="System Status", value=threat_level)

# Weather Intel
if weather:
    st.info(f"🌦️ **CLIMATE INTEL ({details['region']}):** Current Temp: **{weather['temp']}°C** | 7-Day Rainfall: **{weather['rain']}mm**")

# AI Brief
st.markdown("### 🤖 AI Analyst Brief (BLUF)")
with st.spinner('Decrypting intel...'):
    st.info(get_ai_brief(selected_commodity, news_articles, price_change))

# Charts & News
col_chart, col_news = st.columns([2, 1])

with col_chart:
    if not is_osint_only and not price_history.empty:
        st.markdown("### 📈 90-Day Technical Analysis (FININT)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=price_history.index, open=price_history['Open']*details["multiplier"], high=price_history['High']*details["multiplier"], low=price_history['Low']*details["multiplier"], close=price_history['Close']*details["multiplier"], name="Price"))
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history['50_MA']*details["multiplier"], line=dict(color='orange', width=2), name="50-Day MA"))
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    elif not is_osint_only:
        st.warning("Financial chart offline due to missing market data.")

with col_news:
    st.markdown("### 📰 Live OSINT Chatter")
    if news_articles:
        df = pd.DataFrame(news_articles)
        st.dataframe(df.style.map(lambda val: f'color: {"#ff4b4b" if val < 0 else "#00cc96"}', subset=['Threat Score']), hide_index=True)
    else:
        st.info("No immediate threats detected in global news chatter.")
