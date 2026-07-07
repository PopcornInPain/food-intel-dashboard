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

# --- CACHED MACRO & FOREX DATA ---
@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        myr = yf.Ticker("MYR=X").history(period="1d")['Close'].iloc[-1]
        fert = yf.Ticker("NTR").history(period="2d")['Close']
        ship = yf.Ticker("BDRY").history(period="2d")['Close']
        fert_pct = ((fert.iloc[-1] - fert.iloc[-2])/fert.iloc[-2])*100
        ship_pct = ((ship.iloc[-1] - ship.iloc[-2])/ship.iloc[-2])*100
        return myr, fert.iloc[-1], fert_pct, ship.iloc[-1], ship_pct
    except:
        return 4.70, 0.0, 0.0, 0.0, 0.0

USD_TO_MYR, FERT_PRICE, FERT_PCT, SHIP_PRICE, SHIP_PCT = get_macro_data()

# --- THE COMMODITY DATABASE ---
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

if 'custom_foods' not in st.session_state: st.session_state.custom_foods = {}
if 'deleted_foods' not in st.session_state: st.session_state.deleted_foods = []

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
    if ticker == "NONE": return 0.0, 0.0, 0.0, 50.0, pd.DataFrame() 
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="6mo")
        if hist.empty or len(hist) < 15: return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()
        hist['50_MA'] = hist['Close'].rolling(window=50).mean() 
        hist['RSI'] = calculate_rsi(hist)
        raw_today = hist['Close'].iloc[-1]
        raw_yesterday = hist['Close'].iloc[-2]
        return raw_today * multiplier, ((raw_today - raw_yesterday) / raw_yesterday) * 100, hist['50_MA'].iloc[-1] * multiplier, hist['RSI'].iloc[-1], hist.tail(90)
    except:
        return 0.0, 0.0, 0.0, 50.0, pd.DataFrame()

def get_weather_data(lat, lon):
    if not lat or not lon: return None
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&timezone=auto"
        res = requests.get(url).json()
        return {"temp": res['current_weather']['temperature'], "rain": sum(res['daily']['precipitation_sum'][:7])}
    except:
        return None

def get_news_data(search_term):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(f'\"{search_term}\" (shortage OR supply OR price OR export)')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = [{"Headline": e.title, "Threat Score": sia.polarity_scores(e.title)['compound']} for e in feed.entries[:10]]
        return (sum(a['Threat Score'] for a in articles) / len(articles) if articles else 0.0), articles
    except:
        return 0.0, []

def calculate_master_threat(price_pct, sentiment, rsi, fert_pct, ship_pct, weather, is_osint_only):
    score = 0
    if is_osint_only:
        if sentiment < -0.2: score += 50
        if sentiment < -0.5: score += 50
    else:
        if price_pct > 1.5: score += 15
        if price_pct > 3.0: score += 15
        if sentiment < -0.15: score += 15
        if sentiment < -0.40: score += 15
        if rsi > 70: score += 10 
        if fert_pct > 1.0: score += 7.5 
        if ship_pct > 1.0: score += 7.5 
        if weather:
            if weather['temp'] > 32.0: score += 7.5 
            if weather['rain'] < 5.0: score += 7.5 

    if score >= 70: return score, "🔴 DEFCON 1 (CRITICAL)"
    if score >= 40: return score, "🟠 DEFCON 2 (ELEVATED)"
    return score, "🟢 DEFCON 3 (NORMAL)"

def get_ai_brief(commodity, articles, price_change, rsi, fert_pct, weather, threat_score):
    if not groq_client: return "⚠️ AI Offline."
    headlines = [art['Headline'] for art in articles[:5]] if articles else ["No news."]
    weather_txt = f"Temp: {weather['temp']}C, Rain: {weather['rain']}mm" if weather else "N/A"
    
    prompt = f"""
    Act as a CIA analyst for food security. Target: {commodity}. 
    Master Threat Score: {threat_score}/100.
    DATA INPUTS: Price Change: {price_change:.2f}%. RSI: {rsi:.1f}. Global Fertilizer Change: {fert_pct:.2f}%. 
    Weather at Chokepoint: {weather_txt}. 
    Headlines: {headlines}. 
    Write a 2-sentence tactical 'BLUF' summarizing the threat. Connect the macro data (fertilizer/weather/RSI) to the news if relevant.
    """
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

if not COMMODITIES: st.stop()

selected_category = st.sidebar.selectbox("1. Select Sector", list(COMMODITIES.keys()))
selected_commodity = st.sidebar.selectbox("2. Select Target", list(COMMODITIES[selected_category].keys()))
details = COMMODITIES[selected_category][selected_commodity]
is_osint_only = details.get("ticker") == "NONE"

st.sidebar.divider()
st.sidebar.markdown("### 🚢 Macro Inputs (Logistics)")
st.sidebar.metric("Global Fertilizer (NTR)", f"${FERT_PRICE:.2f}", f"{FERT_PCT:.2f}%")
st.sidebar.metric("Dry Bulk Shipping (BDRY)", f"${SHIP_PRICE:.2f}", f"{SHIP_PCT:.2f}%")

st.sidebar.divider()

# --- AI AUTO-DISCOVER UI ---
st.sidebar.markdown("### 🤖 AI Auto-Discover")
st.sidebar.caption("Type any food. The AI will find financial data, or activate OSINT-Only mode if it's not traded.")
new_food_name = st.sidebar.text_input("Enter Target (e.g., Matcha, Palm Oil)")

if st.sidebar.button("Auto-Detect & Deploy"):
    if new_food_name:
        with st.sidebar.status("AI is hunting..."):
            current_cats = list(BASE_COMMODITIES.keys())
            ai_data, status = ai_auto_discover(new_food_name, current_cats)
            
            if ai_data:
                multiplier = 0.01 if ai_data.get("is_cents") else 1.0
                cat = ai_data["category"]
                
                if cat not in st.session_state.custom_foods:
                    st.session_state.custom_foods[cat] = {}
                    
                st.session_state.custom_foods[cat][new_food_name.title()] = {
                    "ticker": ai_data["ticker"],
                    "search": ai_data["search"],
                    "multiplier": multiplier,
                    "unit": ai_data["unit"],
                    "kg_per_unit": ai_data["kg_per_unit"],
                    "lat": None, "lon": None 
                }
                st.rerun()
            else:
                st.sidebar.error("AI Discovery Failed.")

# --- MAIN DASHBOARD UI ---
st.title("🌍 Global Food Supply Threat Matrix")

# --- FIELD MANUAL ---
with st.expander("📖 FIELD MANUAL: System Architecture & Threat Logic", expanded=False):
    st.markdown("""
    ### 🛡️ Intelligence Architecture
    This platform processes 6 separate streams of global data to calculate a **Master Threat Score (0-100)** for agricultural supply chains.

    #### 1. The Master Threat Algorithm (How the score is calculated)
    The system uses a weighted matrix to detect supply shocks before they happen:
    *   **Price Action (30%):** Sudden daily spikes in the global futures market.
    *   **OSINT Chatter (30%):** Natural Language Processing (NLP) scans global news for words like *drought, ban, shortage, strike*.
    *   **Macro Logistics (15%):** Tracks Global Fertilizer (NTR) and Dry Bulk Shipping (BDRY). *Rule of thumb: If fertilizer spikes today, food prices spike in 6 months.*
    *   **Climate Intel (15%):** Live weather APIs track temperature and rainfall at major global chokepoints (e.g., Mato Grosso for Soybeans).
    *   **Technical RSI (10%):** The Relative Strength Index detects if a price spike is caused by fundamental shortages or temporary "Panic Buying" (Overbought > 70).

    #### 2. DEFCON Status Levels
    *   🟢 **DEFCON 3 (NORMAL):** Score 0-39. Standard market conditions.
    *   🟠 **DEFCON 2 (ELEVATED):** Score 40-69. Anomalies detected in weather, news, or logistics. Monitor closely.
    *   🔴 **DEFCON 1 (CRITICAL):** Score 70-100. Multiple indicators are flashing red. Severe supply chain disruption is imminent or occurring.

    #### 3. AI Auto-Discovery & OSINT-Only Mode
    Use the **Command Center (Left Sidebar)** to track new foods. 
    *   If you search for a globally traded commodity (like *Palm Oil*), the AI will find its financial ticker and build a full dashboard. 
    *   If you search for a niche item that does not trade on the stock market (like *Matcha* or *Salt*), the AI will automatically activate **OSINT-Only Mode**, disabling financial charts and tracking the threat purely through global news sentiment.
    """)

st.divider()

# Fetch Data
price_usd, price_change, trend_ma, rsi, price_history = get_financial_data(details["ticker"], details.get("multiplier", 1.0))
avg_sentiment, news_articles = get_news_data(details["search"])
weather = get_weather_data(details.get("lat"), details.get("lon"))

# Standardized KG/L Math
price_myr = price_usd * USD_TO_MYR
kg_per_unit = details.get("kg_per_unit", 1.0)
price_per_kg_usd = price_usd / kg_per_unit if kg_per_unit > 0 else 0
price_per_kg_myr = price_myr / kg_per_unit if kg_per_unit > 0 else 0

# Determine if it's a liquid for the label
std_unit = "L" if "gallon" in details.get("unit", "").lower() or "liter" in details.get("unit", "").lower() else "kg"

# RUN MASTER THREAT ALGORITHM
threat_score, threat_level = calculate_master_threat(price_change, avg_sentiment, rsi, FERT_PCT, SHIP_PCT, weather, is_osint_only)

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

# --- DEFCON BANNER ---
if "DEFCON 1" in threat_level:
    st.error(f"🚨 **CRITICAL ALERT:** {selected_commodity} Threat Score is {threat_score}/100. Multiple macro indicators (Price, Weather, OSINT, or Logistics) are flashing red.")
elif "DEFCON 2" in threat_level:
    st.warning(f"⚠️ **ELEVATED RISK:** {selected_commodity} Threat Score is {threat_score}/100. Anomalies detected in supply chain inputs.")

if is_osint_only: st.warning("🕵️ **OSINT-ONLY MODE:** Tracking via Global News Sentiment only.")

# --- METRICS ROW (Expanded to 5 columns to fit MYR and Standardized Pricing) ---
if not is_osint_only:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: 
        st.metric(label=f"Price USD ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%")
        st.caption(f"Standardized: **${price_per_kg_usd:.4f} / {std_unit}**")
    with col2: 
        st.metric(label="Price in MYR", value=f"RM {price_myr:.2f}")
        st.caption(f"Standardized: **RM {price_per_kg_myr:.4f} / {std_unit}**")
    with col3: 
        st.metric(label="Technical RSI", value=f"{rsi:.1f}", delta="🔥 OVERBOUGHT" if rsi > 70 else "❄️ OVERSOLD" if rsi < 30 else "⚖️ NEUTRAL", delta_color="off")
    with col4: 
        st.metric(label="OSINT Sentiment", value=f"{avg_sentiment:.2f}", delta="Negative = Threat", delta_color="inverse")
    with col5: 
        st.metric(label="Master Threat", value=f"{threat_score}/100", delta=threat_level, delta_color="inverse" if "DEFCON 1" in threat_level else "off")
else:
    # OSINT Only Mode Layout
    col1, col2, col3 = st.columns(3)
    with col1: st.metric(label="OSINT Sentiment", value=f"{avg_sentiment:.2f}", delta="Negative = Threat", delta_color="inverse")
    with col2: st.metric(label="Master Threat Score", value=f"{threat_score}/100", delta=threat_level, delta_color="inverse" if "DEFCON 1" in threat_level else "off")
    with col3: st.empty()

if weather: st.info(f"🌦️ **CLIMATE INTEL ({details['region']}):** Current Temp: **{weather['temp']}°C** | 7-Day Rainfall: **{weather['rain']}mm**")

# AI Brief
st.markdown("### 🤖 AI Analyst Brief (BLUF)")
with st.spinner('Decrypting intel...'):
    st.info(get_ai_brief(selected_commodity, news_articles, price_change, rsi, FERT_PCT, weather, threat_score))

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
with col_news:
    st.markdown("### 📰 Live OSINT Chatter")
    if news_articles:
        st.dataframe(pd.DataFrame(news_articles).style.map(lambda val: f'color: {"#ff4b4b" if val < 0 else "#00cc96"}', subset=['Threat Score']), hide_index=True)
    else:
        st.info("No immediate threats detected.")
