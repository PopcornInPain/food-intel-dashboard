import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import json

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

# --- LIVE FOREX (USD to MYR) ---
@st.cache_data(ttl=3600)
def get_myr_rate():
    try:
        myr_data = yf.Ticker("MYR=X").history(period="1d")
        return myr_data['Close'].iloc[-1]
    except:
        return 4.70

USD_TO_MYR = get_myr_rate()

# --- THE MASSIVE COMMODITY DATABASE ---
BASE_COMMODITIES = {
    "🌾 Grains & Cereals": {
        "Wheat": {"ticker": "ZW=F", "search": "wheat", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 27.2155},
        "Corn": {"ticker": "ZC=F", "search": "corn", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 25.4012},
        "Soybeans": {"ticker": "ZS=F", "search": "soybeans", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 27.2155},
        "Rough Rice": {"ticker": "ZR=F", "search": "rice", "multiplier": 0.01, "unit": "Hundredweight", "kg_per_unit": 45.3592},
        "Oats": {"ticker": "ZO=F", "search": "oats", "multiplier": 0.01, "unit": "Bushel", "kg_per_unit": 14.515},
    },
    "☕ Softs & Cash Crops": {
        "Cocoa": {"ticker": "CC=F", "search": "cocoa", "multiplier": 1.0, "unit": "Metric Ton", "kg_per_unit": 1000.0},
        "Coffee": {"ticker": "KC=F", "search": "coffee", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
        "Sugar": {"ticker": "SB=F", "search": "sugar", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
        "Orange Juice": {"ticker": "OJ=F", "search": "orange juice", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
    },
    "🥩 Meats & Livestock": {
        "Live Cattle (Beef)": {"ticker": "LE=F", "search": "cattle beef", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
        "Lean Hogs (Pork)": {"ticker": "HE=F", "search": "pork hogs", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
    },
    "🥛 Dairy & Oils": {
        "Class III Milk": {"ticker": "DC=F", "search": "milk dairy", "multiplier": 1.0, "unit": "Hundredweight", "kg_per_unit": 45.3592},
        "Soybean Oil": {"ticker": "ZL=F", "search": "soybean oil", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
    }
}

# --- SESSION STATE FOR CUSTOM FOODS ---
if 'custom_foods' not in st.session_state:
    st.session_state.custom_foods = {}

COMMODITIES = BASE_COMMODITIES.copy()
if st.session_state.custom_foods:
    COMMODITIES["🤖 AI Discovered Targets"] = st.session_state.custom_foods

# --- INTELLIGENCE FUNCTIONS ---
def get_financial_data(ticker, multiplier):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="3mo")
        if hist.empty or len(hist) < 2:
            return 0.0, 0.0, 0.0, pd.DataFrame()
        hist['50_MA'] = hist['Close'].rolling(window=14).mean() 
        raw_today = hist['Close'].iloc[-1]
        raw_yesterday = hist['Close'].iloc[-2]
        today_usd = raw_today * multiplier
        yesterday_usd = raw_yesterday * multiplier
        percent_change = ((today_usd - yesterday_usd) / yesterday_usd) * 100
        trend_50_ma = hist['50_MA'].iloc[-1] * multiplier
        return today_usd, percent_change, trend_50_ma, hist
    except Exception:
        return 0.0, 0.0, 0.0, pd.DataFrame()

def get_news_data(search_term):
    try:
        url = f"https://news.google.com/rss/search?q={search_term}+export+ban+OR+{search_term}+drought+OR+{search_term}+shortage+OR+{search_term}+disease&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        articles = []
        total_sentiment = 0
        for entry in feed.entries[:10]:
            sentiment = sia.polarity_scores(entry.title)['compound']
            total_sentiment += sentiment
            articles.append({"Headline": entry.title, "Threat Score": sentiment})
        avg_sentiment = total_sentiment / 10 if feed.entries else 0
        return avg_sentiment, articles
    except Exception:
        return 0.0, []

def get_ai_brief(commodity, articles, price_change):
    if not groq_client:
        return "⚠️ AI Offline."
    if not articles:
        return "⚠️ No recent news."
    headlines = [art['Headline'] for art in articles[:5]]
    prompt = f"Act as a CIA intelligence analyst for food security. Target: {commodity}. Price change today: {price_change:.2f}%. Read these headlines: {headlines}. Write a 2-sentence tactical 'BLUF' summarizing the supply chain threat. Mention if the news justifies the price movement."
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        return chat.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def ai_auto_discover(food_name):
    if not groq_client:
        return None, "AI is offline. Cannot auto-discover."
    prompt = f"""
    Find the global futures market data for the agricultural commodity: "{food_name}".
    Return ONLY a valid JSON object. No markdown, no extra text.
    Format:
    {{
        "ticker": "The Yahoo Finance futures ticker (e.g. CPO=F for Palm Oil, ZR=F for Rice). If it doesn't trade on futures, return 'NONE'",
        "search": "A short 1-2 word search term for news (e.g. 'palm oil')",
        "unit": "The standard trading unit (e.g. Metric Ton, Pound, Bushel)",
        "kg_per_unit": The exact float number of kilograms in that unit (e.g. 1000.0 for Metric Ton),
        "is_cents": true if the price is quoted in US Cents, false if quoted in US Dollars
    }}
    """
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        raw_response = chat.choices[0].message.content
        # Clean up the response in case the AI added markdown
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data, "Success"
    except Exception as e:
        return None, f"Failed to parse AI response: {e}"

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/US_Department_of_Agriculture_seal.svg/1024px-US_Department_of_Agriculture_seal.svg.png", width=100)
st.sidebar.title("Command Center")

selected_category = st.sidebar.selectbox("1. Select Sector", list(COMMODITIES.keys()))
selected_commodity = st.sidebar.selectbox("2. Select Target", list(COMMODITIES[selected_category].keys()))
details = COMMODITIES[selected_category][selected_commodity]

st.sidebar.divider()

# --- NEW FEATURE: AI AUTO-DISCOVER ---
st.sidebar.markdown("### 🤖 AI Auto-Discover")
st.sidebar.caption("Type a food. The AI will find the financial data and build a dashboard for it automatically.")
new_food_name = st.sidebar.text_input("Enter Target (e.g., Palm Oil, Canola)")

if st.sidebar.button("Auto-Detect & Deploy"):
    if new_food_name:
        with st.sidebar.status("AI is hunting for financial data..."):
            ai_data, status = ai_auto_discover(new_food_name)
            
            if ai_data and ai_data.get("ticker") != "NONE":
                multiplier = 0.01 if ai_data["is_cents"] else 1.0
                st.session_state.custom_foods[new_food_name.title()] = {
                    "ticker": ai_data["ticker"],
                    "search": ai_data["search"],
                    "multiplier": multiplier,
                    "unit": ai_data["unit"],
                    "kg_per_unit": ai_data["kg_per_unit"]
                }
                st.rerun()
            else:
                st.sidebar.error("Could not find a global futures market for this item.")

# --- MAIN DASHBOARD UI ---
st.title("🌍 Global Food Supply Threat Matrix")
st.markdown(f"**Live Forex Rate:** 1 USD = {USD_TO_MYR:.2f} MYR")

# Fetch Data
price_usd, price_change, trend_ma, price_history = get_financial_data(details["ticker"], details["multiplier"])
avg_sentiment, news_articles = get_news_data(details["search"])
price_myr = price_usd * USD_TO_MYR

# Standardized KG Math
price_per_kg_usd = price_usd / details["kg_per_unit"] if details["kg_per_unit"] > 0 else 0
price_per_kg_myr = price_myr / details["kg_per_unit"] if details["kg_per_unit"] > 0 else 0

st.header(f"🎯 Target Acquired: {selected_commodity}")

# Top Row Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label=f"Market Price ({details['unit']})", value=f"${price_usd:.2f}", delta=f"{price_change:.2f}%")
    st.caption(f"Standardized: **${price_per_kg_usd:.4f} / kg**")
with col2:
    st.metric(label="Price in MYR", value=f"RM {price_myr:.2f}")
    st.caption(f"Standardized: **RM {price_per_kg_myr:.4f} / kg**")
with col3:
    st.metric(label="OSINT Sentiment", value=f"{avg_sentiment:.2f}", delta="Negative = Threat", delta_color="inverse")
with col4:
    threat_level = "🔴 HIGH RISK" if price_change > 2.0 or avg_sentiment < -0.25 else "🟢 NORMAL"
    st.metric(label="System Status", value=threat_level)
    
# AI Analyst Brief
st.markdown("### 🤖 AI Analyst Brief (BLUF)")
with st.spinner('Decrypting intel...'):
    ai_brief = get_ai_brief(selected_commodity, news_articles, price_change)
    st.info(ai_brief)

# Charts and News Layout
col_chart, col_news = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 90-Day Price Action & Trend (FININT)")
    if not price_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=price_history.index,
                    open=price_history['Open'] * details["multiplier"], 
                    high=price_history['High'] * details["multiplier"],
                    low=price_history['Low'] * details["multiplier"], 
                    close=price_history['Close'] * details["multiplier"],
                    name="Price"))
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history['50_MA'] * details["multiplier"], 
                                 line=dict(color='orange', width=2), name="14-Day Trend"))
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Financial data temporarily unavailable or market is closed.")

with col_news:
    st.markdown("### 📰 Live OSINT Chatter")
    if news_articles:
        df = pd.DataFrame(news_articles)
        def color_threat(val):
            return f'color: {"#ff4b4b" if val < 0 else "#00cc96"}'
        st.dataframe(df.style.map(color_threat, subset=['Threat Score']), hide_index=True)
