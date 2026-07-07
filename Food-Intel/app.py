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
        "Live Cattle (Beef)": {"ticker": "LE=F", "search": "beef", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
        "Lean Hogs (Pork)": {"ticker": "HE=F", "search": "pork", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
    },
    "🥛 Dairy & Oils": {
        "Class III Milk": {"ticker": "DC=F", "search": "milk", "multiplier": 1.0, "unit": "Hundredweight", "kg_per_unit": 45.3592},
        "Soybean Oil": {"ticker": "ZL=F", "search": "soybean oil", "multiplier": 0.01, "unit": "Pound", "kg_per_unit": 0.453592},
    }
}

# --- MEMORY STATE FOR CUSTOM & DELETED FOODS ---
if 'custom_foods' not in st.session_state:
    st.session_state.custom_foods = {}
if 'deleted_foods' not in st.session_state:
    st.session_state.deleted_foods = []

# Build the live database (excluding deleted items)
COMMODITIES = {}
for cat, foods in BASE_COMMODITIES.items():
    for comm, data in foods.items():
        if (cat, comm) not in st.session_state.deleted_foods:
            if cat not in COMMODITIES:
                COMMODITIES[cat] = {}
            COMMODITIES[cat][comm] = data

for cat, foods in st.session_state.custom_foods.items():
    for comm, data in foods.items():
        if cat not in COMMODITIES:
            COMMODITIES[cat] = {}
        COMMODITIES[cat][comm] = data

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
        query = f'"{search_term}" (shortage OR supply OR price OR export)'
        safe_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={safe_query}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)
        articles = []
        total_sentiment = 0
        for entry in feed.entries[:10]:
            sentiment = sia.polarity_scores(entry.title)['compound']
            total_sentiment += sentiment
            articles.append({"Headline": entry.title, "Threat Score": sentiment})
        avg_sentiment = total_sentiment / len(articles) if articles else 0
        return avg_sentiment, articles
    except Exception:
        return 0.0, []

def get_ai_brief(commodity, articles, price_change):
    if not groq_client:
        return "⚠️ AI Offline."
    if not articles:
        return "⚠️ No recent news to analyze."
    headlines = [art['Headline'] for art in articles[:5]]
    prompt = f"Act as a CIA intelligence analyst for food security. Target: {commodity}. Price change today: {price_change:.2f}%. Read these headlines: {headlines}. Write a 2-sentence tactical 'BLUF' summarizing the supply chain threat. Mention if the news justifies the price movement."
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        return chat.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

def ai_auto_discover(food_name, existing_categories):
    if not groq_client:
        return None, "AI is offline."
    prompt = f"""
    Find the global futures market data for the agricultural commodity: "{food_name}".
    Return ONLY a valid JSON object. No markdown.
    Format:
    {{
        "category": "MUST be one of these exact strings: {existing_categories}. ONLY invent a new category with an emoji if it is IMPOSSIBLE to fit into the existing ones.",
        "ticker": "The Yahoo Finance futures ticker (e.g. CPO=F for Palm Oil). If it doesn't trade on futures, return 'NONE'",
        "search": "A short 1 word search term for news",
        "unit": "The standard trading unit (e.g. Metric Ton, Pound, Bushel)",
        "kg_per_unit": The exact float number of kilograms in that unit (e.g. 1000.0 for Metric Ton),
        "is_cents": true if the price is quoted in US Cents, false if quoted in US Dollars
    }}
    """
    try:
        chat = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
        raw_response = chat.choices[0].message.content
        clean_json = raw_response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return data, "Success"
    except Exception as e:
        return None, f"Failed to parse AI response: {e}"

# --- SIDEBAR COMMAND CENTER ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/US_Department_of_Agriculture_seal.svg/1024px-US_Department_of_Agriculture_seal.svg.png", width=100)
st.sidebar.title("Command Center")

# Safety check: If user deleted absolutely everything
if not COMMODITIES:
    st.warning("🚨 All targets have been removed from the database.")
    st.info("Use the AI Auto-Discover tool in the sidebar to add a new food commodity.")
    selected_category = None
    selected_commodity = None
else:
    selected_category = st.sidebar.selectbox("1. Select Sector", list(COMMODITIES.keys()))
    selected_commodity = st.sidebar.selectbox("2. Select Target", list(COMMODITIES[selected_category].keys()))
    details = COMMODITIES[selected_category][selected_commodity]

st.sidebar.divider()

# --- AI AUTO-DISCOVER ---
st.sidebar.markdown("### 🤖 AI Auto-Discover")
st.sidebar.caption("Type a food. The AI will find the financial data and categorize it automatically.")
new_food_name = st.sidebar.text_input("Enter Target (e.g., Palm Oil, Lumber)")

if st.sidebar.button("Auto-Detect & Deploy"):
    if new_food_name:
        with st.sidebar.status("AI is hunting for financial data..."):
            current_cats = list(BASE_COMMODITIES.keys())
            ai_data, status = ai_auto_discover(new_food_name, current_cats)
            
            if ai_data and ai_data.get("ticker") != "NONE":
                multiplier = 0.01 if ai_data["is_cents"] else 1.0
                cat = ai_data["category"]
                
                if cat not in st.session_state.custom_foods:
                    st.session_state.custom_foods[cat] = {}
                    
                st.session_state.custom_foods[cat][new_food_name.title()] = {
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

with st.expander("📖 FIELD MANUAL: How to read this intelligence dashboard", expanded=False):
    st.markdown("""
    #### 🔍 How to interpret the data:
    * **Command Center (Left):** Use the sidebar to navigate through dozens of global food sectors. You can also use the AI to discover and add new foods.
    * **Price (USD & MYR):** Live global futures prices. We also calculate the standardized price per Kilogram so you can compare different commodities.
    * **OSINT Sentiment Score:** We scrape global news for threat keywords. `-1.0` is extreme danger, `+1.0` is perfectly safe.
    * **System Status:** Triggers **🔴 HIGH RISK** if the price jumps > 2% OR news sentiment drops below -0.25.
    * **🤖 AI Analyst (BLUF):** *'Bottom Line Up Front'*. Our AI reads the latest news headlines and cross-references them with today's price action to give you a 2-sentence tactical summary.
    """)

st.divider()

# Stop rendering the rest of the page if everything was deleted
if not selected_commodity:
    st.stop()

# Fetch Data
price_usd, price_change, trend_ma, price_history = get_financial_data(details["ticker"], details["multiplier"])
avg_sentiment, news_articles = get_news_data(details["search"])
price_myr = price_usd * USD_TO_MYR

# Standardized KG Math
price_per_kg_usd = price_usd / details["kg_per_unit"] if details["kg_per_unit"] > 0 else 0
price_per_kg_myr = price_myr / details["kg_per_unit"] if details["kg_per_unit"] > 0 else 0

# --- THE HEADER & SUBTLE DELETE BUTTON ---
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.header(f"🎯 Target Acquired: {selected_commodity}")
with col_head2:
    st.write("") # Adds a tiny bit of spacing to align with the header
    # 'tertiary' type makes it look like subtle text instead of a clunky button
    if st.button("🗑️ Remove Target", type="tertiary", help="Remove this commodity from your dashboard"):
        # Logic to delete the item
        if selected_category in st.session_state.custom_foods and selected_commodity in st.session_state.custom_foods[selected_category]:
            del st.session_state.custom_foods[selected_category][selected_commodity]
            if not st.session_state.custom_foods[selected_category]:
                del st.session_state.custom_foods[selected_category]
        else:
            st.session_state.deleted_foods.append((selected_category, selected_commodity))
        st.rerun() # Refresh the page instantly

# Warning if ticker is dead
if price_usd == 0.0:
    st.error(f"⚠️ **INTELLIGENCE FAILURE:** Yahoo Finance returned no data for ticker **{details['ticker']}**. The market may be closed, the ticker may be delisted, or the AI guessed an invalid symbol.")

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
        st.warning("Financial chart offline due to missing market data.")

with col_news:
    st.markdown("### 📰 Live OSINT Chatter")
    if news_articles:
        df = pd.DataFrame(news_articles)
        def color_threat(val):
            return f'color: {"#ff4b4b" if val < 0 else "#00cc96"}'
        st.dataframe(df.style.map(color_threat, subset=['Threat Score']), hide_index=True)
    else:
        st.info("No immediate threats detected in global news chatter.")
