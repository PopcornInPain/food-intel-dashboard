import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from groq import Groq

# --- SETUP & CONFIG ---
st.set_page_config(page_title="Food Supply Intel", layout="wide")

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

# --- EXPANDED COMMODITY DATABASE ---
COMMODITIES = {
    "Wheat": {"ticker": "ZW=F", "search": "wheat", "multiplier": 0.01, "unit": "Bushel"},
    "Rice": {"ticker": "ZR=F", "search": "rice", "multiplier": 0.01, "unit": "Hundredweight"},
    "Cocoa": {"ticker": "CC=F", "search": "cocoa", "multiplier": 1.0, "unit": "Metric Ton"},
    "Sugar": {"ticker": "SB=F", "search": "sugar", "multiplier": 0.01, "unit": "Pound"},
    "Corn": {"ticker": "ZC=F", "search": "corn", "multiplier": 0.01, "unit": "Bushel"},
    "Soybeans": {"ticker": "ZS=F", "search": "soybeans", "multiplier": 0.01, "unit": "Bushel"}
}

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
        url = f"https://news.google.com/rss/search?q={search_term}+export+ban+OR+{search_term}+drought+OR+{search_term}+shortage&hl=en-US&gl=US&ceid=US:en"
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
    prompt = f"""
    Act as a CIA intelligence analyst for food security.
    Target: {commodity}. Price change today: {price_change:.2f}%.
    Read these headlines: {headlines}
    Write a 2-sentence tactical 'BLUF' (Bottom Line Up Front) summarizing the supply chain threat. 
    Mention if the news justifies the price movement.
    """
    
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# --- DASHBOARD UI ---
st.title("🌍 Global Food Supply Threat Matrix")
st.markdown(f"**Live Forex Rate:** 1 USD = {USD_TO_MYR:.2f} MYR")

# --- NEW: FIELD MANUAL / TUTORIAL ---
with st.expander("📖 FIELD MANUAL: How to read this intelligence dashboard", expanded=False):
    st.markdown("""
    ### Welcome to the Food Supply Threat Intelligence System
    This dashboard acts as an early warning radar for global food security. It tracks financial markets (FININT) and global news (OSINT) in real-time to detect supply chain disruptions before they hit the grocery store.

    #### 🔍 How to interpret the data:
    * **Target Commodities:** Click the tabs below (Wheat, Rice, Cocoa, etc.) to switch between different food supply chains.
    * **Price (USD & MYR):** Live global futures prices. The MYR price is calculated dynamically using live Forex data.
    * **OSINT Sentiment Score:** We scrape global news for threat keywords (drought, export ban, shortage, strike). 
        * `+1.0` = Perfectly safe / positive news.
        * `0.0` = Neutral chatter.
        * `-1.0` = Extreme threat / negative news.
    * **System Status:** The system automatically triggers **🔴 HIGH RISK** if the price suddenly jumps more than 2% in a single day, OR if the news sentiment drops below -0.25.
    * **🤖 AI Analyst (BLUF):** *'Bottom Line Up Front'*. Our AI reads the latest news headlines and cross-references them with today's price action to give you a 2-sentence tactical summary of the current threat landscape.
    * **📈 FININT Chart:** The candlestick chart shows daily price movements over the last 90 days. The **orange line** is the 14-day trend (Moving Average). If the current price breaks far above the orange line, a panic-buying event or supply shock may be occurring.
    """)

st.divider()

# Create dynamic tabs
tab_names = [f"🎯 {name}" for name in COMMODITIES.keys()]
tabs = st.tabs(tab_names)

for tab, (commodity_name, details) in zip(tabs, COMMODITIES.items()):
    with tab:
        st.header(f"Target: {commodity_name}")
        
        price_usd, price_change, trend_ma, price_history = get_financial_data(details["ticker"], details["multiplier"])
        avg_sentiment, news_articles = get_news_data(details["search"])
        
        price_myr = price_usd * USD_TO_MYR
        
        # Top Row Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label=f"Price ({details['unit']})", 
                      value=f"${price_usd:.2f}", 
                      delta=f"{price_change:.2f}%")
        with col2:
            st.metric(label="Price in MYR", 
                      value=f"RM {price_myr:.2f}")
        with col3:
            st.metric(label="OSINT Sentiment", 
                      value=f"{avg_sentiment:.2f}", 
                      delta="Negative = Threat", delta_color="inverse")
        with col4:
            threat_level = "🔴 HIGH RISK" if price_change > 2.0 or avg_sentiment < -0.25 else "🟢 NORMAL"
            st.metric(label="System Status", value=threat_level)
            
        # AI Analyst Brief
        st.markdown("### 🤖 AI Analyst Brief (BLUF)")
        with st.spinner('Decrypting intel...'):
            ai_brief = get_ai_brief(commodity_name, news_articles, price_change)
            st.info(ai_brief)

        # Charts and News Layout
        col_chart, col_news = st.columns([2, 1])
        
        with col_chart:
            st.markdown("### 📈 90-Day Price Action & Trend (FININT)")
            if not price_history.empty:
                fig = go.Figure()
                # Candlestick chart
                fig.add_trace(go.Candlestick(x=price_history.index,
                            open=price_history['Open'] * details["multiplier"], 
                            high=price_history['High'] * details["multiplier"],
                            low=price_history['Low'] * details["multiplier"], 
                            close=price_history['Close'] * details["multiplier"],
                            name="Price"))
                # Moving Average Trendline
                fig.add_trace(go.Scatter(x=price_history.index, y=price_history['50_MA'] * details["multiplier"], 
                                         line=dict(color='orange', width=2), name="14-Day Trend"))
                
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_news:
            st.markdown("### 📰 Live OSINT Chatter")
            if news_articles:
                df = pd.DataFrame(news_articles)
                def color_threat(val):
                    return f'color: {"#ff4b4b" if val < 0 else "#00cc96"}'
                st.dataframe(df.style.map(color_threat, subset=['Threat Score']), hide_index=True)
