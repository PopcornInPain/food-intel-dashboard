import streamlit as st
import yfinance as yf
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from groq import Groq

# --- SETUP ---
st.set_page_config(page_title="Food Supply Intel", layout="wide")

# Download sentiment dictionary silently
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

# Initialize AI Client securely
api_key = st.secrets.get("GROQ_API_KEY")
if api_key:
    groq_client = Groq(api_key=api_key)
else:
    groq_client = None

# --- COMMODITY DATABASE ---
COMMODITIES = {
    "Wheat": {"ticker": "ZW=F", "search_term": "wheat"},
    "Corn": {"ticker": "ZC=F", "search_term": "corn"},
    "Soybeans": {"ticker": "ZS=F", "search_term": "soybeans"}
}

# --- FUNCTIONS ---
def get_financial_data(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="1mo")
    today_price = hist['Close'].iloc[-1]
    yesterday_price = hist['Close'].iloc[-2]
    percent_change = ((today_price - yesterday_price) / yesterday_price) * 100
    return today_price, percent_change, hist

def get_news_data(search_term):
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

def get_ai_brief(commodity, articles):
    if not groq_client:
        return "⚠️ AI Offline: API Key missing in Streamlit Secrets."
    headlines = [art['Headline'] for art in articles[:5]]
    prompt = f"""
    Act as a CIA intelligence analyst specializing in global food security.
    Read these recent news headlines about {commodity}.
    Write a 2-sentence 'BLUF' (Bottom Line Up Front) summarizing the current supply chain threat level.
    Make it sound tactical and professional.
    Headlines: {headlines}
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"

# --- DASHBOARD UI ---
st.title("🌍 Global Food Supply Threat Matrix")
st.markdown("Monitoring critical chokepoints in the global agricultural supply chain.")
st.divider()

tabs = st.tabs(["🌾 Wheat", "🌽 Corn", "🌱 Soybeans"])

for tab, (commodity_name, details) in zip(tabs, COMMODITIES.items()):
    with tab:
        st.header(f"Target: {commodity_name}")
        
        current_price, price_change, price_history = get_financial_data(details["ticker"])
        avg_sentiment, news_articles = get_news_data(details["search_term"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"Current Price (Per Bushel)", value=f"${current_price:.2f}", delta=f"{price_change:.2f}%")
        with col2:
            st.metric(label="OSINT Sentiment Score", value=f"{avg_sentiment:.2f}", delta="Negative = High Threat", delta_color="inverse")
        with col3:
            threat_level = "🔴 HIGH RISK" if price_change > 1.5 or avg_sentiment < -0.2 else "🟢 NORMAL"
            st.metric(label="System Status", value=threat_level)
            
        st.markdown("### 🤖 AI Analyst Brief (BLUF)")
        with st.spinner('Decrypting intel...'):
            ai_brief = get_ai_brief(commodity_name, news_articles)
            st.info(ai_brief)

        col_chart, col_news = st.columns([2, 1])
        with col_chart:
            st.markdown("### 📈 30-Day Price Action (FININT)")
            fig = go.Figure(data=[go.Candlestick(x=price_history.index,
                        open=price_history['Open'], high=price_history['High'],
                        low=price_history['Low'], close=price_history['Close'])])
            fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_news:
            st.markdown("### 📰 Live OSINT Chatter")
            df = pd.DataFrame(news_articles)
            def color_threat(val):
                color = '#ff4b4b' if val < 0 else '#00cc96'
                return f'color: {color}'
            st.dataframe(df.style.map(color_threat, subset=['Threat Score']), hide_index=True)
