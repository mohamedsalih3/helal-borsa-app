import os
import time
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
POLYGON_KEY = os.getenv("POLYGON_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Streamlit Page Settings
st.set_page_config(page_title="Halal Penny Stock AI Analyst", page_icon="📊", layout="wide")

# Title and Description
st.title("📊 Halal Penny Stock AI Analyst & Scanner")
st.write("Powered by official Google Gemini AI & Live TradingView Charts.")

# Sidebar - User Inputs
st.sidebar.header("⚙️ Application Settings")

# Verified halal stock watchlist (KNDI removed, total 24 stocks)
default_watchlist = (
    "TDTH, INUV, TYGO, JZXN, HAO, CCTG, LHSW, LGCL, LIMN, POAS, SOAR, JLHL, "
    "CAN, REKR, ZVIA, CERS, SGMO, ORGO, VSEE, DDD, SENS, AMTX, OPTT, AEMD"
)
watchlist_input = st.sidebar.text_area("Scan List (comma-separated):", value=default_watchlist)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

# Price Limit Slider
price_limit = st.sidebar.slider("Maximum Stock Price ($):", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

# News Age Limit Selection (Quota Friendly)
news_age_option = st.sidebar.selectbox(
    "News Age Limit:",
    options=["Last 24 Hours", "Last 3 Days (Recommended)", "Last 1 Week", "Unlimited (Latest News)"],
    index=1
)

# Map age option to hours
age_map = {
    "Last 24 Hours": 24,
    "Last 3 Days (Recommended)": 72,
    "Last 1 Week": 168,
    "Unlimited (Latest News)": 999999
}
max_news_hours = age_map[news_age_option]

# --- STOP BUTTON STATE MANAGEMENT ---
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "stop_scan" not in st.session_state:
    st.session_state.stop_scan = False

# Show stop button in the sidebar if scanning is active
if st.session_state.scanning:
    if st.sidebar.button("⏹️ Stop Scanning", key="stop_btn"):
        st.session_state.stop_scan = True
        st.session_state.scanning = False
        st.sidebar.warning("Stop signal sent! Execution will halt on the next step...")

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def get_polygon_data(ticker):
    """Fetches the previous day's close price and trading volume."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={POLYGON_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {"price": result.get("c"), "volume": result.get("v")}
    except:
        pass
    return None

def get_polygon_news(ticker, max_hours):
    """Fetches the latest news article and validates its publication age."""
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                pub_utc = results[0].get("published_utc")
                if pub_utc:
                    pub_date = datetime.fromisoformat(pub_utc.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    diff_hours = (now - pub_date).total_seconds() / 3600
                    
                    if diff_hours <= max_hours:
                        return {
                            "title": results[0].get("title"),
                            "url": results[0].get("article_url"),
                            "date_str": pub_date.strftime("%d-%m-%Y %H:%M")
                        }
    except Exception as e:
        print(f"Error fetching news ({ticker}): {e}")
    return None

def analyze_news_with_gemini(ticker, news_title):
    """Analyzes the news sentiment using official Google Generative AI in English."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (
            f"You are a stock market analyst. The following news was published about '{ticker}': '{news_title}'.\n"
            "Analyze this news and reply strictly in the following format:\n"
            "SENTIMENT: [Bullish / Bearish / Neutral]\n"
            "ANALYSIS: [A maximum of 1 sentence in English explaining the possible impact of the news on the stock price.]"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"SENTIMENT: Neutral\nANALYSIS: AI Analysis Error: {str(e)}"

# Streamlit Caching for Quota Preservation
@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(ticker, news_title):
    return analyze_news_with_gemini(ticker, news_title)

# ----------------------------------------------------
# MAIN SCAN TRIGGER
# ----------------------------------------------------

# Scan button (disabled while actively scanning)
start_scan = st.button("🔍 Start AI Analysis & Scan", disabled=st.session_state.scanning)

if start_scan:
    if not POLYGON_KEY or not GEMINI_KEY:
        st.error("Please add POLYGON_API_KEY and GEMINI_API_KEY to your .env file or Streamlit Secrets.")
    else:
        st.session_state.scanning = True
        st.session_state.stop_scan = False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_items = len(watchlist)
        
        for index, ticker in enumerate(watchlist):
            # Halt if stop signal is triggered
            if st.session_state.stop_scan:
                status_text.warning("Scanning was interrupted by the user!")
                st.session_state.scanning = False
                st.session_state.stop_scan = False
                st.stop()
            
            status_text.text(f"Analyzing ({index+1}/{total_items}): {ticker}..." )
            
            # Fetch Price and Volume
            stock_data = get_polygon_data(ticker)
            if stock_data:
                price = stock_data["price"]
                volume = stock_data["volume"]
                
                if price <= price_limit:
                    # Fetch News with selected age limit
                    news_data = get_polygon_news(ticker, max_news_hours)
                    
                    if news_data:
                        news_title = news_data["title"]
                        news_url = news_data["url"]
                        news_date = news_data["date_str"]
                        ai_analysis = get_cached_ai_analysis(ticker, news_title)
                        
                        sentiment = "Neutral"
                        analysis = "No comment."
                        for line in ai_analysis.split("\n"):
                            if line.startswith("SENTIMENT:"):
                                sentiment = line.replace("SENTIMENT:", "").strip()
                            elif line.startswith("ANALYSIS:"):
                                analysis = line.replace("ANALYSIS:", "").strip()
                    else:
                        # Skip Gemini to save quota if no recent news exists
                        news_title = f"No current news found within the selected timeframe ({news_age_option})."
                        news_url = "#"
                        news_date = "N/A"
                        sentiment = "Neutral"
                        analysis = "AI analysis skipped to conserve quota since no recent news is available."
                        
                    results.append({
                        "Ticker": ticker,
                        "Price": f"${price:.2f}",
                        "Volume": f"{volume:,}",
                        "News": news_title,
                        "News Link": news_url,
                        "News Date": news_date,
                        "Sentiment": sentiment,
                        "AI Comment": analysis
                    })
            
            # 12-second delay to avoid free tier rate-limiting
            time.sleep(12)
            progress_bar.progress((index + 1) / total_items)
            
        status_text.text("Scan and AI Analysis Successfully Completed!")
        st.session_state.scanning = False
        
        # Visualize Results
        st.write("---")
        st.subheader("📈 Analysis Results & Signals")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df[["Ticker", "Price", "Volume", "News Date", "Sentiment"]], use_container_width=True)
            
            st.write("### 🧠 AI Detailed Analysis Cards & Live Chart Terminal")
            
            # Map non-NASDAQ exchanges
            exchange_map = {"INUV": "AMEX", "SOAR": "NYSE", "OPTT": "AMEX"}
            
            for res in results:
                if "Bullish" in res["Sentiment"]:
                    color_border = "#00C805" # Midas Green for positive sentiment border
                    emoji = "🟩 Bullish (Upward Signal)"
                elif "Bearish" in res["Sentiment"]:
                    color_border = "#FF3B30" # Red
                    emoji = "🟥 Bearish (Downward Signal)"
                else:
                    color_border = "#8E8E93" # Gray
                    emoji = "🟨 Neutral"
                
                exchange = exchange_map.get(res["Ticker"], "NASDAQ")
                tv_symbol = f"{exchange}:{res['Ticker']}"
                
                # Pure HTML Live Chart
                chart_html = f"""
                <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&theme=dark&style=1&timezone=Etc%2FUTC" 
                        width="100%" 
                        height="250" 
                        frameborder="0" 
                        allowtransparency="true" 
                        scrolling="no" 
                        style="margin: 0; padding: 0; border-radius: 10px;">
                </iframe>
                """
                
                # Pure HTML Technical Analysis Gauge
                gauge_html = f"""
                <iframe src="https://s.tradingview.com/embed-widget/technical-analysis/?symbol={tv_symbol}&interval=1D&theme=dark" 
                        width="100%" 
                        height="250" 
                        frameborder="0" 
                        allowtransparency="true" 
                        scrolling="no" 
                        style="margin: 0; padding: 0; border-radius: 10px;">
                </iframe>
                """

                # Render Card
                st.markdown(f"""
                <div style="border: 2px solid {color_border}; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(255,255,255,0.03)">
                    <h3 style="margin:0;">📊 Ticker: <span style="color:#00D2FF">{res['Ticker']}</span> | Price: {res['Price']} | Volume: {res['Volume']}</h3>
                    <p style="margin-top:5px; margin-bottom:5px;"><b>AI Signal:</b> {emoji}</p>
                    <p style="margin-bottom:5px;"><b>Latest News:</b> <a href="{res['News Link']}" target="_blank">{res['News']}</a> <i>({res['News Date']})</i></p>
                    <p style="font-size: 15px; background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        💡 <b>AI Analysis:</b> {res['AI Comment']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Render Charts side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    components.html(chart_html, height=255)
                with col2:
                    components.html(gauge_html, height=255)
                
                # Midas Trading Deep Link Button (Blue #0062FF)
                st.markdown(f"""
                <div style="margin-bottom: 40px; margin-top: 5px;">
                    <a href="https://www.getmidas.com/" target="_blank" style="text-decoration:none;">
                        <button style="background-color:#0062FF; color:white; border:none; padding:10px 20px; border-radius:5px; font-weight:bold; cursor:pointer; font-size:15px; width:100%;">
                            📱 Trade on Midas App
                        </button>
                    </a>
                </div>
                <hr style="border: 1px solid rgba(255,255,255,0.1); margin-bottom:30px;">
                """, unsafe_allow_html=True)
                
        else:
            st.warning("No matching stocks found with your current filters.")