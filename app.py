import os
import time
import requests
import datetime
import pandas as pd
import streamlit as st
import google.generativeai as genai
from datetime import datetime as dt, timezone
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

# Watchlist
default_watchlist = (
    "TDTH, INUV, TYGO, JZXN, HAO, CCTG, LHSW, LGCL, LIMN, POAS, SOAR, JLHL, "
    "CAN, REKR, ZVIA, CERS, SGMO, ORGO, VSEE, DDD, SENS, AMTX, OPTT, AEMD"
)
watchlist_input = st.sidebar.text_area("Scan List (comma-separated):", value=default_watchlist)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

# Price Limit Slider
price_limit = st.sidebar.slider("Maximum Stock Price ($):", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

# News Age Limit Selection
news_age_option = st.sidebar.selectbox(
    "News Age Limit:",
    options=["Last 24 Hours", "Last 3 Days (Recommended)", "Last 1 Week", "Unlimited (Latest News)"],
    index=1
)

age_map = {
    "Last 24 Hours": 24,
    "Last 3 Days (Recommended)": 72,
    "Last 1 Week": 168,
    "Unlimited (Latest News)": 999999
}
max_news_hours = age_map[news_age_option]

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def get_polygon_market_data(ticker):
    """Fiyatı, hacmi ve son 14 günlük ortalamaya göre Hacim Gücünü (Çarpanı) tek seferde çeker."""
    today_str = dt.now().strftime("%Y-%m-%d")
    start_str = (dt.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{today_str}?adjusted=true&sort=desc&limit=15&apiKey={POLYGON_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                recent = results[0]
                price = recent.get("c")
                volume = recent.get("v")
                
                # Önceki günlerin hacim ortalaması
                prev_days = results[1:]
                if prev_days:
                    avg_vol = sum([day.get("v", 0) for day in prev_days]) / len(prev_days)
                else:
                    avg_vol = volume if volume else 1
                
                vol_strength = volume / avg_vol if avg_vol > 0 else 1.0
                return {
                    "price": price,
                    "volume": volume,
                    "vol_strength": f"{vol_strength:.1f}x"
                }
    except Exception as e:
        print(f"Hata ({ticker}): {e}")
    return None

def get_polygon_news(ticker, max_hours):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                pub_utc = results[0].get("published_utc")
                if pub_utc:
                    pub_date = dt.fromisoformat(pub_utc.replace("Z", "+00:00"))
                    now = dt.now(timezone.utc)
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

@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(ticker, news_title):
    return analyze_news_with_gemini(ticker, news_title)

# --- STOP BUTTON STATE MANAGEMENT ---
if "scanning" not in st.session_state:
    st.session_state.scanning = False
if "stop_scan" not in st.session_state:
    st.session_state.stop_scan = False

if st.session_state.scanning:
    if st.sidebar.button("⏹️ Stop Scanning", key="stop_btn"):
        st.session_state.stop_scan = True
        st.session_state.scanning = False
        st.sidebar.warning("Stop signal sent! Execution will halt on the next step...")

# ----------------------------------------------------
# MAIN SCAN TRIGGER
# ----------------------------------------------------

start_scan = st.button("🔍 Start AI Analysis & Scan", disabled=st.session_state.scanning)

if start_scan:
    if not POLYGON_KEY or not GEMINI_KEY:
        st.error("Please add POLYGON_API_KEY and GEMINI_API_KEY to your secrets.")
    else:
        st.session_state.scanning = True
        st.session_state.stop_scan = False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_items = len(watchlist)
        
        for index, ticker in enumerate(watchlist):
            if st.session_state.stop_scan:
                status_text.warning("Scanning was interrupted by the user!")
                st.session_state.scanning = False
                st.session_state.stop_scan = False
                st.stop()
            
            status_text.text(f"Analyzing ({index+1}/{total_items}): {ticker}...")
            
            # Yeni tek istekli veri çekme fonksiyonumuz
            stock_data = get_polygon_market_data(ticker)
            if stock_data:
                price = stock_data["price"]
                volume = stock_data["volume"]
                vol_strength = stock_data["vol_strength"]
                
                if price <= price_limit:
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
                        news_title = f"No current news found within the selected timeframe ({news_age_option})."
                        news_url = "#"
                        news_date = "N/A"
                        sentiment = "Neutral"
                        analysis = "AI analysis skipped to conserve quota since no recent news is available."
                        
                    results.append({
                        "Ticker": ticker,
                        "Price": f"${price:.2f}",
                        "Volume": f"{volume:,}",
                        "Vol Strength": vol_strength,
                        "News": news_title,
                        "News Link": news_url,
                        "News Date": news_date,
                        "Sentiment": sentiment,
                        "AI Comment": analysis
                    })
            
            time.sleep(12)
            progress_bar.progress((index + 1) / total_items)
            
        status_text.text("Scan and AI Analysis Successfully Completed!")
        st.session_state.scanning = False
        
        # Visualize Results
        st.write("---")
        st.subheader("📈 Analysis Results & Signals")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df[["Ticker", "Price", "Volume", "Vol Strength", "News Date", "Sentiment"]], use_container_width=True)
            
            st.write("### 🧠 AI Detailed Analysis Cards & Action Terminal")
            
            exchange_map = {"INUV": "AMEX", "SOAR": "NYSE", "OPTT": "AMEX"}
            
            for res in results:
                if "Bullish" in res["Sentiment"]:
                    color_border = "#00C805"
                    emoji = "🟩 Bullish (Upward Signal)"
                elif "Bearish" in res["Sentiment"]:
                    color_border = "#FF3B30"
                    emoji = "🟥 Bearish (Downward Signal)"
                else:
                    color_border = "#8E8E93"
                    emoji = "🟨 Neutral"
                
                exchange = exchange_map.get(res["Ticker"], "NASDAQ")
                tv_symbol = f"{exchange}:{res['Ticker']}"
                tv_chart_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"

                st.markdown(f"""
                <div style="border: 2px solid {color_border}; padding: 15px; border-radius: 10px; margin-bottom: 15px; background-color: rgba(255,255,255,0.03)">
                    <h3 style="margin:0;">📊 Ticker: <span style="color:#00D2FF">{res['Ticker']}</span> | Price: {res['Price']} | Volume: {res['Volume']} (<span style="color:#00C805">{res['Vol Strength']}</span>)</h3>
                    <p style="margin-top:5px; margin-bottom:5px;"><b>AI Signal:</b> {emoji}</p>
                    <p style="margin-bottom:5px;"><b>Latest News:</b> <a href="{res['News Link']}" target="_blank">{res['News']}</a> <i>({res['News Date']})</i></p>
                    <p style="font-size: 15px; background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        💡 <b>AI Analysis:</b> {res['AI Comment']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <a href="{tv_chart_url}" target="_blank" style="text-decoration:none;">
                        <button style="background-color:#1E293B; color:#F8FAFC; border:1px solid #475569; padding:10px 20px; border-radius:5px; font-weight:bold; cursor:pointer; font-size:15px; width:100%;">
                            📊 View Live Chart
                        </button>
                    </a>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <a href="https://www.getmidas.com/" target="_blank" style="text-decoration:none;">
                        <button style="background-color:#0062FF; color:white; border:none; padding:10px 20px; border-radius:5px; font-weight:bold; cursor:pointer; font-size:15px; width:100%;">
                            📱 Trade on Midas App
                        </button>
                    </a>
                    """, unsafe_allow_html=True)
                
                st.markdown('<hr style="border: 1px solid rgba(255,255,255,0.05); margin-bottom:20px; margin-top:20px;">', unsafe_allow_html=True)
                
        else:
            st.warning("No matching stocks found with your current filters.")
