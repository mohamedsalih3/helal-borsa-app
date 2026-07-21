import os
import time
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from datetime import datetime, timezone
from dotenv import load_dotenv

# Çevresel değişkenleri yükle
load_dotenv()
POLYGON_KEY = os.getenv("POLYGON_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API'sini yapılandır
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Streamlit Sayfa Ayarları
st.set_page_config(page_title="Helal Penny Stock AI Analisti", page_icon="📊", layout="wide")

# Başlık ve Açıklama
st.title("📊 Helal Penny Stock AI Analisti & Tarayıcısı")
st.write("Resmi Google Gemini Yapay Zekası ve Canlı TradingView Grafikleriyle güçlendirilmiş borsa terminaliniz.")

# Sol Panel (Sidebar)
st.sidebar.header("⚙️ Uygulama Ayarları")

# Güncellenmiş ve arındırılmış helal borsa listemiz (KNDI çıkarıldı, toplam 24 hisse)
default_watchlist = (
    "TDTH, INUV, TYGO, JZXN, HAO, CCTG, LHSW, LGCL, LIMN, POAS, SOAR, JLHL, "
    "CAN, REKR, ZVIA, CERS, SGMO, ORGO, VSEE, DDD, SENS, AMTX, OPTT, AEMD"
)
watchlist_input = st.sidebar.text_area("Tarama Listesi (Hisseleri virgülle ayırın):", value=default_watchlist)
watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

# Fiyat Sınırı Kaydırıcısı
price_limit = st.sidebar.slider("Maksimum Hisse Fiyatı ($):", min_value=1.0, max_value=20.0, value=10.0, step=0.5)

# Haber Yaş Sınırı Seçimi (Kota Dostu)
news_age_option = st.sidebar.selectbox(
    "Haber Yaş Sınırı:",
    options=["Son 24 Saat", "Son 3 Gün (Önerilen)", "Son 1 Hafta", "Sınırsız (En Son Haber)"],
    index=1
)

# Yaş sınırını saate çeviren harita
age_map = {
    "Son 24 Saat": 24,
    "Son 3 Gün (Önerilen)": 72,
    "Son 1 Hafta": 168,
    "Sınırsız (En Son Haber)": 999999
}
max_news_hours = age_map[news_age_option]

# ----------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ----------------------------------------------------

def get_polygon_data(ticker):
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
        print(f"Haber çekme hatası ({ticker}): {e}")
    return None

def analyze_news_with_gemini(ticker, news_title):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"Sen bir borsa analistisin. '{ticker}' hissesi hakkında şu haber yayınlandı: '{news_title}'.\n"
            "Bu haberi analiz et ve bana sadece şu formatta yanıt ver:\n"
            "YON: [Olumlu / Olumsuz / Nötr]\n"
            "ACIKLAMA: [Haberin hisseye olası etkisini açıklayan maksimum 1 Türkçe cümle.]"
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"YON: Nötr\nACIKLAMA: Yapay zeka analiz hatası: {str(e)}"

@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(ticker, news_title):
    return analyze_news_with_gemini(ticker, news_title)

# ----------------------------------------------------
# ANA TARAMA TETİKLEYİCİSİ
# ----------------------------------------------------

if st.button("🔍 Yapay Cihan / Borsa Analizini Başlat"):
    if not POLYGON_KEY or not GEMINI_KEY:
        st.error("Lütfen .env dosyasına veya Streamlit Secrets alanına POLYGON_API_KEY ve GEMINI_API_KEY ekleyin.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_items = len(watchlist)
        
        for index, ticker in enumerate(watchlist):
            status_text.text(f"Analiz ediliyor ({index+1}/{total_items}): {ticker}...")
            
            # Fiyat ve Hacim Çek
            stock_data = get_polygon_data(ticker)
            if stock_data:
                price = stock_data["price"]
                volume = stock_data["volume"]
                
                if price <= price_limit:
                    news_data = get_polygon_news(ticker, max_news_hours)
                    
                    if news_data:
                        news_title = news_data["title"]
                        news_url = news_data["url"]
                        news_date = news_data["date_str"]
                        ai_analysis = get_cached_ai_analysis(ticker, news_title)
                        
                        yon = "Nötr"
                        aciklama = "Yorum yok."
                        for line in ai_analysis.split("\n"):
                            if line.startswith("YON:"):
                                yon = line.replace("YON:", "").strip()
                            elif line.startswith("ACIKLAMA:"):
                                aciklama = line.replace("ACIKLAMA:", "").strip()
                    else:
                        news_title = f"Belirlenen zaman diliminde ({news_age_option}) güncel haber bulunamadı."
                        news_url = "#"
                        news_date = "Yok"
                        yon = "Nötr"
                        aciklama = "Güncel haber bulunmadığı için yapay zeka analizi kota tasarrufu amacıyla atlandı."
                        
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": f"${price:.2f}",
                        "Hacim": f"{volume:,}",
                        "Haber": news_title,
                        "Haber Linki": news_url,
                        "Haber Tarihi": news_date,
                        "Yön": yon,
                        "AI Yorumu": aciklama
                    })
            
            time.sleep(12)
            progress_bar.progress((index + 1) / total_items)
            
        status_text.text("Tarama ve Yapay Zeka Analizi Başarıyla Tamamlandı!")
        
        # Sonuçları Görselleştir
        st.write("---")
        st.subheader("📈 Analiz Sonuçları ve Sinyaller")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df[["Hisse", "Fiyat", "Hacim", "Haber Tarihi", "Yön"]], use_container_width=True)
            
            st.write("### 🧠 Yapay Zeka Detaylı Analiz Kartları ve Grafik Terminali")
            
            # NYSE veya AMEX üzerinde olan hisseler için borsa belirleme
            exchange_map = {"INUV": "AMEX", "SOAR": "NYSE", "OPTT": "AMEX"}
            
            for res in results:
                if "Olumlu" in res["Yön"]:
                    color_border = "#00C805" # Midas Yeşili
                    emoji = "🟩 Bullish (Yukarı Yönlü Sinyal)"
                elif "Olumsuz" in res["Yön"]:
                    color_border = "#FF3B30" # Kırmızı
                    emoji = "🟥 Bearish (Aşağı Yönlü Sinyal)"
                else:
                    color_border = "#8E8E93" # Gri
                    emoji = "🟨 Nötr"
                
                # TradingView için borsa belirleme (Varsayılan NASDAQ)
                exchange = exchange_map.get(res["Hisse"], "NASDAQ")
                tv_symbol = f"{exchange}:{res['Hisse']}"
                
                # TradingView Mini Mum Grafik HTML'i
                chart_html = f"""
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
                  {{
                    "symbol": "{tv_symbol}",
                    "width": "100%",
                    "height": 220,
                    "locale": "en",
                    "dateRange": "12M",
                    "colorTheme": "dark",
                    "isTransparent": true,
                    "autosize": false,
                    "largeChartUrl": ""
                  }}
                  </script>
                </div>
                """
                
                # TradingView Teknik Analiz Kadranı (Gauge) HTML'i
                gauge_html = f"""
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
                  {{
                    "interval": "1D",
                    "width": "100%",
                    "isTransparent": true,
                    "height": 220,
                    "symbol": "{tv_symbol}",
                    "showIntervalTabs": false,
                    "locale": "en",
                    "colorTheme": "dark"
                  }}
                  </script>
                </div>
                """

                # Kart Başlığı
                st.markdown(f"""
                <div style="border: 2px solid {color_border}; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(255,255,255,0.03)">
                    <h3 style="margin:0;">📊 Hisse: <span style="color:#00D2FF">{res['Hisse']}</span> | Fiyat: {res['Fiyat']} | Hacim: {res['Hacim']}</h3>
                    <p style="margin-top:5px; margin-bottom:5px;"><b>Yapay Zeka Sinyali:</b> {emoji}</p>
                    <p style="margin-bottom:5px;"><b>Haber:</b> <a href="{res['Haber Linki']}" target="_blank">{res['Haber']}</a> <i>({res['Haber Tarihi']})</i></p>
                    <p style="font-size: 15px; background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                        💡 <b>Yapay Zeka Analizi:</b> {res['AI Yorumu']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # TradingView Grafiği ve Teknik Gösterge Kadrani Yan Yana (Telefonda alt alta otomatik sığar)
                col1, col2 = st.columns(2)
                with col1:
                    components.html(chart_html, height=230)
                with col2:
                    components.html(gauge_html, height=230)
                
                # Midas Hızlı Erişim İşlem Butonu
                st.markdown(f"""
                <div style="margin-bottom: 40px; margin-top: 5px;">
                    <a href="https://www.getmidas.com/" target="_blank" style="text-decoration:none;">
                        <button style="background-color:#00C805; color:white; border:none; padding:10px 20px; border-radius:5px; font-weight:bold; cursor:pointer; font-size:15px; width:100%;">
                            📱 Midas Uygulamasında İşlem Yap
                        </button>
                    </a>
                </div>
                <hr style="border: 1px solid rgba(255,255,255,0.1); margin-bottom:30px;">
                """, unsafe_allow_html=True)
                
        else:
            st.warning("Filtrelerinize uyan uygun bir hisse bulunamadı.")
