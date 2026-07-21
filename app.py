import os
import time
import requests
import pandas as pd
import streamlit as st
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
st.write("Resmi Google Gemini Yapay Zekasıyla güçlendirilmiş, VS Code üzerinde çalışan yerel borsa asistanınız.")

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

# --- YENİ: Haber Yaş Sınırı Seçimi (Kota Dostu) ---
news_age_option = st.sidebar.selectbox(
    "Haber Yaş Sınırı:",
    options=["Son 24 Saat", "Son 3 Gün (Önerilen)", "Son 1 Hafta", "Sınırsız (En Son Haber)"],
    index=1 # Varsayılan olarak Cuma akşamı haberlerini kaçırmamak için "Son 3 Gün" seçilidir
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
    """Hissenin son fiyat ve hacmini çeker."""
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
    """Hissenin en son haberini çeker ve tarih kontrolü yapar."""
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=1&apiKey={POLYGON_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                pub_utc = results[0].get("published_utc")
                if pub_utc:
                    # ISO formatındaki tarihi okuyup saat dilimini eşitliyoruz
                    pub_date = datetime.fromisoformat(pub_utc.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    # Aradaki saat farkını hesaplıyoruz
                    diff_hours = (now - pub_date).total_seconds() / 3600
                    
                    # Eğer haber belirlediğimiz yaş sınırının içindeyse kabul et
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
    """Resmi Google Generative AI kütüphanesiyle haberi doğrudan analiz eder."""
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

# Streamlit Önbellek Koruması (Kota Aşımını Engellemek İçin)
@st.cache_data(show_spinner=False)
def get_cached_ai_analysis(ticker, news_title):
    return analyze_news_with_gemini(ticker, news_title)

# ----------------------------------------------------
# ANA TARAMA TETİKLEYİCİSİ
# ----------------------------------------------------

if st.button("🔍 Yapay Zeka Analizini ve Taramayı Başlat"):
    if not POLYGON_KEY or not GEMINI_KEY:
        st.error("Lütfen .env dosyasına POLYGON_API_KEY ve GEMINI_API_KEY ekleyin.")
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
                    # Haberi Çek (Yeni Eklenen max_news_hours filtresiyle)
                    news_data = get_polygon_news(ticker, max_news_hours)
                    
                    if news_data:
                        news_title = news_data["title"]
                        news_url = news_data["url"]
                        news_date = news_data["date_str"]
                        
                        # Yapay Zeka Analizi (Önbellekli ve Stabil)
                        ai_analysis = get_cached_ai_analysis(ticker, news_title)
                        
                        # Yapay zeka yanıtını ayrıştır
                        yon = "Nötr"
                        aciklama = "Yorum yok."
                        for line in ai_analysis.split("\n"):
                            if line.startswith("YON:"):
                                yon = line.replace("YON:", "").strip()
                            elif line.startswith("ACIKLAMA:"):
                                aciklama = line.replace("ACIKLAMA:", "").strip()
                    else:
                        # Eğer belirlenen yaş sınırında haber yoksa kota harcamamak için es geç
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
            
            # API limitine takılmamak için 12 saniye bekleme
            time.sleep(12)
            progress_bar.progress((index + 1) / total_items)
            
        status_text.text("Tarama ve Yapay Zeka Analizi Başarıyla Tamamlandı!")
        
        # Sonuçları Görselleştir
        st.write("---")
        st.subheader("📈 Analiz Sonuçları ve Sinyaller")
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df[["Hisse", "Fiyat", "Hacim", "Haber Tarihi", "Yön"]], use_container_width=True)
            
            st.write("### 🧠 Yapay Zeka Detaylı Analiz Kartları")
            
            for res in results:
                if "Olumlu" in res["Yön"]:
                    color_border = "green"
                    emoji = "🟩 Bullish (Yukarı Yönlü Sinyal)"
                elif "Olumsuz" in res["Yön"]:
                    color_border = "red"
                    emoji = "🟥 Bearish (Aşağı Yönlü Sinyal)"
                else:
                    color_border = "gray"
                    emoji = "🟨 Nötr"
                    
                st.markdown(f"""
                <div style="border: 2px solid {color_border}; padding: 15px; border-radius: 10px; margin-bottom: 15px; background-color: rgba(255,255,255,0.05)">
                    <h4>📊 Hisse: <span style="color:cyan">{res['Hisse']}</span> | Fiyat: {res['Fiyat']} | Hacim: {res['Hacim']}</h4>
                    <p><b>Sinyal Durumu:</b> {emoji}</p>
                    <p><b>Yayınlanan Son Haber:</b> <a href="{res['Haber Linki']}" target="_blank">{res['Haber']}</a> <i>({res['Haber Tarihi']})</i></p>
                    <p style="font-size: 16px; background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;">
                        💡 <b>Yapay Zeka Analizi:</b> {res['AI Yorumu']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Filtrelerinize uyan uygun bir hisse bulunamadı.")