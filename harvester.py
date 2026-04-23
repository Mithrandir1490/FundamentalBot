import yfinance as yf
import pandas as pd
import time
import random
import requests

# ==========================================
# LISTA MAESTRA COMPLETA (175 TICKERS)
# ==========================================
TICKERS = [
    "ADYEN.AS", "UBER", "ADP", "DSY.PA", "UNH", "TEM", "OSCR", "HIMS", "DECK", "ADBE", 
    "ACN", "DLO", "FDS", "WKL.AS", "LULU", "NVO", "GEV", "BE", "VRT", "CEG", 
    "NEE", "SRE", "VST", "V", "MA", "MCO", "SPGI", "ISRG", "AXON", "ABNB", 
    "ANET", "BSX", "TTD", "NOW", "CRM", "SCHW", "BLK", "GS", "XOM", "CVX", 
    "CAT", "DE", "FIX", "ETN", "HON", "WM", "SMCI", "ALAB", "CORT", "ONTO", 
    "AX", "VIV", "GHM", "SLAB", "LSCC", "LASR", "SITM", "MCHP", "MRVL", "BAM", 
    "DHR", "QSR", "BABA", "GE", "CPNG", "EXPE", "ROK", "ZBRA", "CGNX", "PATH", 
    "PEGA", "MDT", "PRCT", "OMCL", "SYK", "TER", "LECO", "OII", "FARO", "PTC", 
    "QCOM", "AVAV", "TDY", "KTOS", "NOC", "GD", "RTX", "LHX", "APP", "IREN", 
    "AMAT", "KLAC", "RMBS", "SIMO", "ARM", "SNPS", "CRDO", "GLW", "AMKR", "PWR", 
    "CCJ", "BWXT", "UUUU", "TMQ", "UAMY", "MP", "FCX", "TECK", "SCCO", "IONQ", 
    "RGTI", "COIN", "SPOT", "DDOG", "RXRX", "POET", "RBLX", "CRCL", "BMNR", "ACHR", 
    "BEAM", "MOH", "ENB", "TOST", "AMGN", "FOX", "UTHR", "GOLD", "WBA", "JNJ", 
    "HD", "ABBV", "O", "BLDR", "TPL", "FICO", "DPZ", "URI", "BKNG", "MNST", 
    "WDAY", "SOFI", "NU", "NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", 
    "PANW", "CRWD", "FTNT", "ZS", "OKTA", "SNOW", "PLTR", "LLY", "VRTX", "REGN", 
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "MARA", "RIOT", "WMT", "TGT", "COST", 
    "NFLX", "TSLA", "PYPL", "SHOP", "SE"
]

# ==========================================
# MOTOR RECOLECTOR
# ==========================================
def get_session():
    """Crea una sesión con User-Agent para evitar bloqueos"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

def run_harvest():
    raw_data = []
    sess = get_session()
    total = len(TICKERS)
    
    print(f"🚀 Iniciando recolección de {total} activos...")
    
    for i, ticker in enumerate(TICKERS):
        try:
            print(f"[{i+1}/{total}] Descargando: {ticker}")
            t = yf.Ticker(ticker, session=sess)
            info = t.info
            
            # Extraemos solo lo necesario para el modelo Logit
            raw_data.append({
                "Ticker": ticker,
                "Sector": info.get('sector', 'Other'),
                "MarketCap": info.get('marketCap'),
                "PE_Fwd": info.get('forwardPE'),
                "Growth": info.get('revenueGrowth'),
                "ROE": info.get('returnOnEquity'),
                "Margin": info.get('profitMargins'),
                "D_E": info.get('debtToEquity'),
                "Ebitda_G": info.get('earningsGrowth')
            })
            
            # Pausa inteligente: cada 10 tickers una pausa más larga
            if i % 10 == 0 and i > 0:
                time.sleep(random.uniform(3, 6))
            else:
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            print(f"⚠️ Error en {ticker}: {e}")
            continue

    # Convertir a DataFrame y guardar
    df = pd.DataFrame(raw_data)
    
    # Limpieza básica antes de guardar
    df.to_csv("logit_data.csv", index=False)
    print("✅ Proceso terminado. Archivo 'logit_data.csv' actualizado.")

if __name__ == "__main__":
    run_harvest()
