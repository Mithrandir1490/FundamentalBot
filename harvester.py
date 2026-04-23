import yfinance as yf
import pandas as pd
import time
import random

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

def run_harvest():
    raw_data = []
    total = len(TICKERS)
    print(f"🚀 Iniciando recolección de {total} activos...")
    
    for i, ticker in enumerate(TICKERS):
        try:
            print(f"[{i+1}/{total}] 🔍 Consultando: {ticker}")
            
            t = yf.Ticker(ticker)
            info = t.info
            
            if info and 'sector' in info:
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
            
            # PAUSA CRÍTICA: Al ser muchos tickers, necesitamos ir despacio 
            # para que Yahoo no nos bloquee a mitad del camino.
            time.sleep(random.uniform(1.5, 3.5))
                
        except Exception as e:
            print(f"⚠️ Error en {ticker}: {e}")
            continue

    if raw_data:
        df = pd.DataFrame(raw_data)
        df.to_csv("logit_data.csv", index=False)
        print(f"📊 ¡Misión cumplida! Se guardaron {len(df)} activos.")
    else:
        print("❌ El robot no pudo recolectar nada.")

if __name__ == "__main__":
    run_harvest()
