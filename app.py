import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Bot 5: Radar Pro", layout="wide")

# --- LISTA MAESTRA (Asegúrate de que los tickers sean correctos) ---
TICKERS = list(set([
    "ADYEN.AS", "UBER", "ADP", "DSY.PA", "UNH", "TEM", "OSCR", "HIMS", "DECK", "ADBE", "ACN", "DLO", "FDS", "WKL.AS", "LULU", "NVO", 
    "GEV", "BE", "VRT", "CEG", "NEE", "SRE", "VST", "V", "MA", "MCO", "SPGI", "ISRG", "AXON", "ABNB", "ANET", "BSX", 
    "TTD", "NOW", "CRM", "SCHW", "BLK", "GS", "XOM", "CVX", "CAT", "DE", "FIX", "ETN", "HON", "WM", "SMCI", "ALAB", 
    "CORT", "ONTO", "AX", "VIV", "GHM", "SLAB", "LSCC", "LASR", "SITM", "MCHP", "MRVL", "BAM", "DHR", 
    "QSR", "BABA", "GE", "CPNG", "EXPE", "ROK", "ZBRA", "CGNX", "PATH", "PEGA", "MDT", "PRCT", "OMCL", "SYK", "TER", "LECO", 
    "OII", "FARO", "PTC", "QCOM", "AVAV", "TDY", "KTOS", "NOC", "GD", "RTX", "LHX", "APP", 
    "IREN", "AMAT", "KLAC", "RMBS", "SIMO", "ARM", "SNPS", "CRDO", "GLW", "AMKR", "PWR", "CCJ", "BWXT", 
    "UUUU", "TMQ", "UAMY", "MP", "FCX", "TECK", "SCCO", "IONQ", "RGTI", "COIN", "SPOT", "DDOG", "RXRX", "POET", 
    "RBLX", "CRCL", "BMNR", "ACHR", "BEAM", "MOH", "ENB", "TOST", "AMGN", "FOX", "UTHR", "GOLD", "WBA", 
    "JNJ", "HD", "ABBV", "O", "BLDR", "TPL", "FICO", "DPZ", "URI", "BKNG", "MNST", "WDAY", "SOFI", "NU",
    "NVDA", "AMD", "TSM", "AVGO", "MU", "ASML", "LRCX", "PANW", "CRWD", "FTNT", "ZS", "OKTA", "SNOW", "PLTR", "LLY", "VRTX", "REGN",
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "MARA", "RIOT", "WMT", "TGT", "COST", "NFLX", "TSLA", "PYPL", "SHOP", "SE"
]))

# --- FUNCIÓN PARA ENGAÑAR AL SERVIDOR (User-Agent) ---
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

@st.cache_data(ttl=43200)
def get_logit_data(ticker_list):
    raw_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sess = get_session()
    
    for i, ticker in enumerate(ticker_list):
        try:
            status_text.text(f"Analizando {ticker} ({i+1}/{len(ticker_list)})...")
            t = yf.Ticker(ticker, session=sess)
            info = t.info
            
            if info and 'sector' in info:
                raw_data.append({
                    "Ticker": ticker,
                    "Sector": info.get('sector', 'Other'),
                    "MarketCap": info.get('marketCap', np.nan),
                    "PE_Fwd": info.get('forwardPE', np.nan),
                    "Growth": info.get('revenueGrowth', np.nan),
                    "ROE": info.get('returnOnEquity', np.nan),
                    "Margin": info.get('profitMargins', np.nan),
                    "D_E": info.get('debtToEquity', np.nan),
                    "Ebitda_G": info.get('earningsGrowth', np.nan)
                })
            
            # --- TÉCNICA DE EVASIÓN: Sueño aleatorio ---
            if i % 5 == 0:
                time.sleep(random.uniform(1.5, 3.0)) 
            else:
                time.sleep(random.uniform(0.3, 0.7))
                
            progress_bar.progress((i + 1) / len(ticker_list))
            
        except Exception as e:
            continue # Si falla uno, seguimos con el siguiente
    
    if not raw_data: return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    
    # Limpieza y Conversión
    cols_num = ['MarketCap', 'PE_Fwd', 'Growth', 'ROE', 'Margin', 'D_E', 'Ebitda_G']
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['PE_Fwd', 'Growth', 'ROE'], how='all')
    if df.empty: return df

    # --- Lógica Logit (igual a la anterior) ---
    df['Log_MC'] = np.log(df['MarketCap'].replace(0, np.nan))
    for col in ['PE_Fwd', 'Growth', 'ROE', 'Margin', 'Ebitda_G']:
        if col in df.columns:
            df[f'z_{col}'] = df.groupby('Sector')[col].transform(
                lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0
            )

    df['logit_z'] = (0.5 + (df.get('z_Growth', 0) * 1.5) + (df.get('z_Ebitda_G', 0) * 1.0) +
                     (df.get('z_ROE', 0) * 1.2) + (df.get('z_Margin', 0) * 0.8) +
                     (df['Log_MC'].fillna(df['Log_MC'].mean()) * 0.25) - 
                     (df.get('z_PE_Fwd', 0) * 1.8) - (df.get('D_E', 100)/100 * 0.7))
    
    df['Prob_Ganga'] = 1 / (1 + np.exp(-df['logit_z'].astype(float)))
    
    def semaforo(p):
        if p >= 0.90: return "💎 7) GANGA"
        if p >= 0.80: return "🔥 6) MUY BARATA"
        if p >= 0.70: return "✅ 5) BARATA"
        if p >= 0.50: return "➖ 4) REGULAR"
        if p >= 0.30: return "⚠️ 3) CARA"
        if p >= 0.15: return "🚨 2) MUY CARA"
        return "❌ 1) EVITAR"

    df['Estatus'] = df['Prob_Ganga'].apply(semaforo)
    return df.sort_values("Prob_Ganga", ascending=False)

# --- UI ---
st.title("🏛️ Radar de Valor Logit (Anti-Bloqueo)")

if st.button("🚀 Iniciar Escaneo"):
    with st.spinner("Conectando con servidores financieros..."):
        df_final = get_logit_data(TICKERS)
        
        if df_final.empty:
            st.error("Yahoo Finance ha bloqueado temporalmente la IP. Espera 15 min o reduce la lista de tickers.")
        else:
            st.success(f"¡Éxito! Se analizaron {len(df_final)} activos.")
            st.dataframe(df_final[['Ticker', 'Sector', 'Estatus', 'Prob_Ganga', 'PE_Fwd', 'Growth', 'ROE']], use_container_width=True)
