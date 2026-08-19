import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import swisseph as swe
import datetime
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Astrological Setup & Constants
# ---------------------------------------------------------
try:
    # Set swisseph to use Lahiri Ayanamsha for Vedic Sidereal calculations
    swe.set_sid_mode(swe.SIDM_LAHIRI)
except Exception as e:
    st.error(f"Error initializing PySwissEph: {e}")

PLANET_EMOJIS = {
    "Sun": "Sun ☀️",
    "Moon": "Moon 🌕",
    "Mars": "Mars 🔴",
    "Mercury": "Mercury 🟢",
    "Jupiter": "Jupiter 🟠",
    "Venus": "Venus ⚪",
    "Saturn": "Saturn 🪐",
    "Rahu": "Rahu 🐉",
    "Ketu": "Ketu 🐍"
}

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.TRUE_NODE,
    "Ketu": "Ketu"
}

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Moola", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

RASHIS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

# ---------------------------------------------------------
# Core Astrological & Astronomical Helper Functions
# ---------------------------------------------------------
def get_julday(date_obj, time_obj, tz_str):
    """Convert standard date and time to Julian Day in UTC."""
    if isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.date()
    local_tz = pytz.timezone(tz_str)
    dt_local = datetime.datetime.combine(date_obj, time_obj)
    dt_aware = local_tz.localize(dt_local)
    dt_utc = dt_aware.astimezone(pytz.utc)
    utc_hour = dt_utc.hour + (dt_utc.minute / 60.0) + (dt_utc.second / 3600.0)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, utc_hour)

def get_planet_info(julday, planet_id):
    """Get planetary longitude and speed."""
    if planet_id == "Ketu":
        # Ketu is exactly 180 degrees opposite to Rahu
        res, _ = swe.calc_ut(julday, PLANETS["Rahu"], swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = (res[0] + 180.0) % 360.0
        speed = res[3] # Speed is same as Rahu
    else:
        res, _ = swe.calc_ut(julday, planet_id, swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)
        lon = res[0]
        speed = res[3]
    return lon, speed

def get_nakshatra(lon):
    """Determine the Nakshatra, Pada, and index from longitude."""
    nak_len = 360.0 / 27.0
    nak_index = int(lon / nak_len)
    pada = int((lon % nak_len) / (nak_len / 4.0)) + 1
    return NAKSHATRAS[nak_index], pada, nak_index

def get_rashi(lon):
    """Determine the Rashi (Zodiac Sign) from longitude."""
    rashi_index = int(lon / 30.0)
    return RASHIS[rashi_index]

def get_panchang(sun_lon, moon_lon):
    """Calculate Tithi, Yoga, and Karana."""
    # Tithi: 12 degree difference between Moon and Sun
    diff = (moon_lon - sun_lon) % 360.0
    tithi = int(diff / 12.0) + 1
    
    # Yoga: sum of longitudes modulo 360 divided by 13.333
    total = (moon_lon + sun_lon) % 360.0
    yoga = int(total / (360.0 / 27.0)) + 1
    
    # Karana: Half of a Tithi (6 degrees)
    karana = int(diff / 6.0) + 1
    return tithi, yoga, karana

def angular_distance(lon1, lon2):
    """Calculate the shortest angular distance between two longitudes."""
    diff = abs(lon1 - lon2) % 360
    return min(diff, 360 - diff)

def is_aspect(lon1, lon2, target_angle, orb):
    """Check if two longitudes are in aspect within a given orb."""
    dist = angular_distance(lon1, lon2)
    return abs(dist - target_angle) <= orb

# ---------------------------------------------------------
# Financial Data Engine
# ---------------------------------------------------------
@st.cache_data
def fetch_market_data(ticker, start_date, end_date):
    """Fetch financial data from Yahoo Finance and calculate metrics."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if returning from yf in recent versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Select required columns just to be safe
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Calculate Returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Forward returns for backtesting
        df['T+1_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['T+3_Return'] = df['Close'].shift(-3) / df['Close'] - 1
        df['T+5_Return'] = df['Close'].shift(-5) / df['Close'] - 1
        
        # Forward Excursions (MFE/MAE) for T+1 holding period
        df['T+1_High'] = df['High'].shift(-1)
        df['T+1_Low'] = df['Low'].shift(-1)
        df['MFE'] = (df['T+1_High'] - df['Close']) / df['Close']
        df['MAE'] = (df['T+1_Low'] - df['Close']) / df['Close']
        
        # Volatility / ATR Calculation (True Range and 14-day SMA of TR)
        df['TR'] = np.maximum((df['High'] - df['Low']), 
                   np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                              abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        return df
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Main Streamlit App UI
# ---------------------------------------------------------
st.set_page_config(page_title="AstroQuant Engine", layout="wide", initial_sidebar_state="expanded")

# Sidebar Configuration
st.sidebar.title("AstroQuant Config ⚙️")
st.sidebar.markdown("Combine quantitative finance with Vedic astrology.")

st.sidebar.subheader("Global Data Feed")
NSE_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "INFOSYS": "INFY.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "AXIS BANK": "AXISBANK.NS",
    "KOTAK MAHINDRA BANK": "KOTAKBANK.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "HCL TECHNOLOGIES": "HCLTECH.NS",
    "MAHINDRA & MAHINDRA": "M&M.NS",
    "WIPRO": "WIPRO.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS",
    "TITAN": "TITAN.NS",
    "NESTLE INDIA": "NESTLEIND.NS",
    "JSW STEEL": "JSWSTEEL.NS",
    "POWER GRID": "POWERGRID.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "NTPC": "NTPC.NS",
    "GRASIM": "GRASIM.NS",
    "INDUSIND BANK": "INDUSINDBK.NS",
    "HINDALCO": "HINDALCO.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "DR. REDDY'S": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    "ONGC": "ONGC.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "COAL INDIA": "COALINDIA.NS",
    "TATA CONSUMER": "TATACONSUM.NS",
    "APOLLO HOSPITALS": "APOLLOHOSP.NS",
    "UPL": "UPL.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "DIVI'S LABS": "DIVISLAB.NS",
    "EICHER MOTORS": "EICHERMOT.NS",
    "LTIMINDTREE": "LTIM.NS",
    "CUSTOM (Enter manually)": "CUSTOM"
}

selected_asset = st.sidebar.selectbox("Search / Select Asset", list(NSE_SYMBOLS.keys()), index=0)
if selected_asset == "CUSTOM (Enter manually)":
    ticker = st.sidebar.text_input("Enter Yahoo Finance Ticker", value="^NSEI", help="For NSE stocks, append .NS (e.g. RELIANCE.NS, TCS.NS)")
else:
    ticker = NSE_SYMBOLS[selected_asset]
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

st.sidebar.subheader("Time & Location Settings")
tz_choice = st.sidebar.selectbox("Timezone", ["Asia/Kolkata", "UTC", "America/New_York"], index=0)
calc_time = st.sidebar.time_input("Calculation Time", value=datetime.time(9, 15))

st.sidebar.subheader("Astrology Engine Settings")
ayanamsha_choice = st.sidebar.selectbox("Ayanamsha System", ["Lahiri (Chitra Paksha)", "Raman", "Krishnamurti (KP)", "Fagan/Bradley"])
node_choice = st.sidebar.radio("Lunar Nodes", ["True Node", "Mean Node"])

# Apply Engine Settings Dynamically
if ayanamsha_choice == "Lahiri (Chitra Paksha)":
    swe.set_sid_mode(swe.SIDM_LAHIRI)
elif ayanamsha_choice == "Raman":
    swe.set_sid_mode(swe.SIDM_RAMAN)
elif ayanamsha_choice == "Krishnamurti (KP)":
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)
elif ayanamsha_choice == "Fagan/Bradley":
    swe.set_sid_mode(swe.SIDM_FAGAN_BRADLEY)

PLANETS["Rahu"] = swe.TRUE_NODE if node_choice == "True Node" else swe.MEAN_NODE

st.sidebar.markdown("---")
mode = st.sidebar.radio("Navigation", ["1. Backtest Strategy", "2. Historical Inspector", "3. Live Dashboard"])

# Load Market Data
df = fetch_market_data(ticker, start_date, end_date)

if df.empty:
    st.warning("No market data found for the selected ticker and dates. Please check the ticker symbol.")
    st.stop()

# ---------------------------------------------------------
# Mode 1: Backtest Strategy
# ---------------------------------------------------------
if mode == "1. Backtest Strategy":
    st.title("Astrological Strategy Backtester 📈")
    
    st.sidebar.subheader("Strategy Parameters")
    strategy_mode = st.sidebar.radio("Strategy Type", ["Nakshatra Backtest", "Planetary Conjunction"])
    
    if strategy_mode == "Nakshatra Backtest":
        target_nak = st.sidebar.selectbox("Select Target Nakshatra (Moon Transit)", NAKSHATRAS)
    else:
        p1 = st.sidebar.selectbox("Planet 1", list(PLANETS.keys()), format_func=lambda x: PLANET_EMOJIS[x], index=0)
        p2 = st.sidebar.selectbox("Planet 2", list(PLANETS.keys()), format_func=lambda x: PLANET_EMOJIS[x], index=4)
        target_angle = st.sidebar.selectbox("Aspect Type (Degrees)", [0, 90, 180], format_func=lambda x: {0: "Conjunction (0°)", 90: "Square (90°)", 180: "Opposition (180°)"}[x])
        orb = st.sidebar.slider("Orb (Degrees)", 0.0, 10.0, 3.0, 0.1)
    
    st.markdown(f"### Backtesting Results: **{ticker}** ({start_date} to {end_date})")
    
    signals = []
    
    # Generate signals via astronomical positions
    with st.spinner('Calculating Ephemeris & Generating Signals...'):
        for date, row in df.iterrows():
            jd = get_julday(date, calc_time, tz_choice)
            is_signal = False
            
            if strategy_mode == "Nakshatra Backtest":
                moon_lon, _ = get_planet_info(jd, PLANETS["Moon"])
                nak_name, _, _ = get_nakshatra(moon_lon)
                if nak_name == target_nak:
                    is_signal = True
            else:
                lon1, _ = get_planet_info(jd, PLANETS[p1])
                lon2, _ = get_planet_info(jd, PLANETS[p2])
                if is_aspect(lon1, lon2, target_angle, orb):
                    is_signal = True
            
            signals.append(is_signal)
            
    df['Signal'] = signals
    
    sig_df = df[df['Signal']].copy()
    
    if len(sig_df) > 0:
        total_signals = len(sig_df)
        # Advanced Metrics Calculations
        profitable_trades = (sig_df['T+1_Return'] > 0).sum()
        win_rate = (profitable_trades / total_signals) * 100
        
        gross_profit = sig_df.loc[sig_df['T+1_Return'] > 0, 'T+1_Return'].sum()
        gross_loss = abs(sig_df.loc[sig_df['T+1_Return'] < 0, 'T+1_Return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        avg_ret_1 = sig_df['T+1_Return'].mean() * 100
        avg_mfe = sig_df['MFE'].mean() * 100
        avg_mae = sig_df['MAE'].mean() * 100
        
        # Cumulative Equity Calculation
        df['Strat_Return'] = np.where(df['Signal'], df['Daily_Return'].shift(-1), 0)
        df['Equity_Strat'] = (1 + df['Strat_Return'].fillna(0)).cumprod()
        df['Equity_BH'] = (1 + df['Daily_Return'].fillna(0)).cumprod()
        
        cum_pnl = (df['Equity_Strat'].iloc[-1] - 1) * 100
        
        # Max Drawdown Calculation
        df['Peak'] = df['Equity_Strat'].cummax()
        df['Drawdown'] = (df['Equity_Strat'] - df['Peak']) / df['Peak']
        max_dd = df['Drawdown'].min() * 100
        
        # Display Metrics UI
        st.markdown("#### Backtest Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", total_signals)
        m2.metric("Profitable Trades", f"{profitable_trades} ({win_rate:.1f}%)")
        m3.metric("Profit Factor", f"{profit_factor:.2f}")
        m4.metric("Cumulative P&L", f"{cum_pnl:.2f}%")
        
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Avg Return (T+1)", f"{avg_ret_1:.2f}%")
        m6.metric("Max Drawdown", f"{max_dd:.2f}%")
        m7.metric("Avg Favorable Excursion (MFE)", f"+{avg_mfe:.2f}%")
        m8.metric("Avg Adverse Excursion (MAE)", f"{avg_mae:.2f}%")
        
        # Cumulative Equity Curve
        st.markdown("#### Cumulative Returns (Signal Strategy vs Buy & Hold)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity_Strat'], mode='lines', name='Strategy Equity', line=dict(color='lime', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Equity_BH'], mode='lines', name='Buy & Hold Equity', line=dict(color='gray', width=1)))
        
        fig.update_layout(height=500, template='plotly_dark', margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # ------------------- Add Trade Log & Planetary Impact -------------------
        st.markdown("---")
        st.subheader("Trade Log & Planetary Impact Analysis")
        
        trade_logs = []
        with st.spinner("Analyzing Planetary Impacts on Trades..."):
            for d, r in sig_df.iterrows():
                jd = get_julday(d, calc_time, tz_choice)
                trade_ret = r['T+1_Return']
                if pd.isna(trade_ret):
                    continue
                    
                outcome = "Bullish" if trade_ret > 0 else "Bearish"
                
                row_data = {
                    "Date": d.date(),
                    "T+1 Return": trade_ret,
                    "Outcome": outcome
                }
                for p_name, p_id in PLANETS.items():
                    lon, _ = get_planet_info(jd, p_id)
                    row_data[p_name] = get_rashi(lon)
                
                trade_logs.append(row_data)
            
        if trade_logs:
            trade_df = pd.DataFrame(trade_logs)
            
            display_df = trade_df.copy()
            display_df['T+1 Return'] = display_df['T+1 Return'].apply(lambda x: f"{x*100:.2f}%")
            
            st.markdown(f"**Total Executed Trades:** {len(trade_df)}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Planetary Impact Summary")
            st.write("Number of Bullish vs Bearish trades categorized by each planet's Zodiac Sign (Rashi).")
            
            impact_tabs = st.tabs([PLANET_EMOJIS[p] for p in PLANETS.keys()])
            for i, p_name in enumerate(PLANETS.keys()):
                with impact_tabs[i]:
                    dist = trade_df.groupby([p_name, 'Outcome']).size().unstack(fill_value=0)
                    if "Bullish" not in dist.columns:
                        dist["Bullish"] = 0
                    if "Bearish" not in dist.columns:
                        dist["Bearish"] = 0
                    dist["Total Trades"] = dist["Bullish"] + dist["Bearish"]
                    dist["Win Rate"] = (dist["Bullish"] / dist["Total Trades"] * 100).round(2).astype(str) + "%"
                    st.dataframe(dist.sort_values("Total Trades", ascending=False), use_container_width=True)
            
            # --- New Logic based on user request ---
            st.markdown("---")
            st.markdown("### 📊 Specific Bullish Percentage Summary")
            if strategy_mode == "Planetary Conjunction":
                summary_data = {
                    "Configuration": [f"{PLANET_EMOJIS[p1]} + {PLANET_EMOJIS[p2]} Conjunction/Aspect"],
                    "Total Signals": [total_signals],
                    "Bullish Signals": [len(trade_df[trade_df['Outcome'] == 'Bullish'])],
                    "Bearish Signals": [len(trade_df[trade_df['Outcome'] == 'Bearish'])],
                    "Bullish Percentage": [f"{win_rate:.2f}%"]
                }
                st.table(pd.DataFrame(summary_data))
            elif strategy_mode == "Nakshatra Backtest":
                summary_data = {
                    "Configuration": [f"{PLANET_EMOJIS['Moon']} in {target_nak} Nakshatra"],
                    "Total Signals": [total_signals],
                    "Bullish Signals": [len(trade_df[trade_df['Outcome'] == 'Bullish'])],
                    "Bearish Signals": [len(trade_df[trade_df['Outcome'] == 'Bearish'])],
                    "Bullish Percentage": [f"{win_rate:.2f}%"]
                }
                st.table(pd.DataFrame(summary_data))
        
    else:
        st.info("No signals generated for the given parameters. Try expanding your date range or increasing the orb limit.")

# ---------------------------------------------------------
# Mode 2: Historical Date & Cosmic Inspector
# ---------------------------------------------------------
elif mode == "2. Historical Inspector":
    st.title("Historical Date & Cosmic Inspector 🌌")
    
    st.sidebar.subheader("Inspector Parameters")
    min_d, max_d = df.index.min().date(), df.index.max().date()
    inspect_date = st.sidebar.date_input("Select Historical Date", value=max_d, min_value=min_d, max_value=max_d)
    
    orb_limit = st.sidebar.slider("Aspect Orb Limit (Degrees)", 0.0, 10.0, 3.0, 0.5)
    
    inspect_date_dt = pd.to_datetime(inspect_date)
    
    # Check if selected date is in our trading days
    if inspect_date_dt not in df.index:
        st.warning("Selected date is a non-trading day (e.g., weekend/holiday). Showing price data for the closest prior trading day, but exact astrological data for the selected date.")
        # Find nearest date prior
        past_dates = df.index[df.index <= inspect_date_dt]
        if len(past_dates) > 0:
            closest_date = past_dates[-1]
        else:
            closest_date = df.index[0]
    else:
        closest_date = inspect_date_dt
        
    st.markdown(f"### Cosmic Snapshot for **{inspect_date.strftime('%B %d, %Y')}**")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # 1. Price Summary & Mini Chart
        st.subheader("📊 Price Action Context")
        
        # Extract a window of 11 days (5 before, 1 current, 5 after)
        loc = df.index.get_loc(closest_date)
        start_loc = max(0, loc - 5)
        end_loc = min(len(df), loc + 6)
        window_df = df.iloc[start_loc:end_loc]
        
        row = df.loc[closest_date]
        c1, c2, c3 = st.columns(3)
        c1.metric("Close", f"{row['Close']:.2f}", f"{row['Daily_Return']*100:.2f}%")
        c2.metric("High", f"{row['High']:.2f}")
        c3.metric("ATR (Volatility)", f"{row['ATR']:.2f}")
        
        fig = go.Figure(data=[go.Candlestick(x=window_df.index,
                    open=window_df['Open'], high=window_df['High'],
                    low=window_df['Low'], close=window_df['Close'])])
        
        # Highlight the selected day
        fig.add_vline(x=closest_date, line_width=2, line_dash="dash", line_color="rgba(255, 255, 0, 0.5)")
        fig.update_layout(height=350, template='plotly_dark', margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # 4. Panchang Breakdown
        st.subheader("📜 Daily Panchang")
        jd = get_julday(inspect_date, calc_time, tz_choice)
        sun_lon, _ = get_planet_info(jd, PLANETS["Sun"])
        moon_lon, _ = get_planet_info(jd, PLANETS["Moon"])
        
        tithi, yoga, karana = get_panchang(sun_lon, moon_lon)
        weekday = inspect_date.strftime("%A")
        
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Vara", weekday)
        pc2.metric("Tithi", f"{tithi}/30")
        pc3.metric("Yoga", f"{yoga}/27")
        pc4.metric("Karana", f"{karana}/60")

    with col_right:
        # 2. Planetary Alignment Table
        st.subheader("🪐 Planetary Alignments")
        
        planets_data = []
        for p_name, p_id in PLANETS.items():
            lon, speed = get_planet_info(jd, p_id)
            nak_name, pada, _ = get_nakshatra(lon)
            rashi = get_rashi(lon)
            
            # Direct/Retrograde Status
            status = "Direct ➡️"
            if speed < 0:
                status = "Retrograde ⬅️"
            if p_name in ["Rahu", "Ketu"]:
                status = "Retrograde ⬅️"
            if p_name in ["Sun", "Moon"]:
                status = "Direct ➡️"
                
            # Combustion (Simplified Rule: within 8° of Sun)
            combust = "No"
            if p_name not in ["Sun", "Moon", "Rahu", "Ketu"]:
                if angular_distance(lon, sun_lon) <= 8:
                    combust = "Yes 🔥"
                    
            planets_data.append({
                "Planet": PLANET_EMOJIS[p_name],
                "Longitude": f"{lon:.2f}°",
                "Rashi": rashi,
                "Nakshatra": f"{nak_name} (P{pada})",
                "Status": status,
                "Combust": combust
            })
            
        ptable = pd.DataFrame(planets_data)
        st.dataframe(ptable, use_container_width=True, hide_index=True)
        
        # 3. Active Aspects
        st.subheader("✨ Active Aspects Matrix")
        
        aspects = []
        planet_names = list(PLANETS.keys())
        
        # Extract numerical longitude for calculation
        ptable['Lon_Num'] = ptable['Longitude'].str.replace('°', '').astype(float)
        
        for i in range(len(planet_names)):
            for j in range(i+1, len(planet_names)):
                p1, p2 = planet_names[i], planet_names[j]
                
                # Exclude Rahu/Ketu opposition as it's always 180
                if (p1 == "Rahu" and p2 == "Ketu") or (p1 == "Ketu" and p2 == "Rahu"):
                    continue
                
                lon1 = ptable.loc[ptable['Planet'] == PLANET_EMOJIS[p1], 'Lon_Num'].values[0]
                lon2 = ptable.loc[ptable['Planet'] == PLANET_EMOJIS[p2], 'Lon_Num'].values[0]
                
                dist = angular_distance(lon1, lon2)
                
                if abs(dist - 0) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} ☌ {PLANET_EMOJIS[p2]}** (Conjunct, Dist: {dist:.1f}°)")
                elif abs(dist - 90) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} □ {PLANET_EMOJIS[p2]}** (Square, Dist: {dist:.1f}°)")
                elif abs(dist - 180) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} ☍ {PLANET_EMOJIS[p2]}** (Opposition, Dist: {dist:.1f}°)")
                    
        if len(aspects) > 0:
            for a in aspects:
                st.markdown(f"- {a}")
        else:
            st.info("No major aspects found within the given orb limit.")
            
        # Qualitative Nakshatra Logic (Sample logic)
        moon_nak = ptable.loc[ptable['Planet'] == PLANET_EMOJIS['Moon'], 'Nakshatra'].values[0].split(" ")[0]
        
        # Sample broad categorization of Nakshatras for financial markets
        bullish_naks = ["Pushya", "Rohini", "Uttara", "Shravana", "Dhanishta", "Anuradha", "Punarvasu"]
        bearish_naks = ["Ashlesha", "Jyeshtha", "Moola", "Bharani", "Krittika", "Ardra"]
        
        # Handle matching with first word of Nakshatra
        base_moon_nak = moon_nak
        if base_moon_nak in bullish_naks:
            st.success(f"**Market Sentiment Signal**: Moon in {moon_nak} - Typically **Bullish/Teji** 🐂")
        elif base_moon_nak in bearish_naks:
            st.error(f"**Market Sentiment Signal**: Moon in {moon_nak} - Typically **Bearish/Mandi** 🐻")
        else:
            st.warning(f"**Market Sentiment Signal**: Moon in {moon_nak} - Typically **Volatile/Teevra** ⚖️")

# ---------------------------------------------------------
# Mode 3: Live Dashboard
# ---------------------------------------------------------
elif mode == "3. Live Dashboard":
    st.title("Live Planetary Dashboard 🔴")
    
    st.sidebar.subheader("Live Settings")
    orb_limit = st.sidebar.slider("Live Aspect Orb (Degrees)", 0.0, 10.0, 3.0, 0.5)
    
    now = datetime.datetime.now(pytz.timezone(tz_choice))
    current_date = now.date()
    current_time = now.time()
    
    st.markdown(f"### Live Cosmic Snapshot as of **{now.strftime('%B %d, %Y %I:%M %p %Z')}**")
    
    jd = get_julday(current_date, current_time, tz_choice)
    sun_lon, _ = get_planet_info(jd, PLANETS["Sun"])
    moon_lon, _ = get_planet_info(jd, PLANETS["Moon"])
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🪐 Live Planetary Alignments")
        
        planets_data = []
        for p_name, p_id in PLANETS.items():
            lon, speed = get_planet_info(jd, p_id)
            nak_name, pada, _ = get_nakshatra(lon)
            rashi = get_rashi(lon)
            
            status = "Direct ➡️"
            if speed < 0:
                status = "Retrograde ⬅️"
            if p_name in ["Rahu", "Ketu"]:
                status = "Retrograde ⬅️"
            if p_name in ["Sun", "Moon"]:
                status = "Direct ➡️"
                
            combust = "No"
            if p_name not in ["Sun", "Moon", "Rahu", "Ketu"]:
                if angular_distance(lon, sun_lon) <= 8:
                    combust = "Yes 🔥"
                    
            planets_data.append({
                "Planet": PLANET_EMOJIS[p_name],
                "Longitude": f"{lon:.2f}°",
                "Rashi": rashi,
                "Nakshatra": f"{nak_name} (P{pada})",
                "Status": status,
                "Combust": combust
            })
            
        ptable = pd.DataFrame(planets_data)
        st.dataframe(ptable, use_container_width=True, hide_index=True)
        
    with col_right:
        st.subheader("📜 Current Panchang")
        tithi, yoga, karana = get_panchang(sun_lon, moon_lon)
        weekday = current_date.strftime("%A")
        
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Vara", weekday)
        pc2.metric("Tithi", f"{tithi}/30")
        pc3.metric("Yoga", f"{yoga}/27")
        pc4.metric("Karana", f"{karana}/60")
        
        st.subheader("✨ Active Aspects Matrix")
        aspects = []
        planet_names = list(PLANETS.keys())
        ptable['Lon_Num'] = ptable['Longitude'].str.replace('°', '').astype(float)
        
        for i in range(len(planet_names)):
            for j in range(i+1, len(planet_names)):
                p1, p2 = planet_names[i], planet_names[j]
                if (p1 == "Rahu" and p2 == "Ketu") or (p1 == "Ketu" and p2 == "Rahu"):
                    continue
                lon1 = ptable.loc[ptable['Planet'] == PLANET_EMOJIS[p1], 'Lon_Num'].values[0]
                lon2 = ptable.loc[ptable['Planet'] == PLANET_EMOJIS[p2], 'Lon_Num'].values[0]
                dist = angular_distance(lon1, lon2)
                
                if abs(dist - 0) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} ☌ {PLANET_EMOJIS[p2]}** (Conjunct, Dist: {dist:.1f}°)")
                elif abs(dist - 90) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} □ {PLANET_EMOJIS[p2]}** (Square, Dist: {dist:.1f}°)")
                elif abs(dist - 180) <= orb_limit:
                    aspects.append(f"**{PLANET_EMOJIS[p1]} ☍ {PLANET_EMOJIS[p2]}** (Opposition, Dist: {dist:.1f}°)")
                    
        if len(aspects) > 0:
            for a in aspects:
                st.markdown(f"- {a}")
        else:
            st.info("No major aspects found right now within the given orb limit.")
            
        moon_nak = ptable.loc[ptable['Planet'] == PLANET_EMOJIS['Moon'], 'Nakshatra'].values[0].split(" ")[0]
        bullish_naks = ["Pushya", "Rohini", "Uttara", "Shravana", "Dhanishta", "Anuradha", "Punarvasu"]
        bearish_naks = ["Ashlesha", "Jyeshtha", "Moola", "Bharani", "Krittika", "Ardra"]
        
        base_moon_nak = moon_nak
        if base_moon_nak in bullish_naks:
            st.success(f"**Live Market Sentiment**: Moon in {moon_nak} - Typically **Bullish/Teji** 🐂")
        elif base_moon_nak in bearish_naks:
            st.error(f"**Live Market Sentiment**: Moon in {moon_nak} - Typically **Bearish/Mandi** 🐻")
        else:
            st.warning(f"**Live Market Sentiment**: Moon in {moon_nak} - Typically **Volatile/Teevra** ⚖️")
