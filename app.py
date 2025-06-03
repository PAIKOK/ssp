import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config with responsive settings
st.set_page_config(
    page_title="Advanced Stock Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive CSS with media queries
st.markdown("""
<style>
    /* Base styles */
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8fff8;
    }
    .warning-box {
    border: 2px solid #000000;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    background-color: #2d2d2d;
    color: #ffffff;
}
    .info-box {
        border: 2px solid #000000;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #e3f2fd;
    }
    
    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .stMetric {
            padding: 0.5rem !important;
        }
        .stMetric > div {
            font-size: 0.9rem !important;
        }
    }
    
    @media (max-width: 992px) {
        .stMetric {
            padding: 0.3rem !important;
        }
        .stMetric > div {
            font-size: 0.8rem !important;
        }
        .stPlotlyChart {
            height: 400px !important;
        }
    }
    
    @media (max-width: 768px) {
        .stMetric {
            padding: 0.2rem !important;
        }
        .stMetric > div {
            font-size: 0.7rem !important;
        }
        .stPlotlyChart {
            height: 350px !important;
        }
        .stDataFrame {
            font-size: 0.8rem !important;
        }
    }
    
    /* Make sidebar responsive */
    @media (max-width: 576px) {
        .stSidebar {
            width: 100% !important;
            padding: 1rem !important;
        }
        .stButton>button {
            width: 100% !important;
        }
    }
    
    /* Ensure proper spacing on mobile */
    .stApp {
        padding: 1rem;
    }
    
    /* Responsive columns */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    /* Plotly chart responsiveness */
    .js-plotly-plot .plotly {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Stock symbol mapping for different markets
STOCK_SYMBOLS = {
    'US Stocks': {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corp.',
        'TSLA': 'Tesla Inc.',
        'AMZN': 'Amazon.com Inc.',
        'NVDA': 'NVIDIA Corp.',
        'META': 'Meta Platforms Inc.',
        'NFLX': 'Netflix Inc.',
        'SPY': 'SPDR S&P 500 ETF',
        'QQQ': 'Invesco QQQ Trust',
        'VOO': 'Vanguard S&P 500 ETF',
        'BRK-B': 'Berkshire Hathaway',
        'JPM': 'JPMorgan Chase',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc.',
        'PG': 'Procter & Gamble',
        'UNH': 'UnitedHealth Group',
        'HD': 'Home Depot',
        'MA': 'Mastercard Inc.',
        'DIS': 'Walt Disney Co.',
        'PYPL': 'PayPal Holdings',
        'ADBE': 'Adobe Inc.',
        'CRM': 'Salesforce Inc.',
        'CMCSA': 'Comcast Corp.'
    },
    'Indian Stocks (NSE)': {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'HINDUNILVR.NS': 'Hindustan Unilever',
        'ICICIBANK.NS': 'ICICI Bank',
        'KOTAKBANK.NS': 'Kotak Mahindra Bank',
        'BHARTIARTL.NS': 'Bharti Airtel',
        'ITC.NS': 'ITC Limited',
        'SBIN.NS': 'State Bank of India',
        'LT.NS': 'Larsen & Toubro',
        'ASIANPAINT.NS': 'Asian Paints',
        'MARUTI.NS': 'Maruti Suzuki',
        'HCLTECH.NS': 'HCL Technologies',
        'WIPRO.NS': 'Wipro Limited'
    },
    'Indian Stocks (BSE)': {
        'RELIANCE.BO': 'Reliance Industries',
        'TCS.BO': 'Tata Consultancy Services',
        'HDFCBANK.BO': 'HDFC Bank',
        'INFY.BO': 'Infosys',
        'HINDUNILVR.BO': 'Hindustan Unilever'
    }
}

def validate_and_format_symbol(symbol):
    """Validate and format stock symbol with suggestions"""
    symbol = symbol.upper().strip()
    
    # Check if symbol exists in our predefined lists
    for market, stocks in STOCK_SYMBOLS.items():
        if symbol in stocks:
            return symbol, None
        # Check if it's a base symbol that needs market suffix
        for stock_symbol, name in stocks.items():
            if symbol == stock_symbol.split('.')[0]:
                return stock_symbol, f"Found {name} - using {stock_symbol}"
    
    # Common symbol corrections
    corrections = {
        'RELIANCE': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'HDFC': 'HDFCBANK.NS',
        'INFY': 'INFY.NS',
        'INFOSYS': 'INFY.NS',
        'WIPRO': 'WIPRO.NS',
        'MARUTI': 'MARUTI.NS',
        'AIRTEL': 'BHARTIARTL.NS',
        'SBI': 'SBIN.NS',
        'ITC': 'ITC.NS'
    }
    
    if symbol in corrections:
        return corrections[symbol], f"Corrected to {corrections[symbol]}"
    
    # If no correction found, return as is
    return symbol, None

@st.cache_data
def get_stock_data(symbol, period="2y"):
    """Fetch stock data from Yahoo Finance with enhanced error handling"""
    try:
        # Validate and format symbol
        formatted_symbol, correction_msg = validate_and_format_symbol(symbol)
        
        if correction_msg:
            st.info(f"ℹ️ {correction_msg}")
        
        stock = yf.Ticker(formatted_symbol)
        
        # Try to get basic info first to validate symbol
        try:
            info = stock.info
            if not info or 'symbol' not in info:
                # If info is empty, the symbol might be invalid
                raise ValueError(f"Invalid symbol: {formatted_symbol}")
        except Exception as info_error:
            st.warning(f"⚠️ Could not fetch company info for {formatted_symbol}")
            info = {}
        
        # Get historical data
        data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data available for symbol: {formatted_symbol}")
        
        # Check if we have sufficient data
        if len(data) < 100:
            st.warning(f"⚠️ Limited data available for {formatted_symbol} ({len(data)} days). Consider using a longer period.")
        
        return data, info, formatted_symbol
        
    except Exception as e:
        error_msg = str(e)
        if "No data found" in error_msg or "Invalid symbol" in error_msg:
            st.error(f"❌ Symbol '{symbol}' not found. Please check the symbol format.")
            st.markdown("""
            <div class="info-box">
                <h4>💡 Symbol Format Tips:</h4>
                <ul>
                    <li><strong>Indian Stocks:</strong> Add .NS for NSE or .BO for BSE (e.g., RELIANCE.NS, TCS.NS)</li>
                    <li><strong>US Stocks:</strong> Use standard ticker (e.g., AAPL, GOOGL, MSFT)</li>
                    <li><strong>Popular Indian stocks:</strong> Try RELIANCE.NS, TCS.NS, HDFCBANK.NS</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"❌ Error fetching data for {symbol}: {error_msg}")
        return None, None, symbol

def add_technical_indicators(df):
    """Enhanced technical indicators with more options"""
    if df is None or df.empty:
        return df
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price-based indicators
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    
    # Support and Resistance levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    return df

def prepare_data_advanced(df):
    """Enhanced data preparation with feature selection"""
    if df is None or df.empty:
        return None, None
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    if len(df) < 100:  # Need more data for advanced features
        return None, None
    
    # Create multiple target variables
    df['Target_1d'] = df['Close'].shift(-1)  # Next day
    df['Target_5d'] = df['Close'].shift(-5)  # 5 days ahead
    df['Target_direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df = df.dropna()
    
    # Enhanced feature selection
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_200',
        'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
        'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'Volume_MA', 'Volume_ratio', 'High_Low_Pct',
        'Price_Change', 'Price_Change_5d', 'Volatility',
        'Resistance', 'Support'
    ]
    
    # Check available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features]
    y = df['Target_1d']
    
    return X, y

def train_ensemble_model(X, y):
    """Train multiple models and create ensemble"""
    if X is None or y is None or len(X) == 0:
        return None, None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        trained_models[name] = {
            'model': model,
            'metrics': {'MSE': mse, 'MAE': mae, 'R2': r2, 'RMSE': np.sqrt(mse)},
            'predictions': y_pred
        }
        predictions[name] = y_pred
    
    # Create ensemble prediction (average)
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    ensemble_metrics = {
        'MSE': ensemble_mse,
        'MAE': ensemble_mae,
        'R2': ensemble_r2,
        'RMSE': np.sqrt(ensemble_mse)
    }
    
    return trained_models, scaler, ensemble_metrics

def make_ensemble_prediction(models, scaler, latest_data):
    """Make ensemble prediction"""
    if not models or scaler is None or latest_data is None:
        return None, None
    
    try:
        # Get the latest row of features
        latest_features = latest_data.iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        predictions = []
        for name, model_info in models.items():
            pred = model_info['model'].predict(latest_features_scaled)[0]
            predictions.append(pred)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(predictions)
        
        # Individual predictions
        individual_preds = {name: pred for name, pred in zip(models.keys(), predictions)}
        
        return ensemble_pred, individual_preds
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def plot_advanced_chart(df, symbol):
    """Create advanced candlestick chart with indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    # Add moving averages
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'],
            mode='lines', name='MA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_50'],
            mode='lines', name='MA 50',
            line=dict(color='blue', width=1)
        ))
    
    # Bollinger Bands
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            mode='lines', name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            mode='lines', name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
        ))
    
    fig.update_layout(
        title=f'{symbol} Advanced Price Chart',
        yaxis_title='Price (₹)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),  # Reduce margins for better mobile display
        autosize=True  # Make chart responsive
    )
    
    return fig

def save_model(models, scaler, symbol):
    """Save trained models to disk"""
    try:
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save individual models
        for name, model_info in models.items():
            model_filename = f'models/{symbol}_{name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(model_info['model'], model_filename)
        
        # Save scaler
        scaler_filename = f'models/{symbol}_scaler.pkl'
        joblib.dump(scaler, scaler_filename)
        
        return True
    except Exception as e:
        st.error(f"Error saving models: {str(e)}")
        return False

def load_model(symbol, model_name):
    """Load trained model from disk"""
    try:
        model_filename = f'models/{symbol}_{model_name.lower().replace(" ", "_")}_model.pkl'
        scaler_filename = f'models/{symbol}_scaler.pkl'
        
        if os.path.exists(model_filename) and os.path.exists(scaler_filename):
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)
            return model, scaler
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def format_currency(value):
    """Format currency values in Indian Rupees"""
    if isinstance(value, (int, float)):
        return f"₹{value:,.2f}"
    return value

def main():
    # Header with responsive adjustments
    st.title("🚀 Advanced Stock Market Prediction System")
    st.markdown("### AI-Powered Stock Analysis & Prediction Platform")
    
    # Sidebar with responsive settings
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Market selection
        market_choice = st.selectbox(
            "Select Market", 
            list(STOCK_SYMBOLS.keys()) + ["Custom Symbol"]
        )
        
        # Stock selection based on market
        if market_choice != "Custom Symbol":
            symbol = st.selectbox(
                f"Select {market_choice} Symbol", 
                list(STOCK_SYMBOLS[market_choice].keys())
            )
            st.info(f"📊 {STOCK_SYMBOLS[market_choice][symbol]}")
        else:
            # Custom symbol input with better guidance
            st.markdown("### Enter Custom Symbol")
            custom_symbol = st.text_input(
                "Stock Symbol:", 
                placeholder="e.g., RELIANCE.NS, AAPL, GOOGL"
            ).upper()
            
            if custom_symbol:
                symbol = custom_symbol
                st.markdown("""
                **Symbol Format Guide:**
                - **Indian NSE:** Add .NS (e.g., RELIANCE.NS)
                - **Indian BSE:** Add .BO (e.g., RELIANCE.BO)  
                - **US Stocks:** No suffix needed (e.g., AAPL)
                """)
            else:
                symbol = "RELIANCE.NS"  # Default to Indian stock
        
        # Time period
        period_options = {'6 Months': '6mo', '1 Year': '1y', '2 Years': '2y', '5 Years': '5y', '10 Years': '10y'}
        period_choice = st.selectbox("Select Time Period", list(period_options.keys()), index=2)
        period = period_options[period_choice]
        
        # Model options
        st.subheader("🤖 Model Settings")
        use_ensemble = st.checkbox("Use Ensemble Prediction", value=True)
        save_models = st.checkbox("Save Trained Models", value=False)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            show_technical = st.checkbox("Show Technical Indicators", value=True)
            show_volume = st.checkbox("Show Volume Analysis", value=True)
            confidence_interval = st.slider("Prediction Confidence %", 80, 99, 95)
        
        # Analyze button with full width on mobile
        if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
            st.session_state.analyze_clicked = True
        else:
            st.session_state.analyze_clicked = False
    
    # Main content
    if st.session_state.get('analyze_clicked', False):
        # Fetch data
        with st.spinner(f"📊 Fetching data for {symbol}..."):
            result = get_stock_data(symbol, period)
            
            if result[0] is not None:
                data, stock_info, formatted_symbol = result
                
                # Stock Information Section with responsive columns
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"📈 {formatted_symbol} Analysis Dashboard")
                    
                    # Key metrics with responsive layout
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                    
                    # Display metrics in responsive cards
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Current Price", format_currency(current_price), 
                                f"{format_currency(price_change)} ({price_change_pct:.2f}%)")
                    with metric_cols[1]:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                    with metric_cols[2]:
                        st.metric("52W High", format_currency(data['High'].max()))
                    with metric_cols[3]:
                        st.metric("52W Low", format_currency(data['Low'].min()))
                
                with col2:
                    if stock_info:
                        st.subheader("ℹ️ Company Info")
                        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                        market_cap = stock_info.get('marketCap', 0)
                        if market_cap > 0:
                            st.write(f"**Market Cap:** {format_currency(market_cap)}")
                        pe_ratio = stock_info.get('forwardPE', stock_info.get('trailingPE', 'N/A'))
                        st.write(f"**P/E Ratio:** {pe_ratio}")
                    else:
                        st.info("Company information not available")
                
                # Advanced chart with responsive settings
                data_with_indicators = add_technical_indicators(data.copy())
                fig_advanced = plot_advanced_chart(data_with_indicators, formatted_symbol)
                st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Model Training and Prediction
                st.subheader("🎯 AI Prediction Engine")
                
                with st.spinner("🧠 Training advanced ML models..."):
                    X, y = prepare_data_advanced(data.copy())
                    
                    if X is not None and y is not None:
                        if use_ensemble:
                            models, scaler, ensemble_metrics = train_ensemble_model(X, y)
                            
                            if models is not None:
                                # Make ensemble prediction
                                ensemble_pred, individual_preds = make_ensemble_prediction(models, scaler, X)
                                
                                if ensemble_pred is not None:
                                    # Prediction results with responsive layout
                                    st.markdown("""
                                    <div class="prediction-box">
                                        <h3>🎯 AI Prediction Results</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    pred_change = ensemble_pred - current_price
                                    pred_change_pct = (pred_change / current_price) * 100
                                    
                                    pred_cols = st.columns(3)
                                    with pred_cols[0]:
                                        st.metric("🔮 Predicted Next Day Price", format_currency(ensemble_pred))
                                    with pred_cols[1]:
                                        st.metric("📊 Expected Change", 
                                                format_currency(pred_change), 
                                                f"{pred_change_pct:.2f}%")
                                    with pred_cols[2]:
                                        direction = "📈 BULLISH" if pred_change > 0 else "📉 BEARISH"
                                        st.metric("🎯 Signal", direction)
                                    
                                    # Individual model predictions
                                    if individual_preds:
                                        st.subheader("🤖 Individual Model Predictions")
                                        pred_df = pd.DataFrame({
                                            'Model': list(individual_preds.keys()),
                                            'Prediction': [format_currency(pred) for pred in individual_preds.values()],
                                            'Change': [format_currency(pred - current_price) for pred in individual_preds.values()]
                                        })
                                        st.dataframe(pred_df, use_container_width=True, height=150)
                                    
                                    # Model performance with responsive layout
                                    st.subheader("📊 Model Performance Metrics")
                                    perf_cols = st.columns(4)
                                    with perf_cols[0]:
                                        st.metric("R² Score", f"{ensemble_metrics['R2']:.4f}")
                                    with perf_cols[1]:
                                        st.metric("RMSE", format_currency(ensemble_metrics['RMSE']))
                                    with perf_cols[2]:
                                        st.metric("MAE", format_currency(ensemble_metrics['MAE']))
                                    with perf_cols[3]:
                                        accuracy = max(0, min(100, ensemble_metrics['R2'] * 100))
                                        st.metric("Accuracy", f"{accuracy:.1f}%")
                                    
                                    # Save models if requested
                                    if save_models:
                                        if save_model(models, scaler, formatted_symbol):
                                            st.success("✅ Models saved successfully!")
                                        else:
                                            st.error("❌ Failed to save models")
                                    
                                else:
                                    st.error("❌ Failed to make prediction")
                            else:
                                st.error("❌ Failed to train models")
                        else:
                            st.info("Please select 'Use Ensemble Prediction' for advanced analysis")
                    else:
                        st.error("❌ Insufficient data for advanced prediction")
                
                # Technical Analysis Section with responsive layout
                if show_technical and 'RSI' in data_with_indicators.columns:
                    st.subheader("📊 Technical Analysis")
                    
                    tech_cols = st.columns(2)
                    
                    with tech_cols[0]:
                        # RSI Chart
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple')
                        ))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(
                            title="Relative Strength Index (RSI)", 
                            yaxis_title="RSI", 
                            height=300,
                            margin=dict(l=20, r=20, t=60, b=20)
                        )
                        st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with tech_cols[1]:
                        # MACD Chart
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue')
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=data_with_indicators.index,
                            y=data_with_indicators['MACD_signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red')
                        ))
                        fig_macd.update_layout(
                            title="MACD", 
                            yaxis_title="MACD", 
                            height=300,
                            margin=dict(l=20, r=20, t=60, b=20)
                        )
                        st.plotly_chart(fig_macd, use_container_width=True)
            else:
                st.error("❌ Unable to fetch stock data. Please check the symbol and try again.")
    
    
    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This application is for educational and research purposes only. The predictions generated by this AI system 
        should not be considered as financial advice. Stock market investments carry inherent risks, and past performance 
        does not guarantee future results. Always conduct your own research and consult with qualified financial advisors 
        before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()