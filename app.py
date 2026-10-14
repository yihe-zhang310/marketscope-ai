import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MarketScope AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MarketScope AI")
st.caption(
    "An educational stock analysis and machine-learning forecasting tool. "
    "Predictions are experimental and are not financial advice."
)


# ============================================================
# HISTORICAL STOCK DATA
# ============================================================

@st.cache_data(ttl=900)
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if data is None or data.empty:
        raise ValueError(
            f"No historical data was returned for {symbol}. "
            "Check the ticker symbol and try again."
        )

    required_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing data columns: {missing}")

    data = data[required_columns].copy()
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Close"])

    if len(data) < 60:
        raise ValueError(
            "Not enough historical observations. "
            "Try a longer historical period."
        )

    return data


# ============================================================
# COMPANY NEWS
# ============================================================

@st.cache_data(ttl=300)
def load_news(symbol):
    """
    Retrieve news using the yfinance Search object's .news property.
    Returns articles, error message, query, and diagnostic information.
    """

    symbol = symbol.strip().upper()

    company_names = {
        "NVDA": "NVIDIA",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "GOOG": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "AMD": "AMD",
        "INTC": "Intel",
        "NFLX": "Netflix",
        "AVGO": "Broadcom",
        "ORCL": "Oracle",
        "CRM": "Salesforce",
        "JPM": "JPMorgan",
        "WMT": "Walmart",
        "DIS": "Disney",
        "COST": "Costco",
        "PLTR": "Palantir",
        "SMCI": "Super Micro Computer",
        "TSM": "Taiwan Semiconductor",
        "BABA": "Alibaba",
        "UBER": "Uber",
        "COIN": "Coinbase",
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
        "BA": "Boeing",
        "IBM": "IBM",
        "ADBE": "Adobe",
        "PYPL": "PayPal",
    }

    company = company_names.get(symbol, "")
    query = f"{symbol} {company}".strip()

    diagnostics = {
        "query": query,
        "response_type": "Not received",
        "article_count": 0,
        "first_article_keys": [],
        "first_article_sample": None,
    }

    try:
        search = yf.Search(
            query,
            news_count=20,
            max_results=10
        )

        # IMPORTANT:
        # Read the news property directly instead of treating
        # the Search object as a dictionary.
        articles = search.news

        diagnostics["response_type"] = type(articles).__name__

        if not isinstance(articles, list):
            articles = []

        diagnostics["article_count"] = len(articles)

        if articles and isinstance(articles[0], dict):
            diagnostics["first_article_keys"] = list(
                articles[0].keys()
            )

            diagnostics["first_article_sample"] = {
                key: value
                for key, value in list(articles[0].items())[:8]
            }

        results = []
        seen_links = set()

        for original in articles:
            if not isinstance(original, dict):
                continue

            article = dict(original)

            # Some response formats nest the article fields.
            content = article.get("content")

            if isinstance(content, dict):
                article = {**article, **content}

            title = str(
                article.get("title")
                or article.get("headline")
                or ""
            ).strip()

            # Extract a URL from common response formats.
            link = article.get("link") or ""

            if isinstance(link, dict):
                link = link.get("url", "")

            if not link:
                canonical = article.get("canonicalUrl")

                if isinstance(canonical, dict):
                    link = canonical.get("url", "")
                elif isinstance(canonical, str):
                    link = canonical

            if not link:
                click_url = article.get("clickThroughUrl")

                if isinstance(click_url, dict):
                    link = click_url.get("url", "")
                elif isinstance(click_url, str):
                    link = click_url

            if not title or not isinstance(link, str):
                continue

            if not link.startswith(("https://", "http://")):
                continue

            if link in seen_links:
                continue

            seen_links.add(link)

            # Published date.
            published = (
                article.get("providerPublishTime")
                or article.get("pubDate")
                or article.get("displayTime")
                or article.get("published")
                or "Date unavailable"
            )

            if isinstance(published, (int, float)):
                try:
                    published = datetime.fromtimestamp(
                        published,
                        tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                except (ValueError, OSError, OverflowError):
                    published = "Date unavailable"

            # Publisher.
            provider = article.get("provider")

            if isinstance(provider, dict):
                publisher = (
                    provider.get("displayName")
                    or provider.get("name")
                    or "News source"
                )
            else:
                publisher = (
                    article.get("publisher")
                    or provider
                    or article.get("source")
                    or "News source"
                )

            results.append({
                "title": title,
                "publisher": str(publisher),
                "link": link,
                "published": str(published),
            })

        error = None

        if not results:
            error = (
                "The search returned no usable article links. "
                "Yahoo Finance may have returned no articles, "
                "or the response format may differ from expected."
            )

        return results, error, query, diagnostics

    except Exception as exc:
        return (
            [],
            f"{type(exc).__name__}: {exc}",
            query,
            diagnostics
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(data):
    """
    Construct lagged features from historical data.
    The target is the next observed trading day's return.
    """

    frame = data.copy()

    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float)

    returns = close.pct_change(fill_method=None)

    # Lagged daily returns.
    frame["return_1"] = returns.shift(1)
    frame["return_2"] = returns.shift(2)
    frame["return_3"] = returns.shift(3)
    frame["return_5"] = returns.shift(5)
    frame["return_10"] = returns.shift(10)
    frame["return_20"] = returns.shift(20)

    # Historical momentum.
    frame["momentum_5"] = (
        close.pct_change(5, fill_method=None).shift(1)
    )
    frame["momentum_10"] = (
        close.pct_change(10, fill_method=None).shift(1)
    )
    frame["momentum_20"] = (
        close.pct_change(20, fill_method=None).shift(1)
    )

    # Historical volatility.
    frame["volatility_5"] = (
        returns.rolling(5).std().shift(1)
    )
    frame["volatility_20"] = (
        returns.rolling(20).std().shift(1)
    )

    # Volume-derived features.
    frame["volume_change"] = (
        volume.pct_change(fill_method=None).shift(1)
    )
    frame["volume_change_5"] = (
        volume.pct_change(5, fill_method=None).shift(1)
    )

    # Predict the next observed daily return.
    frame["target"] = returns.shift(-1)

    frame = frame.replace([np.inf, -np.inf], np.nan)

    return frame


# ============================================================
# MODEL TRAINING AND FORECASTING
# ============================================================

def train_and_forecast(data, horizon):

    feature_columns = [
        "return_1",
        "return_2",
        "return_3",
        "return_5",
        "return_10",
        "return_20",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "volatility_5",
        "volatility_20",
        "volume_change",
        "volume_change_5",
    ]

    feature_frame = make_features(data)

    training_data = feature_frame.dropna(
        subset=feature_columns + ["target"]
    )

    if len(training_data) < 40:
        raise ValueError(
            "Not enough complete observations to train the model. "
            "Select a longer historical period."
        )

    X = training_data[feature_columns]
    y = training_data["target"]

    # Chronological split. Do not shuffle time-series data.
    split_index = int(len(training_data) * 0.8)

    if split_index < 20 or len(training_data) - split_index < 5:
        raise ValueError(
            "Not enough observations for the training and testing split."
        )

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    evaluation_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    evaluation_model.fit(X_train, y_train)

    test_predictions = evaluation_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    directional_accuracy = np.mean(
        np.sign(test_predictions) == np.sign(y_test.to_numpy())
    )

    # Retrain using all labeled observations.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Begin recursive forecasting from the available historical data.
    history = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    current_price = float(history["Close"].iloc[-1])
    last_date = history.index[-1]

    forecasts = []

    for step in range(1, horizon + 1):

        feature_frame = make_features(history)

        latest_features = feature_frame[
            feature_columns
        ].iloc[[-1]]

        if latest_features.isna().to_numpy().any():
            raise ValueError(
                "The latest row contains missing features. "
                "Try a longer historical period."
            )

        predicted_return = float(
            model.predict(latest_features)[0]
        )

        if not np.isfinite(predicted_return):
            raise ValueError(
                "The model generated an invalid return prediction."
            )

        # Prevent extreme numerical predictions from cascading.
        predicted_return = float(
            np.clip(predicted_return, -0.5, 0.5)
        )

        predicted_price = current_price * (1 + predicted_return)

        if not np.isfinite(predicted_price) or predicted_price <= 0:
            raise ValueError(
                "The model generated an invalid predicted price."
            )

        # Business-day approximation; exchange holidays are not excluded.
        future_date = last_date + pd.offsets.BDay(1)

        forecasts.append({
            "Date": future_date,
            "Predicted Price": predicted_price,
            "Predicted Return": predicted_return,
        })

        # Volume is carried forward as a placeholder for future dates.
        last_volume = float(history["Volume"].iloc[-1])

        new_row = pd.DataFrame(
            {
                "Open": [predicted_price],
                "High": [predicted_price],
                "Low": [predicted_price],
                "Close": [predicted_price],
                "Volume": [last_volume],
            },
            index=[future_date]
        )

        history = pd.concat([history, new_row])

        current_price = predicted_price
        last_date = future_date

    forecast_df = pd.DataFrame(forecasts).set_index("Date")

    return (
        forecast_df,
        mae,
        directional_accuracy,
        len(training_data)
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="NVDA",
    help="Examples: NVDA, AAPL, MSFT, GOOGL, TSLA"
).strip().upper()

period_options = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
    "10 years": "10y",
}

period_label = st.sidebar.selectbox(
    "Historical period",
    options=list(period_options.keys()),
    index=2
)

period = period_options[period_label]

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    options=[5, 10, 20],
    index=1,
    format_func=lambda days: f"{days} trading days"
)

show_diagnostics = st.sidebar.checkbox(
    "Show news diagnostics",
    value=True
)

if st.sidebar.button("Refresh data", type="primary"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# MAIN APP
# ============================================================

if not ticker:
    st.warning("Enter a stock ticker in the sidebar.")
    st.stop()

try:

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    with st.spinner(f"Loading historical data for {ticker}..."):
        data = load_data(ticker, period)

    latest_price = float(data["Close"].iloc[-1])
    previous_price = float(data["Close"].iloc[-2])

    price_change = latest_price - previous_price

    percent_change = (
        price_change / previous_price * 100
        if previous_price != 0
        else 0
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.subheader(f"📊 {ticker} — Stock overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest adjusted close",
        f"${latest_price:,.2f}",
        f"{price_change:+.2f} ({percent_change:+.2f}%)"
    )

    col2.metric(
        "Historical observations",
        f"{len(data):,}"
    )

    col3.metric(
        "Historical period",
        period_label
    )

    col4.metric(
        "Forecast horizon",
        f"{horizon} days"
    )

    st.caption(
        "Historical prices are adjusted for relevant corporate actions. "
        "The latest observation may not be today's live market price."
    )

    # --------------------------------------------------------
    # HISTORICAL CHART
    # --------------------------------------------------------

    st.subheader("📈 Historical price")

    historical_fig = go.Figure()

    historical_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Adjusted close",
            line=dict(width=2)
        )
    )

    historical_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(
        historical_fig,
        use_container_width=True
    )

    with st.expander("View historical data"):
        st.dataframe(
            data.sort_index(ascending=False),
            use_container_width=True
        )

    # --------------------------------------------------------
    # NEWS SECTION
    # --------------------------------------------------------

    st.divider()
    st.subheader(f"📰 Recent company news: {ticker}")

    with st.spinner("Retrieving recent news..."):
        news, news_error, news_query, diagnostics = load_news(ticker)

    st.caption(f"Search query: {news_query}")

    if news:

        for article in news[:10]:

            st.markdown(
                f"**[{article['title']}]({article['link']})**"
            )

            st.caption(
                f"{article['publisher']} · {article['published']}"
            )

            st.divider()

    else:

        st.info(
            "No usable headlines were returned. "
            "Stock analysis and forecasting can continue below."
        )

        if news_error:
            st.caption(f"News retrieval detail: {news_error}")

        if show_diagnostics:

            with st.expander(
                "News diagnostics",
                expanded=True
            ):

                st.write(
                    "**Search query:**",
                    diagnostics.get("query", "")
                )

                st.write(
                    "**Response type:**",
                    diagnostics.get("response_type", "Unknown")
                )

                st.write(
                    "**Articles received:**",
                    diagnostics.get("article_count", 0)
                )

                st.write(
                    "**First article keys:**",
                    diagnostics.get("first_article_keys", [])
                )

                sample = diagnostics.get("first_article_sample")

                if sample is not None:
                    st.write("**First article sample:**")
                    st.json(sample)

                st.caption(
                    "If no articles were received, Yahoo Finance may "
                    "have returned an empty result or restricted access. "
                    "If articles were received but none appeared, the "
                    "sample can help identify the correct field names."
                )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    st.divider()
    st.subheader("🤖 Machine-learning forecast")

    with st.spinner("Training and evaluating the Random Forest model..."):

        (
            forecast_df,
            mae,
            directional_accuracy,
            training_rows
        ) = train_and_forecast(data, horizon)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Test MAE",
        f"{mae:.4%}"
    )

    metric2.metric(
        "Directional accuracy",
        f"{directional_accuracy:.1%}"
    )

    metric3.metric(
        "Training observations",
        f"{training_rows:,}"
    )

    st.caption(
        "MAE measures average absolute prediction error in daily returns. "
        "Directional accuracy measures how often predicted and actual "
        "returns have the same sign on the held-out test set. "
        "These historical metrics do not guarantee future accuracy."
    )

    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    st.subheader(f"🔮 Next {horizon} trading-day forecast")

    forecast_fig = go.Figure()

    recent_history = data.tail(
        min(90, len(data))
    )

    forecast_fig.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history["Close"],
            mode="lines",
            name="Historical adjusted close",
            line=dict(width=2)
        )
    )

    connecting_dates = [
        data.index[-1],
        *forecast_df.index.tolist()
    ]

    connecting_prices = [
        latest_price,
        *forecast_df["Predicted Price"].tolist()
    ]

    forecast_fig.add_trace(
        go.Scatter(
            x=connecting_dates,
            y=connecting_prices,
            mode="lines+markers",
            name="Model forecast",
            line=dict(width=2, dash="dash"),
            marker=dict(size=6)
        )
    )

    forecast_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader("Forecast values")

    display_forecast = forecast_df.copy()
    display_forecast.index.name = "Forecast date"

    display_forecast["Predicted Price"] = (
        display_forecast["Predicted Price"].map(
            lambda value: f"${value:,.2f}"
        )
    )

    display_forecast["Predicted Return"] = (
        display_forecast["Predicted Return"].map(
            lambda value: f"{value:+.2%}"
        )
    )

    st.dataframe(
        display_forecast,
        use_container_width=True
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.warning(
        "**Model limitations:** This is an experimental educational "
        "forecast, not a reliable prediction of future market prices. "
        "The model uses historical price and volume-derived features. "
        "It does not include company fundamentals, news sentiment, "
        "economic conditions, or other external factors. Multi-day "
        "forecasts are recursive, so errors can accumulate. Future "
        "volume is carried forward as a placeholder, and forecast dates "
        "use business days rather than an exchange-specific calendar."
    )

except Exception as exc:

    st.error(
        f"Unable to complete the analysis for {ticker}: "
        f"{type(exc).__name__}: {exc}"
    )

    st.info(
        "Check the ticker symbol and try again. If the problem persists, "
        "review the Streamlit app logs for more details."
    )
