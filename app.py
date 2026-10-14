
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="MarketScope AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MarketScope AI")
st.write(
    "Explore historical stock prices, recent news, and "
    "experimental machine-learning forecasts."
)


# -------------------------
# User controls
# -------------------------
st.sidebar.header("Stock Search")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="NVDA"
).strip().upper()

period_options = {
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Historical data",
    list(period_options.keys()),
    index=1
)

forecast_days = st.sidebar.selectbox(
    "Forecast horizon (trading days)",
    [5, 10, 20],
    index=1
)


# -------------------------
# Historical data retrieval
# -------------------------
@st.cache_data(ttl=300)
def load_data(symbol, period):
    df = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if df.empty:
        return df

    return df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna()



@st.cache_data(ttl=60)
def load_news(symbol):
    """Retrieve news headlines from Yahoo Finance."""

    try:
        search = yf.Search(
            symbol,
            news_count=20,
            max_results=10
        )

        search_results = search.search()

        # Handle either a dictionary or an object response.
        if isinstance(search_results, dict):
            articles = search_results.get("news", []) or []
        else:
            articles = getattr(
                search_results, "news", []
            ) or []

        results = []

        for article in articles:
            title = article.get("title", "")
            link = article.get("link", "")
            publisher = article.get(
                "publisher", "Unknown source"
            )

            if not title or not link:
                continue

            if not link.startswith("https://"):
                continue

            published = article.get("providerPublishTime")

            if isinstance(published, (int, float)):
                try:
                    published = datetime.fromtimestamp(
                        published,
                        tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC")
                except (ValueError, OSError, OverflowError):
                    published = "Date unavailable"
            else:
                published = article.get(
                    "pubDate", "Date unavailable"
                )

            results.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": published
            })

        return results, None

    except Exception as e:
        return [], f"{type(e).__name__}: {e}"



# -------------------------
# Feature engineering
# -------------------------
def make_features(df):
    """Build features from historical market observations."""

    close = df["Close"]
    returns = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # Historical returns
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f"return_lag_{lag}"] = returns.shift(lag)

    # Momentum
    features["momentum_5"] = close.pct_change(5)
    features["momentum_10"] = close.pct_change(10)
    features["momentum_20"] = close.pct_change(20)

    # Volatility
    features["volatility_5"] = returns.rolling(5).std()
    features["volatility_20"] = returns.rolling(20).std()

    # Trading volume changes
    features["volume_change"] = df["Volume"].pct_change()
    features["volume_change_5"] = df["Volume"].pct_change(5)

    # Next trading day's return
    target = returns.shift(-1)

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features, target


# -------------------------
# Train model and forecast
# -------------------------
def train_and_forecast(df, horizon):

    features, target = make_features(df)

    dataset = features.copy()
    dataset["target"] = target
    dataset = dataset.dropna()

    if len(dataset) < 150:
        raise ValueError(
            "Not enough clean historical observations. "
            "Try a longer historical period."
        )

    X = dataset.drop(columns=["target"])
    y = dataset["target"]

    # Chronological train/test split
    split = int(len(dataset) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if len(X_test) == 0:
        raise ValueError(
            "Not enough observations for model evaluation."
        )

    # Evaluate on the held-out period
    test_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    test_model.fit(X_train, y_train)

    test_predictions = test_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    directional_accuracy = np.mean(
        np.sign(test_predictions)
        == np.sign(y_test.to_numpy())
    )

    # Train final model on all labeled observations
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Forecast recursively
    history = df.copy()

    future_dates = []
    predicted_prices = []

    last_price = float(history["Close"].iloc[-1])

    for _ in range(horizon):

        current_features, _ = make_features(history)
        latest_features = current_features.iloc[[-1]]

        if latest_features.isna().any(axis=None):
            raise ValueError(
                "Unable to construct forecast features."
            )

        predicted_return = float(
            model.predict(latest_features)[0]
        )

        # Cap extreme extrapolated daily returns
        predicted_return = float(
            np.clip(predicted_return, -0.15, 0.15)
        )

        last_price *= 1 + predicted_return
        predicted_prices.append(last_price)

        # Approximate the next business day.
        # Exchange holidays are not accounted for.
        next_date = (
            history.index[-1] + pd.offsets.BDay(1)
        )

        future_dates.append(next_date)

        # Placeholder for unknown future observations.
        # Future volume is assumed to remain unchanged.
        next_row = pd.DataFrame(
            {
                "Open": [last_price],
                "High": [last_price],
                "Low": [last_price],
                "Close": [last_price],
                "Volume": [
                    history["Volume"].iloc[-1]
                ]
            },
            index=[next_date]
        )

        history = pd.concat([history, next_row])

    forecast = pd.Series(
        predicted_prices,
        index=future_dates,
        name="Predicted adjusted price"
    )

    return (
        forecast,
        mae,
        directional_accuracy,
        len(X_test)
    )


# -------------------------
# Main application
# -------------------------
if not ticker:
    st.info("Enter a stock ticker in the sidebar.")
    st.stop()

try:

    # Load stock history
    with st.spinner(f"Loading {ticker}..."):
        data = load_data(
            ticker,
            period_options[selected_period]
        )

    if data.empty:
        st.error(
            "No historical data found. "
            "Check the ticker symbol."
        )
        st.stop()

    latest_price = float(data["Close"].iloc[-1])

    # -------------------------
    # Stock overview
    # -------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest available close",
        f"${latest_price:,.2f}"
    )

    if len(data) > 1:
        previous_close = float(data["Close"].iloc[-2])

        change = latest_price - previous_close

        change_pct = (
            change / previous_close * 100
            if previous_close != 0
            else 0
        )

        col2.metric(
            "Daily change",
            f"${change:,.2f}",
            f"{change_pct:+.2f}%"
        )
    else:
        col2.metric("Daily change", "Unavailable")

    col3.metric(
        "Historical observations",
        f"{len(data):,}"
    )

    # -------------------------
    # Recent news
    # -------------------------
    st.subheader(f"📰 Recent news search: {ticker}")

    with st.spinner("Searching Yahoo Finance news..."):
        news_items, news_error = load_news(ticker)

    if news_error:

        # Temporary diagnostic message:
        # Shows the actual exception instead of hiding it.
        st.warning(
            f"News retrieval error: {news_error}"
        )

    elif not news_items:

        st.info(
            "The search completed, but no usable headlines "
            "were returned."
        )

    else:

        st.caption(
            f"{len(news_items)} available headlines. "
            "Some results may concern related companies "
            "or broader markets."
        )

        for item in news_items:

            with st.container(border=True):

                st.markdown(
                    f"**{item['title']}**"
                )

                st.caption(
                    f"{item['publisher']} · "
                    f"{item['published']}"
                )

                st.markdown(
                    f"[Read original article]({item['link']})"
                )

    # -------------------------
    # Forecast
    # -------------------------
    with st.spinner(
        "Training and evaluating the model..."
    ):

        (
            forecast,
            mae,
            direction_acc,
            test_count
        ) = train_and_forecast(
            data,
            forecast_days
        )

    # -------------------------
    # Stock chart
    # -------------------------
    st.subheader(
        "Historical prices and experimental forecast"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Historical adjusted close",
            line=dict(color="#2563eb", width=2)
        )
    )

    # Connect historical price to the first forecast point
    fig.add_trace(
        go.Scatter(
            x=[data.index[-1], forecast.index[0]],
            y=[
                data["Close"].iloc[-1],
                forecast.iloc[0]
            ],
            mode="lines",
            showlegend=False,
            line=dict(
                color="#f59e0b",
                width=2,
                dash="dash"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode="lines+markers",
            name="Experimental forecast",
            line=dict(
                color="#f59e0b",
                width=3,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=600,
        xaxis_title="Date",
        yaxis_title="Adjusted price (USD)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # Forecast summary
    # -------------------------
    predicted_price = float(forecast.iloc[-1])

    predicted_change = (
        predicted_price / latest_price - 1
    ) * 100

    st.subheader("Forecast summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        f"Estimated price in {forecast_days} trading days",
        f"${predicted_price:,.2f}"
    )

    col2.metric(
        "Projected cumulative change",
        f"{predicted_change:+.2f}%"
    )

    col3.metric(
        "Held-out directional accuracy",
        f"{direction_acc:.1%}"
    )

    st.caption(
        f"Test-period return MAE: {mae:.4%} per day. "
        f"Evaluation observations: {test_count}."
    )

    # -------------------------
    # Limitations
    # -------------------------
    st.warning(
        "Experimental research forecast only. The model uses "
        "historical market features and does not yet use news "
        "sentiment, political events, company announcements, "
        "or future market conditions. Recursive forecasts can "
        "compound errors. The projected line is not a "
        "confidence interval. Past performance does not "
        "guarantee future results."
    )

    # -------------------------
    # Forecast table
    # -------------------------
    with st.expander("View forecast values"):

        st.dataframe(
            forecast.to_frame(),
            use_container_width=True
        )

    # -------------------------
    # Historical data table
    # -------------------------
    with st.expander("View historical data"):

        st.dataframe(
            data.sort_index(ascending=False),
            use_container_width=True
        )

except Exception as e:

    st.error(
        "The analysis could not be completed. "
        "Check the ticker, data availability, or logs."
    )

    st.exception(e)
