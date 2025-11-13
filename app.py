import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import requests


# ---------------------- GLOBAL STYLE ---------------------- #

APP_CSS = """
<style>
/* Base */
html, body, [class*="css"] {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", sans-serif;
}

/* App background */
main.stApp {
    background: radial-gradient(circle at top, #0b2b1b 0, #021109 30%, #020b07 60%, #000 100%);
    color: #f4f6f3;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, rgba(12, 50, 32, 0.95), rgba(26, 82, 52, 0.96));
    border-radius: 24px;
    padding: 24px 28px;
    border: 1px solid rgba(198, 166, 103, 0.4);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-left h1 {
    font-size: 2.1rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
    color: #f8f5e9;
}
.hero-tagline {
    font-size: 0.95rem;
    opacity: 0.9;
}
.hero-pill {
    background: rgba(5, 26, 15, 0.9);
    border-radius: 999px;
    padding: 8px 16px;
    border: 1px solid rgba(198, 166, 103, 0.6);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
}
.hero-pill span.emoji {
    font-size: 1.2rem;
}
.hero-badge {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(246, 233, 187, 0.9);
}

/* KPI cards */
.kpi-row {
    margin-top: 18px;
    margin-bottom: 8px;
}
.kpi-card {
    background: radial-gradient(circle at top left, rgba(36, 96, 60, 0.9), rgba(9, 32, 21, 0.96));
    border-radius: 20px;
    padding: 16px 18px 14px 18px;
    border: 1px solid rgba(188, 158, 99, 0.35);
    box-shadow: 0 14px 32px rgba(0, 0, 0, 0.55);
}
.kpi-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    opacity: 0.85;
    color: #f5edd8;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 600;
    margin-top: 4px;
    color: #ffffff;
}
.kpi-sub {
    font-size: 0.85rem;
    opacity: 0.85;
    margin-top: 4px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(7, 31, 20, 0.85);
    border-radius: 999px;
    padding: 6px 16px;
    border: 1px solid rgba(87, 122, 98, 0.6);
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: rgba(198, 166, 103, 0.9);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #255b3a, #386c49);
    color: #f9f6ee;
    border: 1px solid rgba(198, 166, 103, 0.9);
}

/* Section headers */
.section-header {
    margin-top: 20px;
    margin-bottom: 6px;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f3e5c3;
}

/* DataFrames */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    border: 1px solid rgba(198, 166, 103, 0.9);
    background: linear-gradient(135deg, #2a5d3c, #173724);
    color: #f9f6ee;
    font-weight: 500;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #376f49, #214332);
    border-color: rgba(240, 206, 135, 1);
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #f7f0da !important;
}
[data-testid="stMetricLabel"] {
    color: #f7f0da !important;
}
</style>
"""


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


def kpi_card(title: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- DATA HELPERS ---------------------- #

def make_demo_data() -> pd.DataFrame:
    """Create a fake tee-sheet for demo purposes."""
    rng = pd.date_range("2024-07-01", periods=14 * 12 * 6, freq="10min")
    df = pd.DataFrame({"tee_time": rng})
    df = df[df["tee_time"].dt.hour.between(6, 18)]

    # fake booking pattern: mornings + mid-afternoon stronger
    df["booked"] = np.where(
        (df["tee_time"].dt.hour.between(7, 10))
        | (df["tee_time"].dt.hour.between(14, 16)),
        np.random.binomial(1, 0.8, size=len(df)),
        np.random.binomial(1, 0.35, size=len(df)),
    )

    base_price = np.where(df["tee_time"].dt.hour < 12, 90, 75)
    noise = np.random.normal(0, 8, size=len(df))
    df["price"] = base_price + noise

    # optional "member / guest" + channel columns for extra dashboards
    df["player_type"] = np.where(
        np.random.rand(len(df)) < 0.55, "Member", "Guest"
    )
    df["channel"] = np.where(
        np.random.rand(len(df)) < 0.7, "Pro Shop", "Online"
    )
    return df


def coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float)):
        return x == 1
    if isinstance(x, str):
        return x.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "booked",
            "sold",
            "reserved",
        }
    return False


def ensure_datetime_col(df: pd.DataFrame) -> pd.DataFrame:
    """Find or build a tee_time datetime column."""
    for c in ["tee_time", "datetime", "start_time", "time", "date_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].notna().any():
                df["tee_time"] = df[c]
                return df

    date_cols = [c for c in df.columns if "date" in c.lower()]
    time_cols = [c for c in df.columns if "time" in c.lower()]

    if date_cols and time_cols:
        df["tee_time"] = pd.to_datetime(
            df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str),
            errors="coerce",
        )
        if df["tee_time"].notna().any():
            return df

    raise ValueError("No datetime column found. Provide tee_time or (date + time).")


def clean_teetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize tee-sheet CSV into standard format."""
    df = df.copy()
    df = ensure_datetime_col(df)

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan

    # booked flag
    book_col = next(
        (
            c
            for c in df.columns
            if c.lower() in {"booked", "is_booked", "reserved", "filled", "status"}
        ),
        None,
    )
    if book_col:
        df["booked"] = df[book_col].apply(coerce_bool)
    else:
        df["booked"] = False

    df["weekday"] = df["tee_time"].dt.day_name()
    df["hour"] = df["tee_time"].dt.hour
    df["date"] = df["tee_time"].dt.date

    # fill missing price
    if df["price"].isna().any():
        group_med = df.groupby(["weekday", "hour"])["price"].transform("median")
        df["price"] = df["price"].fillna(group_med).fillna(df["price"].median())

    return df.sort_values("tee_time").reset_index(drop=True)


def add_time_bins(df: pd.DataFrame, slot_minutes: int = 10) -> pd.DataFrame:
    """Create slot_index + slot_hour/minute based on tee_time."""
    df = df.copy()
    dt = df["tee_time"]
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute

    slot_index = (minute_of_day // slot_minutes).astype(int)
    slot_start_min = slot_index * slot_minutes
    slot_hour = (slot_start_min // 60).astype(int)
    slot_min = (slot_start_min % 60).astype(int)

    df["slot_index"] = slot_index
    df["slot_minutes"] = slot_minutes
    df["slot_hour"] = slot_hour
    df["slot_minute"] = slot_min

    return df


def fmt_time_ampm(h: int, m: int) -> str:
    hh = h % 12 or 12
    ampm = "AM" if h < 12 else "PM"
    return f"{hh}:{m:02d}{ampm}"


def kpis(df: pd.DataFrame):
    total = len(df)
    booked = int(df["booked"].sum())
    util = booked / total if total else 0.0
    revenue = float(df.loc[df["booked"], "price"].sum())
    potential = float(df["price"].sum())
    rev_per_round = revenue / booked if booked else 0.0
    rev_patt = revenue / total if total else 0.0  # revenue per available tee time
    return total, booked, util, revenue, potential, rev_per_round, rev_patt


def daily_utilization(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date").agg(util=("booked", "mean")).reset_index()


def utilization_matrix_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["weekday", "hour"]).agg(
        slots=("booked", "size"),
        booked=("booked", "sum"),
    )
    grp["util"] = np.where(grp["slots"] > 0, grp["booked"] / grp["slots"], np.nan)
    return grp["util"].unstack("hour").sort_index()


# ---------------------- PRICING / MODEL ---------------------- #

def featurize(tee_df: pd.DataFrame, slot_minutes: int = 10):
    df = add_time_bins(tee_df, slot_minutes=slot_minutes).copy()
    df["is_weekend"] = df["tee_time"].dt.weekday >= 5

    minute_of_day = df["slot_hour"] * 60 + df["slot_minute"]

    X = pd.DataFrame(
        {
            "slot_index": df["slot_index"],
            "minute_of_day": minute_of_day,
            "is_weekend": df["is_weekend"].astype(int),
            "price": df["price"],
        }
    )

    y = df["booked"].astype(int)
    meta = df[["weekday", "slot_index", "slot_hour", "slot_minute", "price"]]

    return X, y, meta


def train_model(tee_df: pd.DataFrame, slot_minutes: int = 10):
    X, y, _ = featurize(tee_df, slot_minutes=slot_minutes)
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X, y)
    return clf


def expected_utilization(clf, tee_df: pd.DataFrame, slot_minutes: int = 10):
    X, _, meta = featurize(tee_df, slot_minutes=slot_minutes)
    proba = clf.predict_proba(X)[:, 1]

    meta = meta.copy()
    meta["p_book"] = proba

    agg = meta.groupby(
        ["weekday", "slot_index", "slot_hour", "slot_minute"]
    ).agg(expected_util=("p_book", "mean"), avg_price=("price", "mean")).reset_index()

    return agg


def compute_pricing_actions(
    tee_df: pd.DataFrame,
    slot_minutes: int = 10,
    target_util: float = 0.75,
    top_n: int = 10,
) -> pd.DataFrame:
    """Always returns the softest blocks, never empty unless no data."""
    if tee_df.empty:
        return pd.DataFrame()

    try:
        clf = train_model(tee_df, slot_minutes=slot_minutes)
        util_df = expected_utilization(clf, tee_df, slot_minutes=slot_minutes)
    except Exception:
        tmp = add_time_bins(tee_df, slot_minutes=slot_minutes)
        grp = tmp.groupby(
            ["weekday", "slot_index", "slot_hour", "slot_minute"]
        ).agg(
            slots=("booked", "size"),
            booked=("booked", "sum"),
            avg_price=("price", "mean"),
        ).reset_index()
        grp["expected_util"] = np.where(
            grp["slots"] > 0, grp["booked"] / grp["slots"], 0.0
        )
        util_df = grp

    util_df = util_df.sort_values(
        ["expected_util", "weekday", "slot_index"]
    ).reset_index(drop=True)

    soft = util_df.head(top_n).copy()

    gap = (target_util - soft["expected_util"]).clip(lower=0)
    soft["suggested_discount"] = (0.10 + 0.20 * (gap / target_util)).clip(upper=0.35)
    soft["new_price"] = soft["avg_price"] * (1 - soft["suggested_discount"])

    soft["Time"] = soft.apply(
        lambda r: fmt_time_ampm(int(r["slot_hour"]), int(r["slot_minute"])), axis=1
    )
    soft["Expected Utilization"] = (soft["expected_util"] * 100).map(
        lambda x: f"{x:.2f}%"
    )
    soft["Average Price"] = soft["avg_price"].map(lambda x: f"${x:.2f}")
    soft["Suggested Discount"] = (soft["suggested_discount"] * 100).map(
        lambda x: f"{x:.2f}%"
    )
    soft["New Price"] = soft["new_price"].map(lambda x: f"${x:.2f}")

    pretty = soft[
        [
            "weekday",
            "Time",
            "Expected Utilization",
            "Average Price",
            "Suggested Discount",
            "New Price",
        ]
    ].rename(columns={"weekday": "Weekday"})

    return pretty


def build_text_report(df: pd.DataFrame, slot_minutes: int) -> str:
    """Generate a simple text report summarizing performance and pricing."""
    total, booked, util, revenue, potential, rev_per_round, rev_patt = kpis(df)
    gap = potential - revenue

    today_date = df["date"].max()
    today_df = df[df["date"] == today_date]
    if not today_df.empty:
        t_total, t_booked, t_util, t_rev, t_pot, _, _ = kpis(today_df)
        t_gap = t_pot - t_rev
    else:
        t_total = t_booked = 0
        t_util = t_rev = t_pot = t_gap = 0.0

    trend = daily_utilization(df)
    avg_util_7d = trend["util"].tail(7).mean() * 100 if len(trend) >= 1 else 0.0

    actions = compute_pricing_actions(
        df, slot_minutes=slot_minutes, target_util=0.75, top_n=10
    )

    lines = []
    lines.append("TeeIQ Elite Performance Report")
    lines.append("================================")
    lines.append("")
    lines.append(f"Tee-time interval: {slot_minutes} minutes")
    lines.append("")
    lines.append("Overall performance")
    lines.append("-------------------")
    lines.append(f"Total slots:       {total:,}")
    lines.append(f"Booked slots:      {booked:,}")
    lines.append(f"Utilization:       {util*100:.1f}%")
    lines.append(f"Booked revenue:    ${revenue:,.0f}")
    lines.append(f"Potential revenue: ${potential:,.0f}")
    lines.append(f"Revenue gap:       ${gap:,.0f}")
    lines.append(f"Rev / round:       ${rev_per_round:,.0f}")
    lines.append(f"Rev / tee time:    ${rev_patt:,.0f}")
    lines.append("")
    lines.append("Most recent day")
    lines.append("---------------")
    lines.append(f"Date:              {today_date}")
    lines.append(f"Slots:             {t_total:,}")
    lines.append(f"Booked slots:      {t_booked:,}")
    lines.append(f"Utilization:       {t_util*100:.1f}%")
    lines.append(f"Booked revenue:    ${t_rev:,.0f}")
    lines.append(f"Revenue gap:       ${t_gap:,.0f}")
    lines.append("")
    lines.append("Recent trend")
    lines.append("------------")
    lines.append(f"Average utilization over last 7 days: {avg_util_7d:.1f}%")
    lines.append("")
    lines.append("AI pricing suggestions (top softest blocks)")
    lines.append("--------------------------------------------")

    if actions.empty:
        lines.append("No pricing suggestions available (not enough data).")
    else:
        lines.append(f"Showing {len(actions)} softest blocks by expected utilization:")
        lines.append("")
        for _, row in actions.iterrows():
            lines.append(
                f"- {row['Weekday']} @ {row['Time']}: "
                f"Util {row['Expected Utilization']}, "
                f"Avg {row['Average Price']}, "
                f"Discount {row['Suggested Discount']}, "
                f"New {row['New Price']}"
            )

    lines.append("")
    lines.append("Next steps")
    lines.append("----------")
    lines.append("1) Apply recommended pricing to softest blocks.")
    lines.append("2) Monitor utilization and revenue over the next 7 days.")
    lines.append("3) Re-run TeeIQ and update your pricing plan.")

    return "\n".join(lines)


# ---------------------- OPTIONAL WEATHER ---------------------- #

def fetch_daily_weather(lat: float, lon: float):
    """Free weather via Open-Meteo (no API key). Returns small DataFrame or None."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "daily" not in data or "time" not in data["daily"]:
            return None
        d = data["daily"]
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(d["time"]).date,
                "temp_max": d["temperature_2m_max"],
                "temp_min": d["temperature_2m_min"],
                "precip": d["precipitation_sum"],
            }
        )
        return out
    except Exception:
        return None


# ---------------------- STREAMLIT APP ---------------------- #

st.set_page_config(
    page_title="TeeIQ Elite – Run your course like a hedge fund",
    page_icon="⛳",
    layout="wide",
)


def main() -> None:
    inject_css()

    # HERO
    st.markdown(
        """
        <div class="hero">
            <div class="hero-left">
                <div class="hero-badge">TEEIQ ELITE</div>
                <h1>Run Your Course Like a Hedge Fund</h1>
                <div class="hero-tagline">
                    Slot-level pricing, utilization, and member insights for championship clubs.
                </div>
            </div>
            <div class="hero-right">
                <div class="hero-pill">
                    <span class="emoji">⛳</span>
                    <span>Optimizing today's tee sheet • Protecting tomorrow's prestige</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar: data source
    with st.sidebar:
        st.subheader("Tee Sheet Data")
        tee_file = st.file_uploader("Upload tee-sheet CSV", type=["csv"])
        use_demo = st.checkbox("Use demo data", value=not bool(tee_file))

        st.markdown("---")
        st.subheader("Tee Grid")
        slot_minutes = st.selectbox(
            "Tee-time interval (minutes)",
            options=[5, 7, 8, 9, 10, 12, 15],
            index=[5, 7, 8, 9, 10, 12, 15].index(10),
            help="Match your tee-sheet spacing.",
        )

        st.markdown("---")
        st.subheader("Weather (optional)")
        lat = st.number_input("Latitude", value=33.503, format="%.6f")
        lon = st.number_input("Longitude", value=-82.020, format="%.6f")
        want_weather = st.checkbox("Include 7-day weather outlook", value=False)

    if tee_file is not None:
        raw_df = pd.read_csv(tee_file)
    elif use_demo:
        raw_df = make_demo_data()
    else:
        raw_df = None

    if raw_df is None:
        st.info("Upload a tee-sheet CSV or enable 'Use demo data' in the sidebar.")
        return

    # Clean data
    try:
        df = clean_teetimes(raw_df)
    except Exception as e:
        st.error(f"Data error: {e}")
        return

    # KPIs
    (
        total,
        booked,
        util,
        revenue,
        potential,
        rev_per_round,
        rev_patt,
    ) = kpis(df)

    st.markdown('<div class="kpi-row"></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Utilization",
            f"{util*100:.0f}%",
            f"{booked:,} of {total:,} tee times booked",
        )
    with c2:
        kpi_card(
            "Booked Revenue",
            f"${revenue:,.0f}",
            f"Potential at rack: ${potential:,.0f}",
        )
    with c3:
        gap = potential - revenue
        kpi_card(
            "Revenue Gap",
            f"${gap:,.0f}",
            "Lost to empty or underpriced inventory",
        )
    with c4:
        kpi_card(
            "RevPATT",
            f"${rev_patt:,.0f}",
            "Revenue per available tee time",
        )

    # Tabs
    tab_overview, tab_teetimes, tab_pricing, tab_mix, tab_weather, tab_reports = st.tabs(
        [
            "Overview",
            "Tee Sheet & Heatmaps",
            "Dynamic Pricing AI",
            "Members & Channels",
            "Weather & Pace",
            "Reports",
        ]
    )

    # Overview
    with tab_overview:
        st.markdown(
            '<div class="section-header">Booking Trend</div>',
            unsafe_allow_html=True,
        )
        trend = daily_utilization(df)
        if trend.empty:
            st.info("Not enough data to show a trend.")
        else:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(trend["date"], trend["util"] * 100, marker="o")
            ax.set_ylabel("Utilization (%)", color="#f7f2df")
            ax.set_xlabel("Date", color="#f7f2df")
            ax.tick_params(colors="#f7f2df")
            ax.spines["bottom"].set_color("#f7f2df")
            ax.spines["left"].set_color("#f7f2df")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<div class="section-header">Most Recent Day Snapshot</div>',
            unsafe_allow_html=True,
        )
        today = df[df["date"] == df["date"].max()]
        if today.empty:
            st.info("No data for the most recent day.")
        else:
            (
                t_total,
                t_booked,
                t_util,
                t_rev,
                t_pot,
                t_rpr,
                t_rpatt,
            ) = kpis(today)
            d1, d2, d3 = st.columns(3)
            with d1:
                kpi_card(
                    "Today Utilization",
                    f"{t_util*100:.0f}%",
                    f"{t_booked:,}/{t_total:,} tee times",
                )
            with d2:
                kpi_card("Today Revenue", f"${t_rev:,.0f}", "")
            with d3:
                kpi_card("Today RevPATT", f"${t_rpatt:,.0f}", "")

    # Tee Sheet and Heatmaps
    with tab_teetimes:
        st.markdown(
            '<div class="section-header">Weekly Utilization Heatmap (by hour)</div>',
            unsafe_allow_html=True,
        )
        mat = utilization_matrix_hour(df)
        if mat.empty:
            st.info("No utilization data.")
        else:
            data = mat.to_numpy()
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            im = ax2.imshow(data, aspect="auto")
            ax2.set_yticks(range(len(mat.index)))
            ax2.set_yticklabels(mat.index)

            hours = list(mat.columns)
            labels = []
            for h in hours:
                hh = h % 12 or 12
                ampm = "AM" if h < 12 else "PM"
                labels.append(f"{hh}{ampm}")
            ax2.set_xticks(range(len(hours)))
            ax2.set_xticklabels(labels)
            ax2.set_xlabel("Hour of day")
            cbar = fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
            cbar.set_label("Utilization")

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if not np.isnan(v):
                        ax2.text(
                            j,
                            i,
                            f"{v*100:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                        )

            st.pyplot(fig2, use_container_width=True)

        st.markdown(
            '<div class="section-header">Raw Tee Sheet (sample)</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df[["tee_time", "price", "booked", "weekday", "hour"]].head(50),
            use_container_width=True,
        )

    # Dynamic Pricing AI
    with tab_pricing:
        st.markdown(
            '<div class="section-header">Dynamic Pricing Suggestions</div>',
            unsafe_allow_html=True,
        )

        colL, colR = st.columns([2, 1])
        with colL:
            top_n = st.slider("Number of softest blocks to show", 5, 30, 10, 1)
        with colR:
            target_util = st.slider("Target utilization", 0.5, 0.95, 0.75, 0.01)

        if st.button("Generate pricing actions"):
            actions = compute_pricing_actions(
                df,
                slot_minutes=slot_minutes,
                target_util=target_util,
                top_n=top_n,
            )
            if actions.empty:
                st.info("No actions available (no data).")
            else:
                top_row = actions.iloc[0]
                st.markdown("### Top Hole-in-One Recommendation")
                st.markdown(
                    f"**Block:** {top_row['Weekday']} @ {top_row['Time']}  \n"
                    f"**Expected Utilization:** {top_row['Expected Utilization']}  \n"
                    f"**Average Price:** {top_row['Average Price']}  \n"
                    f"**Suggested Discount:** {top_row['Suggested Discount']}  \n"
                    f"**New Price:** {top_row['New Price']}"
                )
                st.caption(
                    "These are your softest blocks. Lead with these for pricing and promotions."
                )
                st.dataframe(actions, use_container_width=True)

                # Quick viz of expected utilization
                chart_df = actions.copy()
                chart_df["UtilPct"] = chart_df["Expected Utilization"].str.rstrip(
                    "%"
                ).astype(float)
                fig3, ax3 = plt.subplots(figsize=(10, 3))
                ax3.bar(chart_df["Time"], chart_df["UtilPct"])
                ax3.set_ylabel("Expected Utilization (%)")
                ax3.set_xlabel("Time of Day (softest → harder)")
                ax3.tick_params(axis="x", rotation=90)
                st.pyplot(fig3, use_container_width=True)

    # Members & Channels
    with tab_mix:
        st.markdown(
            '<div class="section-header">Member vs Guest Mix</div>',
            unsafe_allow_html=True,
        )
        if "player_type" in df.columns:
            mix = df.groupby("player_type").agg(
                slots=("booked", "size"), booked=("booked", "sum")
            )
            mix["util"] = np.where(
                mix["slots"] > 0, mix["booked"] / mix["slots"], 0
            )
            st.dataframe(mix, use_container_width=True)
        else:
            st.info(
                "No 'player_type' column found. Add a column with values like 'Member' / 'Guest' to unlock this view."
            )

        st.markdown(
            '<div class="section-header">Channel Mix</div>',
            unsafe_allow_html=True,
        )
        if "channel" in df.columns:
            cmix = df.groupby("channel").agg(
                slots=("booked", "size"), booked=("booked", "sum")
            )
            cmix["util"] = np.where(
                cmix["slots"] > 0, cmix["booked"] / cmix["slots"], 0
            )
            st.dataframe(cmix, use_container_width=True)
        else:
            st.info(
                "No 'channel' column found. Add booking channel (e.g., 'Pro Shop', 'Online') to unlock this view."
            )

    # Weather & Pace
    with tab_weather:
        st.markdown(
            '<div class="section-header">7-Day Weather Outlook</div>',
            unsafe_allow_html=True,
        )
        if want_weather:
            wdf = fetch_daily_weather(lat, lon)
            if wdf is None or wdf.empty:
                st.info("Could not fetch weather for these coordinates.")
            else:
                st.dataframe(wdf, use_container_width=True)
                figw, axw = plt.subplots(figsize=(8, 3))
                axw.bar(wdf["date"], wdf["precip"])
                axw.set_ylabel("Precipitation (mm)")
                axw.set_xlabel("Date")
                axw.tick_params(axis="x", rotation=45)
                st.pyplot(figw, use_container_width=True)
        else:
            st.info("Enable 'Include 7-day weather outlook' in the sidebar to see weather here.")

        st.markdown(
            '<div class="section-header">Pace of Play (placeholder)</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "You can later integrate GPS / cart or marshal data here to show pace-of-play hotspots."
        )

    # Reports
    with tab_reports:
        st.markdown(
            '<div class="section-header">Executive Report</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Generate a concise text report summarizing utilization, revenue, and AI pricing suggestions "
            "that you can share with ownership or the board."
        )

        if st.button("Generate report"):
            report_text = build_text_report(df, slot_minutes)
            st.text_area("Report preview", value=report_text, height=320)
            st.download_button(
                "Download report (.txt)",
                data=report_text.encode(),
                file_name="teeiq_elite_report.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
