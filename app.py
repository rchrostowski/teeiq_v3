import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# -------------- DATA HELPERS --------------


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
    return df


def coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float)):
        return x == 1
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "booked", "sold", "reserved"}
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
        (c for c in df.columns
         if c.lower() in {"booked", "is_booked", "reserved", "filled", "status"}),
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
    return total, booked, util, revenue, potential


def daily_utilization(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date").agg(util=("booked", "mean")).reset_index()


def utilization_matrix_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(["weekday", "hour"]).agg(
        slots=("booked", "size"),
        booked=("booked", "sum"),
    )
    grp["util"] = np.where(
        grp["slots"] > 0, grp["booked"] / grp["slots"], np.nan,
    )
    return grp["util"].unstack("hour").sort_index()


# -------------- PRICING / MODEL --------------


def featurize(tee_df: pd.DataFrame, slot_minutes: int = 10):
    df = add_time_bins(tee_df, slot_minutes=slot_minutes).copy()
    df["is_weekend"] = df["tee_time"].dt.weekday >= 5

    minute_of_day = df["slot_hour"] * 60 + df["slot_minute"]

    X = pd.DataFrame({
        "slot_index": df["slot_index"],
        "minute_of_day": minute_of_day,
        "is_weekend": df["is_weekend"].astype(int),
        "price": df["price"],
    })

    y = df["booked"].astype(int)
    meta = df[
        ["weekday", "slot_index", "slot_hour", "slot_minute", "price"]
    ]

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
    ).agg(
        expected_util=("p_book", "mean"),
        avg_price=("price", "mean"),
    ).reset_index()

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
            grp["slots"] > 0, grp["booked"] / grp["slots"], 0.0,
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
        lambda r: fmt_time_ampm(int(r["slot_hour"]), int(r["slot_minute"])),
        axis=1,
    )
    soft["Expected Utilization"] = (soft["expected_util"] * 100).map(
        lambda x: f"{x:.2f}%",
    )
    soft["Average Price"] = soft["avg_price"].map(lambda x: f"${x:.2f}")
    soft["Suggested Discount"] = (soft["suggested_discount"] * 100).map(
        lambda x: f"{x:.2f}%",
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
    total, booked, util, revenue, potential = kpis(df)
    gap = potential - revenue

    today_date = df["date"].max()
    today_df = df[df["date"] == today_date]
    if not today_df.empty:
        t_total, t_booked, t_util, t_rev, t_pot = kpis(today_df)
        t_gap = t_pot - t_rev
    else:
        t_total = t_booked = 0
        t_util = t_rev = t_pot = t_gap = 0.0

    trend = daily_utilization(df)
    avg_util_7d = trend["util"].tail(7).mean() * 100 if len(trend) >= 1 else 0.0

    actions = compute_pricing_actions(df, slot_minutes=slot_minutes, target_util=0.75, top_n=10)

    lines = []
    lines.append("TeeIQ3 Performance Report")
    lines.append("========================")
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
    lines.append("2) Monitor utilization and revenue over next 7 days.")
    lines.append("3) Re-run TeeIQ3 and update your pricing plan.")

    return "\n".join(lines)


# -------------- STREAMLIT APP --------------


st.set_page_config(
    page_title="TeeIQ3 – Run your course like a hedge fund",
    page_icon="⛳",
    layout="wide",
)


def main() -> None:
    st.title("TeeIQ3")
    st.caption("Run your course like a hedge fund. Slot-level pricing intelligence for golf tee sheets.")

    # Sidebar: data source
    with st.sidebar:
        st.subheader("Data")
        tee_file = st.file_uploader("Upload tee-sheet CSV", type=["csv"])
        use_demo = st.checkbox("Use demo data", value=not bool(tee_file))

    if tee_file is not None:
        raw_df = pd.read_csv(tee_file)
    elif use_demo:
        raw_df = make_demo_data()
    else:
        raw_df = None

    if raw_df is None:
        st.info("Upload a tee-sheet CSV or check 'Use demo data'.")
        return

    # Clean data
    try:
        df = clean_teetimes(raw_df)
    except Exception as e:
        st.error(f"Data error: {e}")
        return

    # Tee time spacing
    slot_minutes = st.sidebar.selectbox(
        "Tee-time interval (minutes)",
        options=[5, 7, 8, 9, 10, 12, 15],
        index=[5, 7, 8, 9, 10, 12, 15].index(10),
        help="Match your tee-sheet spacing.",
    )

    # KPIs
    total, booked, util, revenue, potential = kpis(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Utilization", f"{util*100:.0f}%", f"{booked:,} / {total:,} slots")
    with c2:
        st.metric("Booked Revenue", f"${revenue:,.0f}")
    with c3:
        gap = potential - revenue
        st.metric("Revenue Gap", f"${gap:,.0f}")

    # Tabs
    tab_dash, tab_pricing, tab_util, tab_reports = st.tabs(
        ["Executive Dashboard", "Pricing AI", "Utilization", "Reports"]
    )

    # Executive Dashboard
    with tab_dash:
        st.subheader("Booking Trend")
        trend = daily_utilization(df)
        if trend.empty:
            st.info("Not enough data to show a trend.")
        else:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(trend["date"], trend["util"] * 100, marker="o")
            ax.set_ylabel("Utilization (%)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

        st.subheader("Most Recent Day Snapshot")
        today = df[df["date"] == df["date"].max()]
        if today.empty:
            st.info("No data for the most recent day.")
        else:
            t_total, t_booked, t_util, t_rev, t_pot = kpis(today)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("Today Utilization", f"{t_util*100:.0f}%", f"{t_booked:,}/{t_total:,}")
            with d2:
                st.metric("Today Revenue", f"${t_rev:,.0f}")
            with d3:
                st.metric("Today Gap", f"${(t_pot - t_rev):,.0f}")

    # Pricing AI
    with tab_pricing:
        st.subheader("Dynamic Pricing Suggestions")

        colL, colR = st.columns([2, 1])
        with colL:
            top_n = st.slider("Number of softest blocks to show", 5, 30, 10, 1)
        with colR:
            target_util = st.slider("Target utilization", 0.5, 0.95, 0.75, 0.01)

        if st.button("Generate pricing actions"):
            actions = compute_pricing_actions(
                df, slot_minutes=slot_minutes, target_util=target_util, top_n=top_n,
            )
            if actions.empty:
                st.info("No actions available (no data).")
            else:
                top_row = actions.iloc[0]
                st.markdown("### Top Single Recommendation")
                st.markdown(
                    f"**Block:** {top_row['Weekday']} @ {top_row['Time']}  \n"
                    f"**Expected Utilization:** {top_row['Expected Utilization']}  \n"
                    f"**Average Price:** {top_row['Average Price']}  \n"
                    f"**Suggested Discount:** {top_row['Suggested Discount']}  \n"
                    f"**New Price:** {top_row['New Price']}"
                )
                st.caption("These are your softest blocks. Discount and market them first.")
                st.dataframe(actions, use_container_width=True)

    # Utilization
    with tab_util:
        st.subheader("Weekly Utilization Heatmap (by hour)")
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
                            j, i, f"{v*100:.0f}%",
                            ha="center", va="center",
                            fontsize=7, color="white",
                        )

            st.pyplot(fig2, use_container_width=True)

    # Reports
    with tab_reports:
        st.subheader("Generate Report")
        st.write(
            "Create a simple text report summarizing utilization, revenue, and AI pricing suggestions."
        )

        if st.button("Generate report"):
            report_text = build_text_report(df, slot_minutes)
            st.text_area("Report preview", value=report_text, height=300)
            st.download_button(
                "Download report (.txt)",
                data=report_text.encode(),
                file_name="teeiq3_report.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
