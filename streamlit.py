import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from logger_config import setup_logging, get_logger
from app import run_news_analysis, get_report_as_markdown
from setup import setup_crewai_config, setup_api_keys, check_llm_status
from content_extractor import (
    extract_content_from_image,
    fetch_url_preview,
    extract_keywords_from_text,
    compute_engagement_metrics,
    synthesize_claim,
)
try:
    from reddit import scrape_reddit_data, extract_keywords, is_reddit_url, compute_reddit_engagement
except ImportError:
    pass
from trends_service import get_trends_metrics
import traceback

# Initialize logging ONCE at app startup
setup_logging("DEBUG")

st.set_page_config(
    page_title="VerifAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = get_logger(__name__)

# ─── Helper: safely get a value from a dict or Pydantic model ───
def _get(obj, key, default=None):
    """Get attribute from dict or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ─────────────────────────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────────────────────────

def _plot_wordcloud(words):
    """Render a word cloud from a list of words."""
    try:
        from wordcloud import WordCloud
        if not words:
            return
        freq = {w: max(1, len(words) - i) for i, w in enumerate(words)}
        wc = WordCloud(
            width=800, height=300, background_color="#0e1117",
            colormap="cool", max_words=50, prefer_horizontal=0.7
        ).generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        # Fallback if wordcloud not installed
        st.markdown(" · ".join([f"`{w}`" for w in words]))


def _plot_horizontal_bar(labels, values, title, xlabel, color_palette="viridis", figsize=(10, 4)):
    """Generic horizontal bar chart."""
    if not labels or not values:
        return
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette(color_palette, len(labels))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_donut(value, max_val=1.0, label="Score", color_high="#2ecc71", color_low="#e74c3c"):
    """Donut / gauge chart for a single metric."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ratio = min(value / max_val, 1.0) if max_val else 0
    sizes = [ratio, 1 - ratio]
    color = color_high if ratio >= 0.5 else color_low
    wedges, _ = ax.pie(sizes, colors=[color, "#2d2d2d"], startangle=90,
                       wedgeprops=dict(width=0.35))
    ax.text(0, 0, f"{value:.0%}" if max_val <= 1 else f"{value:.0f}",
            ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(0, -0.15, label, ha="center", va="center", fontsize=9, color="#aaa")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_percentage_bars(metrics_dict, title="Content Analysis Metrics"):
    """Horizontal percentage bars for content analysis metrics."""
    if not metrics_dict:
        return
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c"]
    bars = ax.barh(labels, values, color=colors[:len(labels)], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage (%)")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_time_series(ts_data):
    """Line chart for time series data."""
    if not ts_data:
        return
    df = pd.DataFrame(ts_data)
    try:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(df["Date"], df["Count"], alpha=0.3, color="#3498db")
        ax.plot(df["Date"], df["Count"], marker="o", color="#3498db", linewidth=2)
        ax.set_title("Similar Posts Over Time")
        ax.set_ylabel("Post Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.dataframe(df, hide_index=True)


def _plot_bubble_chart(clusters):
    """Bubble chart for topic clusters."""
    if not clusters:
        return
    names = [c["name"] for c in clusters]
    counts = [max(c["count"], 1) for c in clusters]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(names))
    # Scale bubble sizes
    max_count = max(counts)
    sizes = [(c / max_count) * 3000 + 200 for c in counts]
    x_pos = list(range(len(names)))
    y_pos = [0.5] * len(names)
    scatter = ax.scatter(x_pos, y_pos, s=sizes, c=colors, alpha=0.7, edgecolors="white", linewidth=2)
    for i, (name, count) in enumerate(zip(names, counts)):
        ax.annotate(f"{name}\n({count})", (x_pos[i], y_pos[i]),
                    ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-1, len(names))
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")
    ax.set_title("Topic Clusters", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_sources_table(sources_data):
    """Bar chart for source reliability."""
    if not sources_data:
        return
    df = pd.DataFrame(sources_data)
    if df.empty:
        return

    # Rating to numeric for plotting
    rating_map = {"High": 90, "Mostly Factual": 75, "Mixed": 50, "Low": 25}
    df["Score"] = df["Factual Rating"].map(lambda x: rating_map.get(x, 50))

    if df["Score"].sum() == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    color_map = {"High": "#2ecc71", "Mostly Factual": "#27ae60", "Mixed": "#f39c12", "Low": "#e74c3c"}
    colors = [color_map.get(r, "#95a5a6") for r in df["Factual Rating"]]
    bars = ax.bar(df["Domain"], df["Score"], color=colors)
    ax.set_ylabel("Reliability Score")
    ax.set_title("Source Reliability Ratings")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")

    # Add legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
#  DISPLAY REPORT
# ─────────────────────────────────────────────────────────────────
def display_report(report):
    """Display the news analysis report using all NewsAnalysisReport models."""
    if not report:
        st.error("No report to display")
        return

    # Plain string — just render as-is
    if isinstance(report, str):
        st.markdown("## Analysis Report")
        st.markdown(report)
        return

    try:
        _display_structured_report(report)
    except Exception as e:
        st.error(f"Error displaying report: {e}")
        st.markdown("## Raw Report")
        st.text(str(report)[:3000])


def _display_structured_report(report):
    """Render every section of NewsAnalysisReport with rich visualizations."""

    # ── Query Summary ──
    query_summary = _get(report, "query_summary", "News Analysis")
    st.title(f"📊 {query_summary}")

    # ── Key Findings ──
    key_findings = _get(report, "key_findings")
    if key_findings:
        st.header("🔍 Key Findings")
        st.markdown(key_findings)

    # ── Related Articles ──
    related_articles = _get(report, "related_articles", [])
    if related_articles:
        st.header("📰 Related Articles")
        articles_data = []
        for article in related_articles:
            articles_data.append({
                "Title": _get(article, "title", "N/A"),
                "Source": _get(article, "source", "N/A"),
                "Published": _get(article, "published_date", "N/A"),
                "URL": _get(article, "url", "#"),
            })
        df = pd.DataFrame(articles_data)
        st.dataframe(df, column_config={"URL": st.column_config.LinkColumn("URL")}, hide_index=True)

    # ── Related Keywords (Word Cloud) ──
    related_words = _get(report, "related_words", [])
    if related_words:
        st.header("🏷️ Related Keywords")
        _plot_wordcloud(related_words)

    # ── Topic Clusters (Bubble Chart) ──
    topic_clusters = _get(report, "topic_clusters", [])
    if topic_clusters:
        st.header("🗂️ Topic Clusters")
        cluster_data = []
        for cluster in topic_clusters:
            name = _get(cluster, "cluster_name", "Unknown Cluster")
            keywords = _get(cluster, "keywords", [])
            count = _get(cluster, "article_count", 0)
            cluster_data.append({"name": name, "count": count, "keywords": keywords})

        _plot_bubble_chart(cluster_data)

        # Also show details in expander
        with st.expander("Cluster Details"):
            for c in cluster_data:
                st.markdown(f"**{c['name']}** — {c['count']} articles")
                if c["keywords"]:
                    st.caption(", ".join(c["keywords"]))

    # ── Top Sources (Table + Bar Chart) ──
    top_sources = _get(report, "top_sources", [])
    if top_sources:
        st.header("📡 Top Sources")
        sources_data = []
        for source in top_sources:
            sources_data.append({
                "Domain": _get(source, "domain", "N/A"),
                "Factual Rating": _get(source, "factual_rating", "Mixed"),
                "Articles": _get(source, "articles_count", 0),
                "Engagement": _get(source, "engagement", 0),
            })
        df = pd.DataFrame(sources_data)
        st.dataframe(df, hide_index=True)
        _plot_sources_table(sources_data)

    # ── Top Hashtags (Table + Bar Chart) ──
    top_hashtags = _get(report, "top_hashtags", [])
    if top_hashtags:
        st.header("#️⃣ Top Hashtags")
        hashtag_data = []
        for ht in top_hashtags:
            if isinstance(ht, str):
                hashtag_data.append({
                    "Hashtag": ht, "Engagement Rate (%)": 0.0,
                    "Reach": 0, "Sentiment": "Neutral"
                })
            else:
                hashtag_data.append({
                    "Hashtag": _get(ht, "hashtag", "N/A"),
                    "Engagement Rate (%)": _get(ht, "engagement_rate", 0.0),
                    "Reach": _get(ht, "reach", 0),
                    "Sentiment": _get(ht, "sentiment", "Neutral"),
                })
        df = pd.DataFrame(hashtag_data)
        st.dataframe(df, hide_index=True)

        # Bar chart of engagement rates
        if any(d["Engagement Rate (%)"] > 0 for d in hashtag_data):
            _plot_horizontal_bar(
                [d["Hashtag"] for d in hashtag_data],
                [d["Engagement Rate (%)"] for d in hashtag_data],
                "Hashtag Engagement Rates",
                "Engagement Rate (%)",
                color_palette="magma"
            )

    # ── Similar Posts Time Series (Line Chart) ──
    time_series = _get(report, "similar_posts_time_series", [])
    if time_series:
        st.header("📈 Similar Posts Over Time")
        ts_data = []
        for entry in time_series:
            ts_data.append({
                "Date": _get(entry, "date", ""),
                "Count": _get(entry, "count", 0),
            })
        _plot_time_series(ts_data)

    # ── Fake News Sites ──
    fake_news_sites = _get(report, "fake_news_sites", [])
    if fake_news_sites:
        st.header("🚨 Flagged Fake News Sites")
        for site in fake_news_sites:
            st.markdown(f"- ⚠️ {site}")

    # ── Content Analysis Metrics (Percentage Bars) ──
    content_analysis = _get(report, "content_analysis")
    if content_analysis:
        st.header("📝 Content Analysis Metrics")
        metrics = {
            "Language": _get(content_analysis, "language_percentage", 0.0),
            "Coordination": _get(content_analysis, "coordination_percentage", 0.0),
            "Source": _get(content_analysis, "source_percentage", 0.0),
            "Bot-like Activity": _get(content_analysis, "bot_like_activity_percentage", 0.0),
        }
        # Metric cards
        cols = st.columns(4)
        for col, (label, val) in zip(cols, metrics.items()):
            with col:
                st.metric(label, f"{val:.1f}%")
        # Percentage bar chart
        _plot_percentage_bars(metrics)

    # ── Propaganda & Misinformation Analysis ──
    propaganda = _get(report, "propaganda_analysis")
    if propaganda:
        st.header("🛡️ Propaganda, Bots & Misinformation Analysis")

        # Reliability Score Gauge
        risk_score = _get(propaganda, "overall_reliability_score", 0)
        col1, col2 = st.columns([1, 3])
        with col1:
            _plot_donut(risk_score, max_val=100, label="Reliability",
                        color_high="#2ecc71", color_low="#e74c3c")
        with col2:
            if risk_score >= 70:
                st.success(f"✅ Reliability Score: **{risk_score}/100** — HIGH CREDIBILITY")
            elif risk_score >= 40:
                st.warning(f"⚡ Reliability Score: **{risk_score}/100** — MIXED CREDIBILITY")
            else:
                st.error(f"⚠️ Reliability Score: **{risk_score}/100** — LOW CREDIBILITY / HIGH RISK")

        # Propaganda Techniques (Table + Bar Chart)
        techniques = _get(propaganda, "propaganda_techniques", [])
        if techniques:
            st.subheader("Propaganda Techniques Detected")
            tech_data = []
            for tech in techniques:
                if isinstance(tech, str):
                    st.markdown(f"- 🔸 {tech}")
                else:
                    t_name = _get(tech, "technique_name", "Unknown")
                    t_freq = _get(tech, "frequency", 0)
                    t_sev = _get(tech, "severity", 0)
                    t_ex = _get(tech, "example", "")
                    t_exp = _get(tech, "explanation", "")
                    tech_data.append({
                        "Technique": t_name, "Frequency": t_freq,
                        "Severity": t_sev, "Example": t_ex
                    })
                    if t_exp:
                        with st.expander(f"🔸 {t_name} (Severity: {t_sev}/10)"):
                            if t_ex:
                                st.caption(f"Example: *{t_ex}*")
                            st.markdown(f"> {t_exp}")

            if tech_data:
                # Table
                st.dataframe(pd.DataFrame(tech_data), hide_index=True)
                # Severity bar chart
                _plot_horizontal_bar(
                    [d["Technique"] for d in tech_data],
                    [d["Severity"] for d in tech_data],
                    "Propaganda Technique Severity",
                    "Severity (0-10)",
                    color_palette="Reds"
                )

        # Misinformation Indicators (Table + Bar Chart)
        misinfo = _get(propaganda, "misinformation_indicators", [])
        if misinfo:
            st.subheader("Misinformation Indicators")
            misinfo_data = []
            for ind in misinfo:
                if isinstance(ind, str):
                    st.markdown(f"- 🔻 {ind}")
                else:
                    i_type = _get(ind, "indicator_type", "Unknown")
                    i_conf = _get(ind, "confidence", 0)
                    i_corr = _get(ind, "correction", "")
                    sources = _get(ind, "source_verification", [])
                    misinfo_data.append({
                        "Type": i_type,
                        "Confidence": f"{i_conf*100:.1f}%",
                        "Confidence_raw": i_conf,
                        "Correction": i_corr,
                        "Sources": ", ".join(sources) if sources else "—"
                    })

            if misinfo_data:
                st.dataframe(
                    pd.DataFrame(misinfo_data)[["Type", "Confidence", "Correction", "Sources"]],
                    hide_index=True
                )
                _plot_horizontal_bar(
                    [d["Type"] for d in misinfo_data],
                    [d["Confidence_raw"] * 100 for d in misinfo_data],
                    "Misinformation Indicator Confidence",
                    "Confidence (%)",
                    color_palette="YlOrRd"
                )

        # Coordination Patterns
        coordination = _get(propaganda, "coordination_patterns", [])
        if coordination:
            st.subheader("🕸️ Network Coordination")
            for coord in coordination:
                c_type = _get(coord, "pattern_type", "")
                c_str = _get(coord, "strength", 0)
                entities = _get(coord, "entities_involved", [])
                timeline = _get(coord, "timeline", "")
                st.markdown(f"**{c_type}** (Strength: {c_str*100:.1f}%)")
                if entities:
                    st.caption(f"Entities: {', '.join(entities)}")
                if timeline:
                    st.caption(f"Timeline: {timeline}")

        # Bot Activity Metrics (Donut)
        bot_metrics = _get(propaganda, "bot_activity_metrics", {})
        if bot_metrics:
            bot_score = _get(bot_metrics, "bot_likelihood_score", 0)
            if bot_score > 0:
                st.subheader("🤖 Bot Activity")
                col1, col2 = st.columns([1, 3])
                with col1:
                    _plot_donut(bot_score, max_val=1.0, label="Bot Likelihood")
                with col2:
                    if bot_score > 0.5:
                        st.warning(f"Bot Likelihood Score: {bot_score*100:.1f}%")
                    else:
                        st.info(f"Bot Likelihood Score: {bot_score*100:.1f}%")
                    patterns = _get(bot_metrics, "account_creation_patterns", "")
                    if patterns:
                        st.markdown(f"**Account Patterns:** {patterns}")
                    indicators = _get(bot_metrics, "behavioral_indicators", [])
                    for ind in indicators:
                        st.markdown(f"- {ind}")
                    network = _get(bot_metrics, "network_analysis", "")
                    if network:
                        st.markdown(f"**Network:** {network}")

        # Fake News Network Sites (Table + Bar Chart)
        fake_sites = _get(propaganda, "fake_news_sites", [])
        if fake_sites:
            st.subheader("🚨 Fake News Network Sites")
            site_data = []
            for site in fake_sites:
                domain = _get(site, "domain", "Unknown")
                shares = _get(site, "shares", 0)
                engagement = _get(site, "engagement", 0)
                false_stories = _get(site, "known_false_stories", 0)
                site_data.append({
                    "Domain": domain, "Shares": shares,
                    "Engagement": engagement, "False Stories": false_stories
                })
            df = pd.DataFrame(site_data)
            st.dataframe(df, hide_index=True)
            if any(d["Shares"] > 0 for d in site_data):
                _plot_horizontal_bar(
                    [d["Domain"] for d in site_data],
                    [d["Shares"] for d in site_data],
                    "Fake News Site Shares",
                    "Number of Shares",
                    color_palette="Reds"
                )

            # Deceptive practices
            with st.expander("Deceptive Practices by Domain"):
                for site in fake_sites:
                    domain = _get(site, "domain", "Unknown")
                    practices = _get(site, "deceptive_practices", [])
                    if practices:
                        st.markdown(f"**{domain}**:")
                        for p in practices:
                            st.markdown(f"- {p}")

        # Manipulation Timeline
        manip_timeline = _get(propaganda, "manipulation_timeline", [])
        if manip_timeline:
            st.subheader("📅 Information Manipulation Timeline")
            for entry in manip_timeline:
                if isinstance(entry, dict):
                    date = entry.get("date", "N/A")
                    event = entry.get("event", "N/A")
                    st.markdown(f"- **{date}**: {event}")
                else:
                    st.markdown(f"- {entry}")

        # Narrative Fingerprint (Bar Chart)
        fingerprint = _get(propaganda, "narrative_fingerprint", {})
        if fingerprint:
            st.subheader("🧬 Narrative Fingerprint")
            _plot_horizontal_bar(
                list(fingerprint.keys()),
                [v * 100 for v in fingerprint.values()],
                "Narrative Pattern Strength",
                "Strength (%)",
                color_palette="coolwarm"
            )

        # Verification Steps
        steps = _get(propaganda, "recommended_verification_steps", [])
        if steps:
            st.subheader("✅ How to Verify This Information")
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")

    # ── Platform Facts ──
    platform_facts = _get(report, "platform_facts", [])
    if platform_facts:
        st.header("🌐 Platform Facts")
        for fact in platform_facts:
            st.markdown(f"- {fact}")

    # ── Cross-Source Facts ──
    cross_source_facts = _get(report, "cross_source_facts", [])
    if cross_source_facts:
        st.header("✅ Cross-Source Verified Facts")
        for fact in cross_source_facts:
            st.markdown(f"- {fact}")

    # ── Analysis Note ──
    analysis_note = _get(report, "analysis_note", "")
    if analysis_note and analysis_note != "No specific notes.":
        st.info(f"📝 **Note:** {analysis_note}")


# ─── Multimodal News Analysis Pipeline ───

def analyze_multimodal_news(uploaded_image=None, heard_text="", article_urls="",
                            custom_keywords="", llm_provider="gemini", model_name=None):
    """Analyze news content from screenshots/images, heard text, and optional article links."""
    image_data = {}
    url_previews = []

    # 1. Process uploaded screenshot / image
    if uploaded_image is not None:
        with st.spinner("🔍 Inspecting image with Gemini Vision (OCR & OSINT analysis)..."):
            image_bytes = uploaded_image.getvalue()
            mime_type = uploaded_image.type or "image/jpeg"
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            image_data = extract_content_from_image(
                image_input=image_bytes,
                mime_type=mime_type,
                gemini_api_key=gemini_key,
                model_name=model_name
            )

        if image_data.get("error") and not image_data.get("transcribed_text"):
            st.warning(f"⚠️ Image processing note: {image_data['error']}")
        else:
            st.subheader("📸 Screenshot / Image Intelligence")
            col_img1, col_img2 = st.columns([1, 2])
            with col_img1:
                st.image(uploaded_image, caption="Uploaded Evidence", use_container_width=True)

            with col_img2:
                platform = image_data.get("detected_platform", "Unknown")
                attribution = image_data.get("source_attribution", "Unknown")
                core_claim = image_data.get("core_claim", "")
                headline = image_data.get("headline", "")

                st.markdown(f"**Detected Platform:** `{platform}`")
                if attribution and attribution != "Unknown":
                    st.markdown(f"**Source Attribution:** `{attribution}`")
                if headline:
                    st.markdown(f"**Visible Headline:** {headline}")
                if core_claim:
                    st.markdown(f"**Extracted Core Claim:** {core_claim}")

                # Visual authenticity notes
                auth_notes = image_data.get("visual_authenticity_notes", "")
                if auth_notes:
                    st.info(f"🛡️ **Visual Authenticity Cues:** {auth_notes}")

                # Metrics if visible
                det_metrics = image_data.get("detected_metrics", {})
                active_metrics = {k: v for k, v in det_metrics.items() if v is not None}
                if active_metrics:
                    metric_cols = st.columns(len(active_metrics))
                    for idx, (m_key, m_val) in enumerate(active_metrics.items()):
                        with metric_cols[idx]:
                            st.metric(m_key.replace("_", " ").title(), str(m_val))

            if image_data.get("transcribed_text"):
                with st.expander("📄 Verbatim Transcribed Text from Image"):
                    st.text(image_data["transcribed_text"])

    # 2. Process news article links (optional)
    raw_urls = [u.strip() for u in article_urls.split("\n") if u.strip()] if article_urls else []
    if raw_urls:
        with st.spinner("🌐 Fetching previews for referenced news articles..."):
            for url in raw_urls:
                prev = fetch_url_preview(url)
                url_previews.append(prev)

        st.subheader("📰 Referenced News Articles")
        for prev in url_previews:
            with st.container():
                st.markdown(f"**[{prev['title']}]({prev['url']})** (`{prev['domain']}`)")
                st.caption(prev["snippet"])

    # 3. Display user-provided heard news text
    if heard_text.strip():
        st.subheader("🗣️ Reported News / Rumor Text")
        st.info(f'"{heard_text.strip()}"')

    # 4. Synthesize claim, keywords, and platform metrics
    synthesis = synthesize_claim(
        heard_text=heard_text,
        image_data=image_data,
        url_previews=url_previews
    )

    # Incorporate custom keywords if provided
    if custom_keywords.strip():
        user_kw_list = [k.strip() for k in custom_keywords.split(",") if k.strip()]
        for ukw in user_kw_list:
            if ukw.lower() not in [k["text"].lower() for k in synthesis["keywords"]]:
                synthesis["keywords"].insert(0, {"text": ukw, "frequency": 2})
                synthesis["keyword_list"].insert(0, ukw)

    user_query = synthesis["user_query"]
    keywords = synthesis["keywords"]
    platform_metrics = synthesis["platform_metrics"]
    urls_for_analysis = synthesis["url_list"]

    st.subheader("🎯 Synthesized Verification Target")
    st.markdown(f"**Investigation Query:** `{user_query}`")

    if keywords:
        # Keywords are available but not displayed in the UI per user request
        pass

    # 5. Fetch Google Trends data for the synthesized claim (Tier 2)
    with st.spinner("Fetching Google Trends data..."):
        try:
            trends_query = keywords[0]["text"] if keywords else user_query
            trends_metrics = get_trends_metrics(trends_query)
            if trends_metrics:
                platform_metrics.update(trends_metrics)
                ts_count = len(trends_metrics.get("trends_time_series", []))
                if ts_count:
                    st.caption(f"📈 Google Trends: {ts_count} data points retrieved for '{trends_query}'")
        except Exception as e:
            logger.warning("Trends fetch failed: %s", e)

    # 6. Run multi-agent news verification
    st.info(f"🚀 Starting multi-agent investigation for: **{user_query}**")

    top_keywords_for_agents = [k["text"] for k in keywords[:5]]
    with st.spinner("Running in-depth news verification... This may take several minutes."):
        report = run_news_analysis(
            user_query=user_query,
            urls=urls_for_analysis if urls_for_analysis else None,
            keywords=top_keywords_for_agents if top_keywords_for_agents else None,
            llm_provider=llm_provider,
            model_name=model_name,
            platform_metrics=platform_metrics,
        )

    return report


def analyze_reddit_post(url, llm_provider="gemini", model_name=None):
    """Legacy helper: Analyze a Reddit post."""
    try:
        from reddit import scrape_reddit_data, extract_keywords, compute_reddit_engagement
        with st.spinner("Scraping Reddit post..."):
            reddit_data = scrape_reddit_data(url)

        if "error" in reddit_data:
            st.error(f"Error: {reddit_data['error']}")
            return None

        keywords = extract_keywords(reddit_data)
        keyword_list = [kw['text'] for kw in keywords]
        platform_metrics = compute_reddit_engagement(reddit_data)
        user_query = f"News analysis for: {reddit_data['title']}"

        return run_news_analysis(
            user_query=user_query,
            keywords=keyword_list[:5],
            llm_provider=llm_provider,
            model_name=model_name,
            platform_metrics=platform_metrics,
        )
    except Exception as e:
        st.error(f"Reddit analysis failed: {e}")
        return None


def main():
    try:
        st.title("VerifAI: News & Misinformation Verification")
        st.caption("AI-powered multi-agent fact-checking across screenshots, rumors, and news articles")

        # Sidebar
        st.sidebar.header("About VerifAI")
        st.sidebar.markdown(
            """
            **VerifAI** is an advanced news credibility and misinformation verification platform.
            
            ### 📥 Supported Input Sources:
            - 📸 **Screenshots & Images**: Tweet screenshots, WhatsApp forwards, headlines, or infographics
            - 🗣️ **Heard News / Rumors**: Claims or stories you've heard from peers or social channels
            - 🔗 **Article Links (Optional)**: News URLs to cross-verify against authoritative reporting
            
            Multi-agent AI investigates source credibility, propaganda patterns, bot coordinated narratives, and factual veracity.
            """
        )

        # API key + model setup in the sidebar
        with st.sidebar.expander("⚙️ Configuration", expanded=True):
            from setup import GEMINI_MODELS, DEFAULT_GEMINI_MODEL
            selected_model = st.selectbox(
                "Gemini Model",
                options=list(GEMINI_MODELS.keys()),
                index=list(GEMINI_MODELS.keys()).index(DEFAULT_GEMINI_MODEL),
                key="gemini_model_selection",
                help="Choose which Gemini model to use for analysis and vision",
            )
            st.session_state["gemini_model"] = selected_model

            current_gemini_key = os.environ.get("GEMINI_API_KEY", "")
            gemini_api_key = st.text_input(
                "Gemini API Key",
                value=current_gemini_key,
                type="password",
                help="Required for LLM and image inspection. Get one at https://aistudio.google.com/app/apikey",
                key="gemini_api_key_input",
            )

            current_serper_key = os.environ.get("SERPER_API_KEY", "")
            serper_api_key = st.text_input(
                "Serper API Key",
                value=current_serper_key if current_serper_key != "dummy-key-for-ollama" else "",
                type="password",
                help="Required for real-time web search and fact verification",
                key="serper_api_key_1",
            )

            current_serpapi_key = os.environ.get("SERPAPI_API_KEY", "")
            serpapi_api_key = st.text_input(
                "SerpAPI Key (Optional)",
                value=current_serpapi_key,
                type="password",
                help="Used for Google Trends trendlines. Get one at https://serpapi.com",
                key="serpapi_api_key_input",
            )

            if st.button("Save API Keys", key="save_api_keys_1"):
                if gemini_api_key.strip():
                    os.environ["GEMINI_API_KEY"] = gemini_api_key.strip()
                    os.environ["GOOGLE_API_KEY"] = gemini_api_key.strip()
                    st.success("Gemini API key saved!")
                if serper_api_key.strip():
                    os.environ["SERPER_API_KEY"] = serper_api_key.strip()
                    st.success("Serper API key saved!")
                if serpapi_api_key.strip():
                    os.environ["SERPAPI_API_KEY"] = serpapi_api_key.strip()
                    st.success("SerpAPI key saved!")

        # Check LLM status
        llm_status, llm_msg = check_llm_status()
        if llm_status:
            st.sidebar.success(f"✅ {llm_msg}")
        else:
            st.sidebar.error(f"❌ {llm_msg}")

        gemini_model = st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
        llm_provider = "gemini"

        # ─── New Multimodal Input Dashboard ───
        st.header("📥 Submit News Content for Fact-Checking")
        st.markdown(
            "Provide any combination of inputs below: a screenshot or image, news text you've heard, and optional article links."
        )

        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            st.markdown("#### 📸 1. Screenshot or Image")
            uploaded_image = st.file_uploader(
                "Upload a news screenshot, social post, or graphic:",
                type=["png", "jpg", "jpeg", "webp"],
                help="Upload a screenshot of a tweet, WhatsApp forward, news headline, or infographic",
                key="uploaded_image_file"
            )
            if uploaded_image:
                st.image(uploaded_image, caption=f"Selected: {uploaded_image.name}", use_container_width=True)

        with col_right:
            st.markdown("#### 🗣️ 2. News Text You've Heard")
            heard_text = st.text_area(
                "News claim, rumor, or spoken text:",
                placeholder="e.g. 'I heard that the central bank is recalling all 500 currency notes by next week' or paste viral forwarded text...",
                height=130,
                key="heard_text_input",
                help="Type or paste any rumor, headline, or claim you've heard"
            )

            st.markdown("#### 🔗 3. News Article Links (Optional)")
            article_urls_input = st.text_area(
                "News article links to cross-verify (one per line):",
                placeholder="https://example.com/article1\nhttps://reuters.com/article2",
                height=80,
                key="article_urls_input",
                help="Optional: Paste news article URLs to cross-verify against"
            )

            with st.expander("⚙️ Additional Keywords (Optional)"):
                custom_keywords_input = st.text_input(
                    "Custom keywords (comma-separated):",
                    placeholder="e.g. banking, RBI, currency reform",
                    key="custom_keywords_input"
                )

        st.markdown("---")
        verify_clicked = st.button(
            "🔍 Verify & Analyze News Claim",
            type="primary",
            use_container_width=True,
            key="verify_news_claim_btn"
        )

        if verify_clicked:
            # Validate input presence
            has_image = uploaded_image is not None
            has_heard = bool(heard_text.strip())
            has_urls = bool(article_urls_input.strip())

            if not has_image and not has_heard and not has_urls:
                st.error("⚠️ Please provide at least one input source: an uploaded screenshot/image, news text you've heard, or article links.")
            elif not setup_api_keys():
                st.error("API keys not configured. Please set valid Gemini and Serper API keys in the sidebar.")
            else:
                try:
                    report = analyze_multimodal_news(
                        uploaded_image=uploaded_image,
                        heard_text=heard_text,
                        article_urls=article_urls_input,
                        custom_keywords=custom_keywords_input,
                        llm_provider=llm_provider,
                        model_name=gemini_model
                    )

                    if report:
                        st.session_state["report"] = report
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("Failed to generate report.")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("Check your API keys and network connection.")

        # Display the report from session state (persists across reruns)
        if st.session_state.get("report"):
            st.divider()
            display_report(st.session_state["report"])

            st.divider()
            markdown_report = get_report_as_markdown(st.session_state["report"])
            st.download_button(
                label="📥 Download Report as Markdown",
                data=markdown_report,
                file_name=f"news_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_report_btn"
            )

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()

