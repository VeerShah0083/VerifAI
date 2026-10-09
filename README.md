# VerifAI

VerifAI is an AI-powered platform that analyzes news, social media, and online content for accuracy, bias, and credibility. It leverages Google Gemini models, real-time web scraping, and data visualization to help users detect misinformation, propaganda, and coordinated inauthentic behavior.

## Features

- **Multimodal News Fact-Checking**: Verify news claims from screenshots, viral social graphics, rumors you've heard, and reference news links.
- **Visual & OCR Intelligence**: Extract verbatim text, headlines, platforms (X/Twitter, WhatsApp, etc.), and visual authenticity markers from images.
- **Misinformation & Propaganda Detection**: Identify propaganda techniques, misinformation indicators, and fake news sites.
- **Cross-Source Verification**: Scrape and cross-verify optional news article URLs with multi-agent web investigation.
- **Social Media Tracking**: Track news spread, top hashtags, engagement, and sentiment across social platforms.
- **Data Visualization**: Visualize topic clusters, word clouds, time series, and reliability metrics.
- **Comprehensive Reporting**: Generate detailed, structured reports in Markdown or view interactively via Streamlit.

## Project Structure

- `app.py` — Core logic for news analysis, agent/task orchestration, and report generation.
- `streamlit.py` — Streamlit web interface for interactive multimodal fact-checking and visualization.
- `content_extractor.py` — Gemini Vision multimodal extraction, URL preview fetching, keyword extraction, and input synthesis.
- `requirements.txt` — Python dependencies.
- `db/` — Local database and cache files (auto-generated).

## Setup

### 1. Clone the Repository

```sh
# Use PowerShell or your preferred terminal
cd <your-folder>
git clone <repo-url>
cd VerifAI
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure API Keys

You will need:

- **Google Gemini** API key (for LLM and multimodal image inspection — supports Gemini 2.5 Flash and Gemini 3 Flash)
- **Serper** API key (for real-time web search and fact verification)
- **SerpAPI** key (optional, for Google Trends data)

You can set these as environment variables or enter them directly via the Streamlit sidebar.

#### Example `.env` file:

```
GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key
SERPAPI_API_KEY=your_serpapi_key
```

## Usage

### Command-Line Interface

Run the main analysis tool:

```sh
python app.py
```

You will be prompted for:

- News topic to analyze
- (Optional) News URLs, hashtags, or keywords
- API keys (if not set in environment)

A Markdown report will be generated and saved as `news_analysis_report.md`.

### Streamlit Web App

Launch the interactive web interface:

```sh
streamlit run streamlit.py
```

- Upload a screenshot or image (PNG, JPG, WEBP).
- Type or paste news text / rumors you've heard.
- (Optional) Paste news article links to cross-verify.
- View real-time visual analysis, keyword extraction, and comprehensive fact-check reports.
- Configure API keys in the sidebar.

## Output

- **Markdown Report**: Detailed news analysis, key findings, source reliability, propaganda detection, and more.
- **Interactive Visualizations**: Topic clusters, word clouds, time series, and reliability charts (Streamlit UI).

## Advanced

- The system uses modular agents for crawling, content analysis, social tracking, visualization, and misinformation detection.
- Easily extendable for new data sources or analysis tasks.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## License

This project is for research and educational purposes.

---

_For questions or contributions, please open an issue or pull request._
