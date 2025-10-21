# Daily Papers

Automatically fetch, filter, and summarize the latest arXiv papers using LLMs, then deliver personalized daily reports to your inbox.

## ✨ Features

- **Multi-Source Fetching**: Papers from arXiv categories and HuggingFace (trending/daily)
- **AI-Powered Filtering**: Smart relevance filtering based on your research interests
- **Intelligent Ranking**: LLM automatically selects the most valuable papers when count exceeds limit
- **Deep Summarization**: LLM-generated summaries with key contributions, methodology, and results
- **Email Delivery**: Markdown-formatted reports sent directly to your inbox
- **Social Metrics**: View upvotes, GitHub stars, and trending indicators from HuggingFace
- **Multi-User Support**: Configure multiple users with independent settings
- **Cost Tracking**: Token usage and cost estimates

## ⚙️ Quick Start

### 1. Install
```bash
git clone <your-repository-url>
cd daily-papers
pip install -r requirements.txt
```

### 2. Configure
```bash
cp config.example.py config.py
# Edit config.py with your API keys and settings
```

**Required settings in `config.py`:**
- DeepSeek API key (or compatible OpenAI-format API)
- Email SMTP credentials
- User preferences (arXiv categories, prompts)

### 3. Run
```bash
python3 main.py  # Run once
```

## 🛠️ Configuration

Key settings in `config.py`:

```python
AI_CONFIG = {
    "api_key": "your-deepseek-api-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
}

GENERAL_CONFIG = {
    "days_lookback": 1,
    "max_papers_per_user": 10,       # LLM ranks and selects best papers
    "enable_huggingface": True,      # Fetch trending papers
    "huggingface_sort": "trending",  # or None for daily papers
}

USERS_CONFIG = [
    {
        "name": "ML Researcher",
        "email": "researcher@example.com",
        "arxiv_categories": ["cs.LG", "cs.AI"],
        # Optional: "filter_prompt", "summary_prompt", "ranking_prompt"
    },
]
```

**Common arXiv categories**: `cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`, `cs.RO` ([full list](https://arxiv.org/category_taxonomy))

## 🚀 Usage

### Option 1: GitHub Actions (Recommended)
Run automatically on GitHub for FREE - no server needed!

1. Push code to GitHub
2. Add secrets in repo settings (API keys, email credentials)
3. Configure schedule in `.github/workflows/daily-arxiv.yml`

See [GITHUB_ACTIONS_SETUP.md](assets/GITHUB_ACTIONS_SETUP.md) for details.

### Option 2: Local Scheduler
Enable scheduler in `main.py`:
```python
if __name__ == "__main__":
    # daily_job()      # Disable immediate run
    run_scheduler()    # Enable daily at 9 AM
```

Run in background:
```bash
nohup python3 main.py > arxiv_pusher.log 2>&1 &
```

## 💰 Cost Estimates

**DeepSeek V3**: ~$0.50/day for 10 papers (~$15/month)

Alternative APIs: DeepSeek V3.2-Exp, Qwen-Turbo, GLM-4-Long (~$0.30-0.40/day), Gemini 2.0 Flash (free tier)

## 📄 Output Example

Each paper includes:
- Title, authors, publication date, arXiv link
- Source (arXiv/HuggingFace) and social metrics (upvotes, stars)
- Original abstract
- AI-generated summary (contributions, methodology, results)
- Appendix with filtered papers for reference

Reports are emailed as formatted HTML and saved as markdown in `temp/{user_name}/report.md`

## 🙏 Acknowledgments

Inspired by:
- [arxiv-sanity](https://github.com/karpathy/arxiv-sanity-preserver) by Andrej Karpathy
- [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced)
- [customize-arxiv-daily](https://github.com/JoeLeelyf/customize-arxiv-daily)
- [arxiv-pusher](https://github.com/ZhuYizhou2333/ArXiv-Pusher)

---

**Questions?** Open an issue or check the [arXiv API docs](https://info.arxiv.org/help/api/index.html)
