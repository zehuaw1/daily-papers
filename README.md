# Daily Papers

Automatically fetch, filter, and summarize the latest arXiv papers using LLMs, then deliver personalized daily reports to your inbox.

**Refactored Version**: Clean, minimal, and fully functional with English prompts and modern best practices.

## ✨ Features

### Core Functionality
- **Multi-Source Fetching** ⭐: Automatically discover papers from:
  - arXiv (by category)
  - HuggingFace Daily Papers (trending/popular)
  - Papers with Code (community favorites)
- **AI-Powered Filtering** ⭐: Smart relevance filtering based on your research interests
- **Deep Summarization**: LLM-generated summaries including:
  - Key contributions
  - Methodology overview
  - Experimental results
  - Significance and impact
- **Email Delivery**: Markdown-formatted reports sent directly to your inbox
- **Social Metrics** ⭐: See upvotes, GitHub stars, and trending indicators
- **Review Appendix** ⭐: Filtered papers included for reference

### Advanced Features
- **Multi-User Support**: Configure multiple users with independent settings
- **Flexible Configuration**: Customize everything from a single `config.py` file
  - arXiv categories per user
  - Enable/disable social feeds globally or per user
  - Custom filter prompts (define your interests)
  - Custom summary prompts (tailor the output format)
  - Papers per user limit
  - Lookback days
- **Intelligent Paper Ranking** ⭐: When there are more papers than your limit, the LLM automatically selects the most valuable ones based on your research interests (see [FEATURES.md](assets/FEATURES.md))
- **Smart Deduplication**: Automatically merges papers from all sources, preferring social versions (with metrics)
- **Local Reports**: Markdown files saved for each user
- **Scheduled Execution**: Automated daily runs with `apscheduler`
- **Comprehensive Logging**: Detailed logs with `loguru`
- **Cost Tracking**: Token usage and cost estimates (including ranking)

## 📊 Comparison with Popular Projects

| Feature | ArXiv Pusher | arxiv-sanity | daily-arXiv-ai |
|---------|--------------|--------------|----------------|
| Multi-user | ✅ | ❌ | ❌ |
| Trending Papers | ✅ (HuggingFace) | ❌ | ❌ |
| AI Filtering | ✅ | ❌ | ✅ |
| Custom Prompts | ✅ | ❌ | Limited |
| Email Delivery | ✅ | ✅ | ❌ |
| Minimal Setup | ✅ | ❌ | ✅ |
| Cost Tracking | ✅ | ❌ | ❌ |

## 🌐 Social Feed Integration

ArXiv Pusher fetches papers from **multiple sources in parallel**, then merges and deduplicates them:

```
arXiv Categories    ↘
                     → Merge & Dedupe → AI Filter → Summarize → Email
HuggingFace Papers  ↗
```

### Supported Sources

1. **HuggingFace Papers** (FREE)
   - Community-curated trending or daily papers
   - Includes upvote counts and GitHub stars
   - API: `https://huggingface.co/api/daily_papers`
   - Sorting options: `trending` (default) or daily

2. **arXiv Categories** (FREE)
   - Direct arXiv API access
   - Filter by research categories
   - Most comprehensive source

### Configuration

Enable/disable HuggingFace and choose sorting:
```python
GENERAL_CONFIG = {
    "enable_huggingface": True,
    "huggingface_sort": "trending",  # or None for daily papers
}
```

Or per user:
```python
USERS_CONFIG = [
    {
        "name": "ML Researcher",
        "email": "researcher@example.com",
        "arxiv_categories": ["cs.LG"],
        "enable_huggingface": True,  # Override global setting
        "huggingface_sort": "trending",  # or None
    },
]
```

### How Deduplication Works

- Papers are merged by arXiv ID
- **HuggingFace versions are preferred** when duplicates exist (they have upvotes/stars)
- Reports show source and social metrics for each paper

## ⚙️ Installation

### 1. Prerequisites
- Python 3.10+ (recommended)
- Conda (miniconda or anaconda) - [Install here](https://docs.conda.io/en/latest/miniconda.html)
- DeepSeek API key (or compatible OpenAI-format API)

### 2. Clone Repository
```bash
git clone <your-repository-url>
cd daily-papers
```

### 3. Setup Environment

**Option A: Using Conda (Recommended)**
```bash
# Create environment
conda create -n paper python=3.10 -y
conda activate paper

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using pip**
```bash
pip install -r requirements.txt
```

### 4. Configure
```bash
cp config.example.py config.py
# Edit config.py with your API keys and preferences
```

### 5. Test the Setup
```bash
# Make sure you're in the paper environment (if using conda)
conda activate paper

# Run once to test
python3 main.py
```

## 🛠️ Configuration

Edit `config.py` to customize:

```python
# Minimal configuration example
AI_CONFIG = {
    "api_key": "your-deepseek-api-key",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "price_per_million_input_tokens": 0.56,   # For cost tracking
    "price_per_million_output_tokens": 1.68,
}

EMAIL_SERVER_CONFIG = {
    "sender": "your_email@gmail.com",
    "password": "your_app_password",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": True,
}

GENERAL_CONFIG = {
    "days_lookback": 1,              # Days to look back
    "max_papers_per_user": 10,       # Max papers to process (None = unlimited)
    "max_text_length": 100000,       # Max PDF text length

    # Social Feed Integration
    "enable_huggingface": True,      # Fetch from HuggingFace
    "huggingface_sort": "trending",  # "trending" or None for daily
}

# Note: When papers > max_papers_per_user, LLM automatically ranks and selects
# the most valuable papers based on your research interests!

USERS_CONFIG = [
    {
        "name": "ML Researcher",
        "email": "researcher@example.com",
        "arxiv_categories": ["cs.LG", "cs.AI"],

        # Optional custom prompts
        "filter_prompt": "Is this about deep learning? Answer yes/no.\n{abstract}",
        "summary_prompt": "Summarize concisely:\n{text}",
    },
]
```

### arXiv Categories Reference
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CV` - Computer Vision
- `cs.CL` - Natural Language Processing
- `cs.RO` - Robotics

[Full list →](https://arxiv.org/category_taxonomy)

## 🚀 Usage

### Option 1: GitHub Actions (Recommended)

**Run automatically on GitHub for FREE** - no server needed!

See [GITHUB_ACTIONS_SETUP.md](assets/GITHUB_ACTIONS_SETUP.md) for detailed instructions.

Quick setup:
1. Push code to GitHub
2. Add 5 secrets in repo settings (API keys, email credentials)
3. Configure schedule in `.github/workflows/daily-arxiv.yml`
4. Done! Reports will be emailed automatically

Benefits:
- ✅ Completely free (2000 min/month for private repos)
- ✅ No server maintenance
- ✅ Automatic logging
- ✅ Secure credential storage

### Option 2: Run Once (Local Test)
```bash
python3 main.py
```

### Option 3: Run Daily Scheduler (Local)
Edit `main.py`:
```python
if __name__ == "__main__":
    # Comment out immediate run
    # daily_job()

    # Enable scheduler
    run_scheduler()  # Runs daily at 4 PM
```

Customize schedule:
```python
trigger=CronTrigger(hour=16, minute=0)  # 4:00 PM daily
```

Keep script running:
```bash
# Using nohup
nohup python3 main.py > arxiv_pusher.log 2>&1 &

# Or using screen
screen -S arxiv
python3 main.py
# Ctrl+A then D to detach
```

## 💰 Cost Estimates

Using **DeepSeek V3** (cheapest option):

| Users | Papers/Day | Filtering | Summary | **Total/Day** |
|-------|------------|-----------|---------|---------------|
| 1     | 10         | $0.03     | $0.45   | **~$0.50**    |
| 1     | 50         | $0.03     | $2.25   | **~$2.30**    |
| 3     | 10 each    | $0.09     | $1.35   | **~$1.50**    |

**Monthly cost**: $15-70 depending on usage

### Alternative APIs
- **DeepSeek V3.2-Exp**: Even cheaper (~$0.30/day for 10 papers)
- **Qwen-Turbo**: ~$0.40/day for 10 papers
- **GLM-4-Long**: ~$0.30/day for 10 papers
- **Gemini 2.0 Flash**: Free tier available

## 📄 Output Format

### Email Report Structure
```markdown
## 📄 Paper Title Here

**Authors**: Author1, Author2
**Published**: 2025-10-15
**Link**: https://arxiv.org/abs/xxxx.xxxxx
**Category**: cs.LG
**Source**: HuggingFace | 👍 23 upvotes | ⭐ 156 GitHub stars

**Abstract**:
[Original abstract]

**Summary**:
[AI-generated summary with key points]

────────────────────────────────────────

## 📋 Appendix: Filtered Papers
[Papers that didn't pass interest filter]
```

## 🔧 Advanced Usage

### Custom Filter Prompts

**Broad filtering**:
```python
"filter_prompt": """Is this paper about machine learning?

Abstract: {abstract}

Answer: yes or no"""
```

**Narrow filtering**:
```python
"filter_prompt": """Does this paper focus on reinforcement learning
with policy gradient methods or actor-critic algorithms?

Abstract: {abstract}

Answer: yes if directly relevant, no otherwise"""
```

### Custom Summary Prompts

**Detailed analysis**:
```python
"summary_prompt": """Provide detailed analysis:
1. Main contributions (3-5 points)
2. Technical approach
3. Experimental setup and results
4. Limitations and future work

Paper: {text}"""
```

**Concise summary**:
```python
"summary_prompt": """Summarize in 200 words or less.

Paper: {text}"""
```

## 📁 Code Structure

```
Daily-Papers/
├── main.py              # Main application (refactored, ~500 lines)
├── config.py            # User configuration
├── config.example.py    # Configuration template
├── temp/                # Temporary files (PDFs, reports)
│   └── user_name/
│       ├── *.pdf
│       └── report.md
└── arxiv_pusher.log     # Application logs
```

## 🎯 Key Improvements

This refactored version features:
1. ✅ **English prompts** throughout
2. ✅ **Minimal codebase** (~500 lines vs ~670)
3. ✅ **Modular design** with clear separation of concerns
4. ✅ **Better error handling**
5. ✅ **Cleaner configuration** - everything in one place
6. ✅ **Cost tracking** with detailed logging
7. ✅ **Simplified PDF handling** (removed unused HTML extraction)

## 🙏 Acknowledgments

Inspired by:
- [arxiv-sanity](https://github.com/karpathy/arxiv-sanity-preserver) by Andrej Karpathy
- [daily-arXiv-ai-enhanced](https://github.com/dw-dengwei/daily-arXiv-ai-enhanced)
- [customize-arxiv-daily](https://github.com/JoeLeelyf/customize-arxiv-daily)
- [arxiv-pusher](https://github.com/ZhuYizhou2333/ArXiv-Pusher)

---

**Questions?** Open an issue or check the [arXiv API docs](https://info.arxiv.org/help/api/index.html)
