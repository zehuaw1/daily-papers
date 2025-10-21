# ==============================================================================
# ArXiv Pusher Configuration Example
# Copy this file to config.py and fill in your details
# ==============================================================================

AI_CONFIG = {
    "api_key": "sk-your-deepseek-api-key",  # Get from https://platform.deepseek.com
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",  # Non-thinking mode (cheapest)

    # Pricing for cost tracking
    "price_per_million_input_tokens": 0.56,
    "price_per_million_output_tokens": 1.68,
}

EMAIL_SERVER_CONFIG = {
    "sender": "your_email@gmail.com",
    "password": "your_app_specific_password",  # Use app-specific password for Gmail
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "use_tls": True,
}

GENERAL_CONFIG = {
    "days_lookback": 1,
    "max_papers_per_user": 10,  # Process max 10 papers per user
    "max_text_length": 100000,

    # Social Feed Integration (FREE)
    "enable_huggingface": True,  # Fetch papers from HuggingFace
    "huggingface_sort": "trending",  # Sorting: "trending" or None for daily papers
}

DEFAULT_FILTER_PROMPT = """Determine if the following paper is relevant.

Abstract:
{abstract}

Respond with ONLY "yes" or "no"."""

DEFAULT_SUMMARY_PROMPT = """Summarize this academic paper concisely.

Include key contributions, methodology, and results.

Paper:
{text}

Use markdown formatting."""

DEFAULT_RANKING_PROMPT = """You are an AI research assistant helping to prioritize papers for detailed review.

Based on the user's research interests below, select the TOP {max_papers} most valuable and relevant papers from the list.

**User's Research Interests:**
{interests}

**Available Papers ({num_papers} total):**

{papers_list}

**Task:**
Select the {max_papers} most valuable papers based on:
1. Relevance to the user's research interests
2. Novelty and potential impact
3. Social engagement (upvotes, stars) as a secondary signal
4. Methodological rigor and contribution

**Instructions:**
First, reason through your selection process. Explain why you're choosing certain papers.
Then, on the LAST LINE of your response, output the selected paper numbers as a comma-separated list.

**Format:**
[Your reasoning here...]

SELECTED: 1,5,12,3,7"""

USERS_CONFIG = [
    {
        "name": "John Doe",
        "email": "john@example.com",
        "arxiv_categories": ["cs.LG", "cs.AI"],

        # Optional: Custom filter and summary prompts
        # "filter_prompt": "...",
        # "summary_prompt": "...",

        # Optional: Override global social feed settings per user
        # "enable_huggingface": True,
        # "huggingface_sort": "trending",  # or None for daily papers

        # Optional: Custom prompts per user (override defaults)
        # "ranking_prompt": DEFAULT_RANKING_PROMPT,
    },
]
