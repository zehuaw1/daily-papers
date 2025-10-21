"""
ArXiv Paper Pusher - Automatically fetch, filter, and summarize arXiv papers
"""
import os
import requests
import smtplib
import socket
import asyncio
import time
import re
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed

from arxiv import Client, Search, SortCriterion, SortOrder
from PyPDF2 import PdfReader
import openai
import markdown2
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    AI_CONFIG,
    EMAIL_SERVER_CONFIG,
    GENERAL_CONFIG,
    USERS_CONFIG,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_FILTER_PROMPT,
    DEFAULT_RANKING_PROMPT
)



# ============================================================================
# Email Functions
# ============================================================================

async def send_email(subject, content, receiver_email):
    """Send email notification (async version)"""
    html_content = markdown2.markdown(
        content,
        extras=["tables", "mathjax", "fenced-code-blocks"]
    )
    msg = MIMEText(html_content, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SERVER_CONFIG["sender"]
    msg["To"] = receiver_email

    try:
        logger.info(f"Connecting to SMTP server, sending to {receiver_email}...")
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: _send_email_sync(msg, receiver_email)
        )
    except Exception as e:
        logger.error(f"Email failed: {str(e)}")
        return False


def _send_email_sync(msg, receiver_email):
    """Synchronous email sending"""
    server = None
    try:
        server = smtplib.SMTP(
            EMAIL_SERVER_CONFIG["smtp_server"],
            EMAIL_SERVER_CONFIG["smtp_port"],
            timeout=10
        )
        server.starttls()
        server.login(EMAIL_SERVER_CONFIG["sender"], EMAIL_SERVER_CONFIG["password"])

        receivers = receiver_email.split(",") if "," in receiver_email else [receiver_email]
        server.sendmail(EMAIL_SERVER_CONFIG["sender"], receivers, msg.as_string())

        logger.success("Email sent successfully")
        return True
    except socket.timeout:
        logger.warning("SMTP connection timeout, skipping email")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Email failed: {str(e)}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception as e:
                logger.warning(f"Error closing SMTP connection: {str(e)}")


# ============================================================================
# Paper Fetching from Multiple Sources
# ============================================================================

def fetch_huggingface_papers(days_lookback=1, max_papers=50, sort='trending'):
    """Fetch papers from HuggingFace

    Args:
        days_lookback: Number of days to look back
        max_papers: Maximum papers to fetch per day
        sort: Sorting method - 'trending' (default) or None (daily papers)
    """
    try:
        sort_label = f"({sort})" if sort else "(daily)"
        logger.info(f"Fetching papers from HuggingFace {sort_label}...")

        # Fetch papers from recent days
        papers = []
        today = datetime.now()

        for day_offset in range(days_lookback):
            target_date = today - timedelta(days=day_offset)
            date_str = target_date.strftime('%Y-%m-%d')

            url = f"https://huggingface.co/api/daily_papers?date={date_str}"
            if sort:
                url += f"&sort={sort}"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                daily_papers = response.json()
                logger.info(f"Found {len(daily_papers)} papers from HuggingFace on {date_str}")

                for paper_entry in daily_papers[:max_papers]:
                    # HuggingFace nests paper data in a 'paper' key
                    paper_data = paper_entry.get('paper', paper_entry)

                    # Extract arXiv ID from the 'id' field
                    arxiv_id = paper_data.get('id', '')
                    if not arxiv_id:
                        continue

                    # Build arXiv URL
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                    # Parse published date
                    published_str = paper_data.get('publishedAt', '')
                    try:
                        published_dt = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                    except:
                        published_dt = target_date

                    papers.append({
                        "title": paper_data.get('title', 'Untitled'),
                        "url": arxiv_url,
                        "pdf_url": pdf_url,
                        "abstract": paper_data.get('summary', ''),
                        "authors": [a.get('name', '') for a in paper_data.get('authors', [])],
                        "published": published_dt,
                        "categories": [],  # HF doesn't provide categories
                        "primary_category": "Unknown",
                        "source": "HuggingFace",
                        "upvotes": paper_entry.get('upvotes', 0),
                        "github_stars": paper_entry.get('githubStars', 0),
                        "arxiv_id": arxiv_id
                    })
            else:
                logger.warning(f"HuggingFace API returned {response.status_code} for {date_str}")

        logger.success(f"Fetched {len(papers)} papers from HuggingFace")
        return papers

    except Exception as e:
        logger.error(f"Failed to fetch HuggingFace papers: {e}")
        return []


def fetch_papers(arxiv_categories):
    """Fetch papers from specified arXiv categories"""
    search_query = " OR ".join([f"cat:{cat}" for cat in arxiv_categories])
    client = Client()
    search = Search(
        query=search_query,
        sort_by=SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending,
        max_results=100
    )

    papers = []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_date = today - timedelta(days=GENERAL_CONFIG["days_lookback"])

    # Adjust if target date is weekend (go back to Friday)
    weekday = target_date.weekday()
    if weekday >= 5:  # Saturday or Sunday
        target_date -= timedelta(days=weekday - 4)

    logger.info(f"Fetching papers from arXiv since {target_date.strftime('%Y-%m-%d')}")

    for result in client.results(search):
        published_dt = result.published.replace(tzinfo=None)
        if target_date <= published_dt:
            # Extract arXiv ID from entry_id
            arxiv_id = result.entry_id.split('/')[-1]

            papers.append({
                "title": result.title,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "published": result.published,
                "categories": list(result.categories),
                "primary_category": result.primary_category,
                "source": "arXiv",
                "arxiv_id": arxiv_id
            })

    logger.success(f"Found {len(papers)} papers from arXiv")
    return papers


def merge_and_deduplicate_papers(paper_lists):
    """Merge papers from multiple sources and deduplicate by arXiv ID"""
    seen_ids = {}
    merged_papers = []

    # Flatten all paper lists
    all_papers = []
    for papers in paper_lists:
        all_papers.extend(papers)

    logger.info(f"Merging {len(all_papers)} papers from all sources...")

    for paper in all_papers:
        arxiv_id = paper.get('arxiv_id', '')

        if not arxiv_id:
            # If no arXiv ID, try to extract from URL
            url = paper.get('url', '')
            if 'arxiv.org' in url:
                arxiv_id = url.split('/')[-1]
            else:
                # Skip non-arXiv papers
                continue

        # Deduplicate by arXiv ID
        if arxiv_id not in seen_ids:
            seen_ids[arxiv_id] = paper
            merged_papers.append(paper)
        else:
            # If duplicate, prefer papers with more metadata or from social sources
            existing = seen_ids[arxiv_id]
            existing_source = existing.get('source', 'Unknown')
            new_source = paper.get('source', 'Unknown')

            # Prefer social sources (HuggingFace) as they indicate popularity
            if new_source == 'HuggingFace' and existing_source == 'arXiv':
                # Replace with social source version (has upvotes, stars, etc.)
                idx = merged_papers.index(existing)
                merged_papers[idx] = paper
                seen_ids[arxiv_id] = paper
                logger.debug(f"Preferring {new_source} version of: {paper['title'][:50]}...")

    logger.success(f"Deduplicated to {len(merged_papers)} unique papers")
    return merged_papers

# ============================================================================
# PDF Download and Text Extraction
# ============================================================================

def download_pdf(url, filename, max_retries=3):
    """Download PDF with retry mechanism"""
    # Ensure URL points to PDF
    if 'arxiv.org' in url and not url.endswith('.pdf'):
        paper_id = url.split('/')[-1]
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    logger.info(f"Downloading: {url}")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)

                # Validate file size
                file_size = os.path.getsize(filename)
                if file_size < 1000:  # Less than 1KB is suspicious
                    logger.warning(f"Downloaded file too small ({file_size} bytes)")
                    continue

                return True
            else:
                logger.error(f"Download failed: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")

        # Exponential backoff
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))

    return False

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num+1}: {str(e)}")
    except Exception as e:
        logger.error(f"PDF parsing failed: {str(e)}")

    return text


def get_paper_text(paper, user_dir):
    """Get paper text (try PDF first, fallback to abstract)"""
    # Try PDF download
    pdf_path = f"{user_dir}/{paper['title'][:100]}.pdf"  # Limit filename length

    if download_pdf(paper['pdf_url'], pdf_path):
        text = extract_text_from_pdf(pdf_path)
        if text and len(text) > 1000:
            # Truncate if too long (to fit LLM context)
            max_length = GENERAL_CONFIG.get("max_text_length", 100000)
            if len(text) > max_length:
                logger.info(f"Truncating text to {max_length} characters")
                text = text[:max_length]
            return text

    # Fallback to abstract if PDF fails
    logger.warning(f"Using abstract as fallback for: {paper['title']}")
    return paper['abstract']

# ============================================================================
# LLM Functions
# ============================================================================

def llm_call(prompt, temperature=0.7):
    """Make an LLM API call and return response with token stats"""
    # Configure OpenAI with custom base URL for DeepSeek compatibility
    openai.api_key = AI_CONFIG["api_key"]
    openai.api_base = AI_CONFIG["base_url"]

    try:
        response = openai.ChatCompletion.create(
            model=AI_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        usage = response['usage']
        token_stats = {
            'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'total_tokens': usage['total_tokens']
        }

        content = response['choices'][0]['message']['content'].strip()
        return content, token_stats

    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        return None, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


def check_interest(abstract, filter_prompt):
    """Check if user is interested in paper based on abstract"""
    prompt = filter_prompt.format(abstract=abstract)
    answer, token_stats = llm_call(prompt, temperature=0.3)

    if not answer:
        return True, token_stats  # Default to interested on error

    # Check for positive/negative keywords
    answer_lower = answer.lower()
    positive = any(kw in answer_lower for kw in ['yes', 'interested', 'relevant'])
    negative = any(kw in answer_lower for kw in ['no', 'not interested', 'irrelevant'])

    is_interested = positive and not negative

    if not positive and not negative:
        logger.warning(f"Unclear answer, defaulting to interested: {answer}")
        is_interested = True

    return is_interested, token_stats


def summarize_paper(text, summary_prompt):
    """Summarize paper using LLM"""
    prompt = summary_prompt.format(text=text)
    summary, token_stats = llm_call(prompt, temperature=0.7)

    if not summary:
        return "Summarization failed.", token_stats

    return summary, token_stats


def rank_papers_by_value(papers, max_papers, filter_prompt, ranking_prompt):
    """Use LLM to intelligently select the most valuable papers

    Args:
        papers: List of paper dictionaries
        max_papers: Maximum number to select
        filter_prompt: User's research interests (used for ranking context)
        ranking_prompt: Template for ranking prompt

    Returns:
        tuple: (selected_papers, token_stats)
    """
    if len(papers) <= max_papers:
        return papers, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    logger.info(f"Ranking {len(papers)} papers to select top {max_papers}...")

    # Build papers list with metadata
    papers_list = []
    for i, paper in enumerate(papers):
        source_badge = f" [Source: {paper.get('source', 'arXiv')}]"
        upvotes = paper.get('upvotes', 0)
        stars = paper.get('github_stars', 0)
        social_info = ""
        if upvotes > 0 or stars > 0:
            social_info = f" [👍 {upvotes} upvotes, ⭐ {stars} stars]"

        papers_list.append(
            f"{i+1}. **{paper['title']}**{source_badge}{social_info}\n"
            f"   Abstract: {paper['abstract'][:500]}...\n"
        )

    # Extract research interests from filter prompt
    interests = filter_prompt.split('{abstract}')[0].strip()

    # Format the ranking prompt
    prompt = ranking_prompt.format(
        max_papers=max_papers,
        interests=interests,
        num_papers=len(papers),
        papers_list=''.join(papers_list)
    )

    response, token_stats = llm_call(prompt, temperature=0.3)

    if not response:
        logger.warning("Ranking failed, using first N papers")
        return papers[:max_papers], token_stats

    # Parse the response to get paper indices
    try:
        # Extract the last line which should contain "SELECTED: 1,2,3,..."
        lines = response.strip().split('\n')
        last_line = lines[-1].strip()

        # Try to find the SELECTED: prefix
        if 'SELECTED:' in last_line.upper():
            # Extract the part after "SELECTED:"
            numbers_part = last_line.split(':', 1)[1].strip()
        else:
            # Fallback: use the last line as-is
            numbers_part = last_line

        # Extract numbers from the line
        numbers = re.findall(r'\d+', numbers_part)
        selected_indices = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < len(papers)]

        # Limit to max_papers and ensure uniqueness
        selected_indices = list(dict.fromkeys(selected_indices))[:max_papers]

        if len(selected_indices) == 0:
            logger.warning("No valid indices found in response")
            logger.warning(f"Last line was: {last_line}")
            return papers[:max_papers], token_stats

        selected_papers = [papers[i] for i in selected_indices]
        logger.success(f"Selected {len(selected_papers)} papers via LLM ranking")

        # Log which papers were selected
        for i, paper in enumerate(selected_papers, 1):
            logger.info(f"  Rank {i}: {paper['title'][:60]}...")

        return selected_papers, token_stats

    except Exception as e:
        logger.error(f"Failed to parse ranking response: {e}")
        logger.warning(f"Response was: {response[:200]}")
        return papers[:max_papers], token_stats

# ============================================================================
# Reporting and Utilities
# ============================================================================

def log_token_cost(user_name, filter_in, filter_out, gen_in, gen_out):
    """Log token usage and cost"""
    total_in = filter_in + gen_in
    total_out = filter_out + gen_out
    total = total_in + total_out

    # Calculate cost (USD)
    price_in = AI_CONFIG.get("price_per_million_input_tokens", 0)
    price_out = AI_CONFIG.get("price_per_million_output_tokens", 0)

    filter_cost = (filter_in * price_in + filter_out * price_out) / 1_000_000
    gen_cost = (gen_in * price_in + gen_out * price_out) / 1_000_000
    total_cost = filter_cost + gen_cost

    logger.info("=" * 80)
    logger.info(f"[{user_name}] Token Usage:")
    logger.info(f"  Filter:  {filter_in:,} in + {filter_out:,} out = {filter_cost:.4f} CNY")
    logger.info(f"  Summary: {gen_in:,} in + {gen_out:,} out = {gen_cost:.4f} CNY")
    logger.info(f"  Total:   {total_in:,} in + {total_out:,} out = {total:,} tokens")
    logger.info(f"  Cost:    {total_cost:.4f} CNY")
    logger.info("=" * 80)

def build_filtered_appendix(filtered_papers):
    """Build appendix for filtered-out papers"""
    if not filtered_papers:
        return ""

    appendix = ["\n\n" + "=" * 80]
    appendix.append("\n## 📋 Appendix: Filtered Papers")
    appendix.append("\nThese papers were filtered out by AI but included for reference:\n")

    for i, paper in enumerate(filtered_papers, 1):
        appendix.append(f"\n### {i}. {paper['title']}\n")
        appendix.append(f"**Authors**: {', '.join(paper['authors'])}\n")
        appendix.append(f"**Published**: {paper['published'].strftime('%Y-%m-%d')}\n")
        appendix.append(f"**Link**: [{paper['url']}]({paper['url']})\n")
        appendix.append(f"**Category**: {paper.get('primary_category', 'Unknown')}\n")
        appendix.append(f"\n**Abstract**: {paper['abstract']}\n")
        appendix.append("\n" + "─" * 80 + "\n")

    return ''.join(appendix)

# ============================================================================
# Main Processing
# ============================================================================

def filter_papers_concurrent(papers, filter_prompt):
    """Filter papers concurrently using interest check"""
    interested, filtered_out = [], []
    filter_in, filter_out = 0, 0

    def check_paper(i, paper):
        logger.info(f"[{i+1}/{len(papers)}] Checking: {paper['title'][:60]}...")
        is_interested, stats = check_interest(paper['abstract'], filter_prompt)
        return is_interested, paper, stats

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_paper, i, p): p for i, p in enumerate(papers)}

        for future in as_completed(futures):
            try:
                is_interested, paper, stats = future.result()
                filter_in += stats['prompt_tokens']
                filter_out += stats['completion_tokens']

                if is_interested:
                    interested.append(paper)
                else:
                    filtered_out.append(paper)
            except Exception as e:
                logger.error(f"Filter error: {e}")
                interested.append(futures[future])  # Keep paper on error

    logger.info(f"Filtered: {len(interested)} interested, {len(filtered_out)} filtered out")
    return interested, filtered_out, filter_in, filter_out


def build_paper_report(paper, summary):
    """Build markdown report for a single paper"""
    source = paper.get('source', 'Unknown')
    source_badge = f"**Source**: {source}"

    # Add social metrics if available
    social_info = ""
    if source == "HuggingFace":
        upvotes = paper.get('upvotes', 0)
        stars = paper.get('github_stars', 0)
        social_info = f" | 👍 {upvotes} upvotes"
        if stars > 0:
            social_info += f" | ⭐ {stars} GitHub stars"

    return f"""
## 📄 {paper['title']}

**Authors**: {', '.join(paper['authors'])}
**Published**: {paper['published'].strftime('%Y-%m-%d')}
**Link**: [{paper['url']}]({paper['url']})
**Category**: {paper.get('primary_category', 'Unknown')}
{source_badge}{social_info}

**Abstract**:
{paper['abstract']}

**Summary**:
{summary}

{'─' * 80}
"""


def process_user(user_config):
    """Process papers for a single user"""
    user_name = user_config["name"]
    user_email = user_config["email"]
    categories = user_config["arxiv_categories"]
    filter_prompt = user_config.get("filter_prompt", DEFAULT_FILTER_PROMPT)
    summary_prompt = user_config.get("summary_prompt", DEFAULT_SUMMARY_PROMPT)

    # Social feed options
    enable_huggingface = user_config.get("enable_huggingface", GENERAL_CONFIG.get("enable_huggingface", True))
    hf_sort = user_config.get("huggingface_sort", GENERAL_CONFIG.get("huggingface_sort", "trending"))

    # Ranking prompt (can be customized per user)
    ranking_prompt = user_config.get("ranking_prompt", DEFAULT_RANKING_PROMPT)

    logger.info(f"Processing user: {user_name}")

    # Setup
    user_dir = f"temp/{user_name.replace(' ', '_')}"
    os.makedirs(user_dir, exist_ok=True)

    # Fetch papers from multiple sources
    paper_sources = []

    # 1. Fetch from arXiv
    arxiv_papers = fetch_papers(categories)
    paper_sources.append(arxiv_papers)

    # 2. Fetch from HuggingFace (if enabled)
    if enable_huggingface:
        hf_papers = fetch_huggingface_papers(
            days_lookback=GENERAL_CONFIG.get("days_lookback", 1),
            max_papers=50,
            sort=hf_sort
        )
        paper_sources.append(hf_papers)

    # Merge and deduplicate all sources
    papers = merge_and_deduplicate_papers(paper_sources)

    if not papers:
        logger.info(f"No papers found for {user_name}")
        return

    # Filter papers (if filter prompt provided)
    filter_in, filter_out = 0, 0
    filtered_papers = []

    if filter_prompt != DEFAULT_FILTER_PROMPT:
        papers, filtered_papers, filter_in, filter_out = filter_papers_concurrent(
            papers, filter_prompt
        )

    # Apply max papers limit with intelligent ranking
    max_papers = GENERAL_CONFIG.get("max_papers_per_user")
    rank_in, rank_out = 0, 0

    if max_papers and max_papers > 0 and len(papers) > max_papers:
        logger.info(f"Found {len(papers)} papers, need to select top {max_papers}")
        papers, rank_stats = rank_papers_by_value(papers, max_papers, filter_prompt, ranking_prompt)
        rank_in = rank_stats['prompt_tokens']
        rank_out = rank_stats['completion_tokens']
    elif max_papers and max_papers > 0:
        logger.info(f"Found {len(papers)} papers (≤ max {max_papers}), processing all")

    # Summarize papers
    report_sections = []
    gen_in, gen_out = 0, 0

    for i, paper in enumerate(papers, 1):
        try:
            logger.info(f"[{i}/{len(papers)}] Summarizing: {paper['title'][:60]}...")
            text = get_paper_text(paper, user_dir)
            summary, stats = summarize_paper(text, summary_prompt)

            gen_in += stats['prompt_tokens']
            gen_out += stats['completion_tokens']

            report_sections.append(build_paper_report(paper, summary))
        except Exception as e:
            logger.error(f"Error processing paper: {e}")

    # Log costs (including ranking tokens)
    total_filter_in = filter_in + rank_in
    total_filter_out = filter_out + rank_out
    log_token_cost(user_name, total_filter_in, total_filter_out, gen_in, gen_out)

    # Build and send report
    if report_sections:
        full_report = '\n'.join(report_sections)

        # Add filtered papers appendix
        if filtered_papers:
            full_report += build_filtered_appendix(filtered_papers)

        # Send email
        asyncio.run(send_email(
            f"Daily Papers - {user_name}",
            full_report,
            user_email
        ))

        # Save to file
        report_file = f"{user_dir}/report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)

        logger.success(f"Report sent and saved to {report_file}")

# ============================================================================
# Scheduler
# ============================================================================

def daily_job():
    """Daily task: process papers for all configured users"""
    os.makedirs('temp', exist_ok=True)
    logger.info(f"Starting daily job for {len(USERS_CONFIG)} users")

    for user_config in USERS_CONFIG:
        try:
            process_user(user_config)
        except Exception as e:
            logger.error(f"Error processing {user_config['name']}: {e}")

    logger.success("All users processed")


def run_scheduler():
    """Run scheduler for daily execution"""
    scheduler = BlockingScheduler()
    scheduler.add_job(
        daily_job,
        trigger=CronTrigger(hour=9, minute=0),  # Run daily at 9 AM
        id='daily_arxiv_job',
        name='Daily ArXiv paper collection and summary'
    )

    logger.info("Scheduler configured to run daily at 9:00 AM")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    # Configure logger
    logger.add(
        "arxiv_pusher.log",
        rotation="10 MB",
        level="INFO",
        encoding="utf-8"
    )

    # Run once immediately (comment out to disable)
    daily_job()

    # Start scheduler (uncomment to enable)
    # run_scheduler()