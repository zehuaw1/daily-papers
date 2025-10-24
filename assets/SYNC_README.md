# Workflow Sync Script

## Purpose
The `sync_workflow.py` script automatically updates `.github/workflows/daily-arxiv.yml` based on your local `config.py`, keeping secrets secure as GitHub secrets references.

## Usage

```bash
python3 sync_workflow.py
```

## What Gets Synced

The script syncs the following from `config.py` to the workflow:

### ✅ Synced Configuration
- **AI Model Settings**: model name, base_url, pricing
- **Email Settings**: smtp_server, smtp_port, use_tls
- **General Settings**: days_lookback, max_papers_per_user, max_text_length, HuggingFace options
- **Prompts**: DEFAULT_FILTER_PROMPT, DEFAULT_SUMMARY_PROMPT, DEFAULT_RANKING_PROMPT
- **ArXiv Categories**: from USERS_CONFIG

### 🔒 Protected Secrets
These remain as GitHub secrets references (not exposed):
- `GOOGLE_GEMINI_API_KEY`
- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `RECIPIENT_EMAIL`
- `RECIPIENT_NAME`

## Workflow

1. Edit your local `config.py` with your preferences
2. Run `python3 sync_workflow.py`
3. Review changes: `git diff .github/workflows/daily-arxiv.yml`
4. Commit and push if satisfied

## Example Output

```
✅ Successfully updated .github/workflows/daily-arxiv.yml

Updated sections:
  - AI model: gemini-2.5-flash
  - Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
  - SMTP server: smtp.gmail.com
  - Max papers: 10
  - HuggingFace enabled: True

⚠️  Secrets remain as GitHub secrets references (not exposed)
```

## Important Notes

- Always review the diff before committing
- The script preserves secret references in the workflow
- Do NOT commit `config.py` with actual API keys/passwords to the repo
- Consider adding `config.py` to `.gitignore` if it contains secrets
