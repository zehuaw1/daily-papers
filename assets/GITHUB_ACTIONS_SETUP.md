# GitHub Actions Setup Guide

This guide will help you set up ArXiv Pusher to run automatically on GitHub Actions for **completely free**.

## Benefits

- ✅ **Free**: 2000 minutes/month for private repos, unlimited for public repos
- ✅ **Automated**: Runs daily at your chosen time
- ✅ **No server needed**: GitHub handles everything
- ✅ **Secure**: Credentials stored as encrypted secrets
- ✅ **Logs**: Automatic logging and debugging

## Setup Steps

### 1. Push Your Code to GitHub

If you haven't already:

```bash
git add .
git commit -m "Add GitHub Actions workflow"
git push origin main
```

### 2. Configure GitHub Secrets

Go to your repository on GitHub:

**Settings → Secrets and variables → Actions → New repository secret**

Add the following secrets:

| Secret Name | Description | Example Value |
|------------|-------------|---------------|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key | `sk-70a6d8d368cb...` |
| `EMAIL_SENDER` | Gmail address to send from | `youremail@gmail.com` |
| `EMAIL_PASSWORD` | **Gmail App Password** (not regular password!) | `xxxx xxxx xxxx xxxx` |
| `RECIPIENT_EMAIL` | Email to receive reports | `recipient@example.com` |
| `RECIPIENT_NAME` | Recipient name | `Your Name` |

### 3. Get Gmail App Password

**IMPORTANT**: You cannot use your regular Gmail password. You need an **App-Specific Password**.

#### Steps:

1. **Enable 2-Factor Authentication**:
   - Go to: https://myaccount.google.com/security
   - Enable "2-Step Verification"

2. **Generate App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Name it: "ArXiv Pusher"
   - Click "Generate"
   - Copy the 16-character password (e.g., `xxxx xxxx xxxx xxxx`)

3. **Add to GitHub Secrets**:
   - Use this 16-character password as `EMAIL_PASSWORD`

### 4. Configure Schedule

Edit `.github/workflows/daily-arxiv.yml` to set your preferred time:

```yaml
on:
  schedule:
    # Run at 9:00 AM UTC every day
    - cron: '0 9 * * *'
```

**Time Zone Conversion**:
- GitHub Actions uses **UTC time**
- Use https://crontab.guru/ to help with cron syntax
- Examples:
  - `0 9 * * *` = 9:00 AM UTC daily
  - `0 13 * * 1-5` = 1:00 PM UTC, Monday-Friday only
  - `0 1 * * *` = 1:00 AM UTC daily

**Convert your local time to UTC**:
- **EST** (New York): Add 5 hours (9 AM EST = 2 PM UTC = `0 14 * * *`)
- **PST** (Los Angeles): Add 8 hours (9 AM PST = 5 PM UTC = `0 17 * * *`)
- **CST** (Chicago): Add 6 hours (9 AM CST = 3 PM UTC = `0 15 * * *`)

### 5. Test the Workflow

You can manually trigger the workflow to test it:

1. Go to: **Actions** tab in your GitHub repo
2. Click on "Daily ArXiv Paper Digest"
3. Click "Run workflow" → "Run workflow"
4. Watch the logs to see if it succeeds

### 6. Monitor Execution

**Check runs**:
- Go to **Actions** tab
- Click on any run to see detailed logs

**Check emails**:
- Wait for the scheduled time
- Check your inbox (and spam folder!)

## Customization

### Change arXiv Categories

Edit the workflow file to modify categories:

```yaml
"arxiv_categories": ["cs.LG", "cs.AI", "cs.CL"],
```

### Change Number of Papers

Edit the workflow file:

```yaml
"max_papers_per_user": 5,  # Instead of 10
```

### Enable HuggingFace Trending

Edit the workflow file:

```yaml
"enable_huggingface": True,
"huggingface_sort": "trending",  # or None for daily
```

### Add Multiple Recipients

Currently, the workflow supports one recipient. To add more:

1. Create additional secrets: `RECIPIENT_EMAIL_2`, `RECIPIENT_NAME_2`, etc.
2. Modify the workflow's `USERS_CONFIG` section to add more users

## Troubleshooting

### Email not received

1. **Check spam folder**: Gmail may filter automated emails
2. **Check GitHub Actions logs**: Go to Actions tab → click on the run → check for errors
3. **Verify secrets**: Make sure all 5 secrets are set correctly
4. **Test email locally**: Use `test_email.py` to verify SMTP settings

### Workflow not running

1. **Check schedule**: Make sure cron syntax is correct
2. **Check GitHub Actions**: Must be enabled in repo settings
3. **Manual trigger**: Try running manually first (Actions → Run workflow)

### API rate limits

- **DeepSeek**: Generous free tier, unlikely to hit limits
- **arXiv**: No authentication needed, rarely rate-limited
- **HuggingFace**: Free public API

### Cost concerns

- **GitHub Actions**: Free for public repos, 2000 min/month for private
- **DeepSeek API**: ~$0.50/day for 10 papers (estimate)
- **Monthly total**: ~$15/month for daily runs with 10 papers

## Security Notes

### What's Secure:
- ✅ All API keys and passwords stored as encrypted GitHub Secrets
- ✅ Secrets never appear in logs
- ✅ `config.py` generated at runtime and never committed
- ✅ Temporary files cleaned up after each run

### Best Practices:
- 🔒 Use App-Specific Password for Gmail (not your main password)
- 🔒 Never commit `config.py` (already in `.gitignore`)
- 🔒 Use private repo if you want extra privacy
- 🔒 Rotate API keys periodically

## Advanced: Run on Multiple Schedules

You can run different configurations at different times:

```yaml
on:
  schedule:
    # Morning digest - trending papers
    - cron: '0 9 * * *'
    # Evening digest - all new papers
    - cron: '0 21 * * *'
```

Then use different prompts or categories based on the time.

## Support

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Cron Syntax Help**: https://crontab.guru/
- **Gmail App Passwords**: https://support.google.com/accounts/answer/185833

---

**Ready to go?** Once you've set up the secrets, the workflow will run automatically on schedule!
