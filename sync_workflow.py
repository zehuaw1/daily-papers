#!/usr/bin/env python3
"""
Script to sync config.py settings to GitHub Actions workflow file.
This updates the workflow's config.py generation section while keeping secrets as GitHub secrets.
"""

import re
import sys


def escape_for_heredoc(text):
    """Escape text for use in a bash heredoc."""
    # Escape dollar signs to prevent variable expansion
    text = text.replace('$', '\\$')
    # Escape backticks
    text = text.replace('`', '\\`')
    return text


def indent_multiline(text, spaces=8):
    """Add indentation to each line of a multi-line string for YAML formatting.

    Args:
        text: The text to indent
        spaces: Number of spaces to add (default 8 for YAML heredoc content)

    Returns:
        Indented text with proper spacing for YAML structure
    """
    if not text:
        return text

    indent = ' ' * spaces
    lines = text.split('\n')
    # Add indentation to all lines, preserving empty lines
    indented_lines = [indent + line if line.strip() else line for line in lines]
    return '\n'.join(indented_lines)


def extract_python_value(content, var_name):
    """Extract a Python variable value from config.py content."""
    # For dict values
    dict_pattern = rf'{var_name}\s*=\s*\{{([^}}]+(?:\{{[^}}]*\}}[^}}]*)*)\}}'
    match = re.search(dict_pattern, content, re.DOTALL)
    if match:
        return '{' + match.group(1) + '}'

    # For string values (triple-quoted)
    str_pattern = rf'{var_name}\s*=\s*"""(.*?)"""'
    match = re.search(str_pattern, content, re.DOTALL)
    if match:
        return '"""' + match.group(1) + '"""'

    return None


def parse_dict_from_string(dict_str):
    """Parse a dictionary string into a dict object."""
    # Remove comments and extract key-value pairs
    result = {}

    # Extract model
    model_match = re.search(r'"model":\s*"([^"]+)"', dict_str)
    if model_match:
        result['model'] = model_match.group(1)

    # Extract base_url
    base_url_match = re.search(r'"base_url":\s*"([^"]+)"', dict_str)
    if base_url_match:
        result['base_url'] = base_url_match.group(1)

    # Extract pricing
    input_price_match = re.search(r'"price_per_million_input_tokens":\s*([\d.]+)', dict_str)
    if input_price_match:
        result['price_per_million_input_tokens'] = input_price_match.group(1)

    output_price_match = re.search(r'"price_per_million_output_tokens":\s*([\d.]+)', dict_str)
    if output_price_match:
        result['price_per_million_output_tokens'] = output_price_match.group(1)

    # Extract SMTP settings
    smtp_server_match = re.search(r'"smtp_server":\s*"([^"]+)"', dict_str)
    if smtp_server_match:
        result['smtp_server'] = smtp_server_match.group(1)

    smtp_port_match = re.search(r'"smtp_port":\s*(\d+)', dict_str)
    if smtp_port_match:
        result['smtp_port'] = smtp_port_match.group(1)

    use_tls_match = re.search(r'"use_tls":\s*(True|False)', dict_str)
    if use_tls_match:
        result['use_tls'] = use_tls_match.group(1)

    # Extract general config
    days_match = re.search(r'"days_lookback":\s*(\d+)', dict_str)
    if days_match:
        result['days_lookback'] = days_match.group(1)

    max_papers_match = re.search(r'"max_papers_per_user":\s*(\d+|None)', dict_str)
    if max_papers_match:
        result['max_papers_per_user'] = max_papers_match.group(1)

    max_text_match = re.search(r'"max_text_length":\s*(\d+)', dict_str)
    if max_text_match:
        result['max_text_length'] = max_text_match.group(1)

    enable_hf_match = re.search(r'"enable_huggingface":\s*(True|False)', dict_str)
    if enable_hf_match:
        result['enable_huggingface'] = enable_hf_match.group(1)

    hf_sort_match = re.search(r'"huggingface_sort":\s*(None|"[^"]*")', dict_str)
    if hf_sort_match:
        result['huggingface_sort'] = hf_sort_match.group(1)

    return result


def extract_arxiv_categories(content):
    """Extract arxiv_categories from USERS_CONFIG."""
    pattern = r'"arxiv_categories":\s*\[([^\]]+)\]'
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return '"cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "cs.RO", "cs.MA", "cs.GR"'


def main():
    # Read current config.py
    try:
        with open('config.py', 'r') as f:
            config_content = f.read()
    except FileNotFoundError:
        print("❌ Error: config.py not found")
        sys.exit(1)

    # Extract values
    ai_config_str = extract_python_value(config_content, 'AI_CONFIG')
    email_config_str = extract_python_value(config_content, 'EMAIL_SERVER_CONFIG')
    general_config_str = extract_python_value(config_content, 'GENERAL_CONFIG')
    filter_prompt = extract_python_value(config_content, 'DEFAULT_FILTER_PROMPT')
    summary_prompt = extract_python_value(config_content, 'DEFAULT_SUMMARY_PROMPT')
    ranking_prompt = extract_python_value(config_content, 'DEFAULT_RANKING_PROMPT')
    arxiv_categories = extract_arxiv_categories(config_content)

    # Parse configs
    ai_config = parse_dict_from_string(ai_config_str) if ai_config_str else {}
    email_config = parse_dict_from_string(email_config_str) if email_config_str else {}
    general_config = parse_dict_from_string(general_config_str) if general_config_str else {}

    # Clean up prompts (remove triple quotes and prepare for YAML indentation)
    if filter_prompt:
        filter_prompt = filter_prompt.strip('"""').strip()
        # Indent continuation lines for proper YAML structure (first line already on same line as """)
        filter_prompt_lines = filter_prompt.split('\n')
        if len(filter_prompt_lines) > 1:
            # First line stays on the same line as the opening """
            # All subsequent lines need 8 spaces for YAML indentation
            filter_prompt = filter_prompt_lines[0] + '\n' + '\n'.join('        ' + line for line in filter_prompt_lines[1:])
    if summary_prompt:
        summary_prompt = summary_prompt.strip('"""').strip()
        summary_prompt_lines = summary_prompt.split('\n')
        if len(summary_prompt_lines) > 1:
            summary_prompt = summary_prompt_lines[0] + '\n' + '\n'.join('        ' + line for line in summary_prompt_lines[1:])
    if ranking_prompt:
        ranking_prompt = ranking_prompt.strip('"""').strip()
        ranking_prompt_lines = ranking_prompt.split('\n')
        if len(ranking_prompt_lines) > 1:
            ranking_prompt = ranking_prompt_lines[0] + '\n' + '\n'.join('        ' + line for line in ranking_prompt_lines[1:])

    # Read current workflow file
    workflow_path = '.github/workflows/daily-arxiv.yml'
    try:
        with open(workflow_path, 'r') as f:
            workflow_content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: {workflow_path} not found")
        sys.exit(1)

    # Generate new config.py section for the workflow
    config_template = f'''        cat > config.py << 'EOF'
        # ==============================================================================
        # ArXiv Pusher Configuration (Generated by GitHub Actions)
        # ==============================================================================

        # ==============================================================================
        # AI Model Configuration
        # ==============================================================================

        # Google Gemini API (OpenAI-compatible endpoint)
        AI_CONFIG = {{
            "api_key": "${{GOOGLE_GEMINI_API_KEY}}",
            "base_url": "{ai_config.get('base_url', 'https://generativelanguage.googleapis.com/v1beta/openai/')}",
            "model": "{ai_config.get('model', 'gemini-2.5-flash')}",

            # Pricing (for cost tracking) - Gemini 2.5 Flash pricing (USD)
            "price_per_million_input_tokens": {ai_config.get('price_per_million_input_tokens', '0.30')},  # USD
            "price_per_million_output_tokens": {ai_config.get('price_per_million_output_tokens', '2.50')},  # USD
        }}

        # DeepSeek API (commented out)
        # AI_CONFIG = {{
        #     "api_key": "${{DEEPSEEK_API_KEY}}",
        #     "base_url": "https://api.deepseek.com",
        #     "model": "deepseek-chat",
        #
        #     # Pricing (for cost tracking) - DeepSeek V3 pricing
        #     "price_per_million_input_tokens": 2.00,  # CNY
        #     "price_per_million_output_tokens": 3.00,  # CNY
        # }}

        # ==============================================================================
        # Email Server Configuration
        # ==============================================================================
        EMAIL_SERVER_CONFIG = {{
            "sender": "${{EMAIL_SENDER}}",
            "password": "${{EMAIL_PASSWORD}}",
            "smtp_server": "{email_config.get('smtp_server', 'smtp.gmail.com')}",
            "smtp_port": {email_config.get('smtp_port', '587')},
            "use_tls": {email_config.get('use_tls', 'True')},
        }}

        # ==============================================================================
        # General Configuration
        # ==============================================================================
        GENERAL_CONFIG = {{
            "days_lookback": {general_config.get('days_lookback', '1')},
            "max_papers_per_user": {general_config.get('max_papers_per_user', '10')},
            "max_text_length": {general_config.get('max_text_length', '100000')},

            # Social Feed Integration (FREE)
            "enable_huggingface": {general_config.get('enable_huggingface', 'True')},
            "huggingface_sort": {general_config.get('huggingface_sort', 'None')},  # "trending" or None for daily papers
        }}

        # ==============================================================================
        # Default Prompts
        # ==============================================================================

        DEFAULT_FILTER_PROMPT = """{filter_prompt if filter_prompt else 'You are an academic expert...'}"""

        DEFAULT_SUMMARY_PROMPT = """{summary_prompt if summary_prompt else 'You are an academic paper assistant...'}"""

        DEFAULT_RANKING_PROMPT = """{ranking_prompt if ranking_prompt else 'You are an AI research assistant...'}"""

        # ==============================================================================
        # User Configurations
        # ==============================================================================
        USERS_CONFIG = [
            {{
                "name": "${{RECIPIENT_NAME}}",
                "email": "${{RECIPIENT_EMAIL}}",
                "arxiv_categories": [{arxiv_categories}],
            }},
        ]
        EOF'''

    # Find and replace the config generation section in workflow
    # Pattern to match from "cat > config.py" to "EOF" and the echo line
    pattern = r'(      run: \|\n)(        cat > config\.py << \'?EOF\'?.*?        EOF\n        echo "✅ config\.py generated successfully"\n)'

    new_section = config_template + '\n        echo "✅ config.py generated successfully"\n'

    replacement = r'\1' + new_section

    new_workflow_content = re.sub(pattern, replacement, workflow_content, flags=re.DOTALL)

    if new_workflow_content == workflow_content:
        print("⚠️  Warning: No changes detected. Pattern might not match.")
        print("    Workflow file may already be up to date or script needs adjustment.")
        sys.exit(1)

    # Write updated workflow file
    with open(workflow_path, 'w') as f:
        f.write(new_workflow_content)

    print("✅ Successfully updated .github/workflows/daily-arxiv.yml")
    print("\nUpdated sections:")
    print(f"  - AI model: {ai_config.get('model', 'N/A')}")
    print(f"  - Base URL: {ai_config.get('base_url', 'N/A')}")
    print(f"  - SMTP server: {email_config.get('smtp_server', 'N/A')}")
    print(f"  - Max papers: {general_config.get('max_papers_per_user', 'N/A')}")
    print(f"  - HuggingFace enabled: {general_config.get('enable_huggingface', 'N/A')}")
    print("\n⚠️  Secrets remain as GitHub secrets references (not exposed)")


if __name__ == '__main__':
    main()
