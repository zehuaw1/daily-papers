#!/usr/bin/env python3
"""
Simple email testing script for ArXiv Pusher
Tests SMTP connection without costly LLM calls
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from loguru import logger
from config import EMAIL_SERVER_CONFIG, USERS_CONFIG

def send_test_email(recipient_email, recipient_name):
    """Send a simple test email"""

    sender = EMAIL_SERVER_CONFIG["sender"]
    password = EMAIL_SERVER_CONFIG["password"]
    smtp_server = EMAIL_SERVER_CONFIG["smtp_server"]
    smtp_port = EMAIL_SERVER_CONFIG["smtp_port"]
    use_tls = EMAIL_SERVER_CONFIG["use_tls"]

    # Create message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "ArXiv Pusher - Test Email"
    msg["From"] = sender
    msg["To"] = recipient_email

    # Create plain text and HTML versions
    text_content = f"""
Hello {recipient_name},

This is a test email from ArXiv Pusher.

If you're seeing this, your email configuration is working correctly!

Best regards,
ArXiv Pusher
"""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background-color: #f9f9f9; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ArXiv Pusher - Test Email</h1>
        </div>
        <div class="content">
            <p>Hello <strong>{recipient_name}</strong>,</p>
            <p>This is a test email from ArXiv Pusher.</p>
            <p>If you're seeing this, your email configuration is working correctly! ✅</p>
        </div>
        <div class="footer">
            <p>Best regards,<br>ArXiv Pusher</p>
        </div>
    </div>
</body>
</html>
"""

    # Attach both versions
    part1 = MIMEText(text_content, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)

    # Send email
    try:
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")

        if use_tls:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        else:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        logger.info(f"Logging in as: {sender}")
        server.login(sender, password)

        logger.info(f"Sending email to: {recipient_email}")
        server.sendmail(sender, recipient_email, msg.as_string())
        server.quit()

        logger.success(f"✅ Test email sent successfully to {recipient_email}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("Check your email and password in config.py")
        return False

    except smtplib.SMTPException as e:
        logger.error(f"❌ SMTP error: {e}")
        return False

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def main():
    """Test email for all configured users"""
    logger.info("=" * 60)
    logger.info("ArXiv Pusher - Email Configuration Test")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"SMTP Server: {EMAIL_SERVER_CONFIG['smtp_server']}:{EMAIL_SERVER_CONFIG['smtp_port']}")
    logger.info(f"Sender: {EMAIL_SERVER_CONFIG['sender']}")
    logger.info(f"TLS: {EMAIL_SERVER_CONFIG['use_tls']}")
    logger.info("")

    success_count = 0
    for user in USERS_CONFIG:
        logger.info(f"Testing email for: {user['name']} ({user['email']})")
        if send_test_email(user['email'], user['name']):
            success_count += 1
        logger.info("")

    logger.info("=" * 60)
    logger.info(f"Results: {success_count}/{len(USERS_CONFIG)} emails sent successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
