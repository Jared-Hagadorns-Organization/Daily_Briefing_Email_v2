"""Send node: SMTP delivery via Gmail.

Configure via env vars (set as repo secrets):
  GMAIL_ADDRESS, GMAIL_PASSWORD, MAIL_RECIPIENTS

MAIL_RECIPIENTS is a comma-separated list.
GMAIL_PASSWORD should be a Gmail App Password (not your account password).
"""
from __future__ import annotations
import os
import smtplib
from email.message import EmailMessage


def send_node(state: dict) -> dict:
    body = state.get("email_body_html") or "<p>(empty body)</p>"
    subject = state.get("email_subject") or "Daily Briefing"
    recipients = state.get("recipients") or []

    if not recipients:
        return {"errors": ["No recipients configured"]}

    gmail_address = os.environ["GMAIL_ADDRESS"]
    gmail_password = os.environ["GMAIL_PASSWORD"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = ", ".join(recipients)
    msg.set_content("This briefing is best viewed as HTML.")
    msg.add_alternative(body, subtype="html")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(gmail_address, gmail_password)
            s.send_message(msg)
        return {"errors": []}
    except Exception as e:
        return {"errors": [f"SMTP send failed: {e}"]}
