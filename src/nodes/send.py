"""Send node: SMTP delivery to multiple recipients.

Configure via env vars (set as repo secrets):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, MAIL_RECIPIENTS

MAIL_RECIPIENTS is a comma-separated list.
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

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = ", ".join(recipients)
    msg.set_content("This briefing is best viewed as HTML.")
    msg.add_alternative(body, subtype="html")

    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]

    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return {"errors": []}
    except Exception as e:
        return {"errors": [f"SMTP send failed: {e}"]}
