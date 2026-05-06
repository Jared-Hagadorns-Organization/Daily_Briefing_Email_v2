"""One-time local script to obtain a Google OAuth refresh token.

Run this once on your machine:
    python scripts/get_refresh_token.py path/to/client_secret.json

It opens a browser, you grant access, and it prints the refresh_token.
Copy that into the GOOGLE_REFRESH_TOKEN repo secret.
"""
import sys
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def main(client_secret_path: str) -> None:
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    print("\n--- Copy these into GitHub repo secrets ---")
    print(f"GOOGLE_CLIENT_ID={creds.client_id}")
    print(f"GOOGLE_CLIENT_SECRET={creds.client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={creds.refresh_token}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/get_refresh_token.py <client_secret.json>")
        sys.exit(1)
    main(sys.argv[1])
