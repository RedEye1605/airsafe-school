#!/usr/bin/env python3
"""
Discover SPKU API endpoints by monitoring network requests.
This script opens the SPKU portal in Playwright and logs all JSON responses.
"""
from playwright.sync_api import sync_playwright

def main():
    print("="*60)
    print("SPKU API Discovery Tool")
    print("="*60)
    print("\nStarting browser...")
    print("The browser will open and load the SPKU station page.")
    print("Network requests will be logged to the terminal.")
    print("\nInstructions:")
    print("1. Wait for the page to fully load")
    print("2. Click on different stations to explore")
    print("3. Watch for JSON/XHR requests in the terminal")
    print("4. Copy the URL of requests that return useful data")
    print("5. Press Enter to close the browser when done")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def log_response(resp):
            """Log JSON responses from the page."""
            ctype = resp.headers.get("content-type", "")
            url = resp.url
            if any(x in ctype.lower() for x in ["json", "javascript", "text/plain"]):
                print(f"\n[{resp.status}] {ctype}")
                print(f"URL: {url}")
                if resp.status == 200:
                    print(f"Size: {len(resp.body)} bytes")

        page.on("response", log_response)
        print("\nNavigating to SPKU station location page...")
        page.goto("https://udara.jakarta.go.id/lokasi_stasiun", wait_until="networkidle", timeout=30000)

        print("\n" + "="*60)
        print("Browser is open. Interact with the page manually.")
        print("Look for requests that return station data, trends, or history.")
        print("="*60 + "\n")

        input("Press Enter when done capturing requests...")
        browser.close()

    print("\n" + "="*60)
    print("Session closed.")
    print("="*60)

if __name__ == "__main__":
    main()
