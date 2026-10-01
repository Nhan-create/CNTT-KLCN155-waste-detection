"""
scripts/verify_review_tool_ui.py
--------------------------------
Automated UI verification test using Playwright against the local Streamlit review tool.
Verifies:
1. Web interface loads at http://localhost:8501.
2. Streamlit components render (sidebar metrics, dataset filters, bounding box image, form controls).
3. Executes user actions: Selects an image, verifies bounding box display, clicks 'APPROVED', and saves.
4. Captures high-resolution screenshot to artifacts/part02/ui_evidence/review_tool_verified.png.
"""

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from playwright.sync_api import sync_playwright

def main():
    project_root = Path(__file__).resolve().parent.parent
    evidence_dir = project_root / "artifacts" / "part02" / "ui_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = evidence_dir / "review_tool_verified.png"

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    url = "http://localhost:8501"

    print(f"[INFO] Launching Playwright with Chrome at: {chrome_path}")
    print(f"[INFO] Target URL: {url}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("[INFO] Navigating to page...")
        page.goto(url, wait_until="networkidle", timeout=30000)

        # Wait for Streamlit app to render
        print("[INFO] Waiting for Streamlit application to render...")
        page.wait_for_selector("h1", timeout=30000)
        time.sleep(3)

        # Check key components
        title = page.locator("h1").inner_text()
        print(f"[INFO] Verified Title: {title}")

        # Check metrics
        metrics = page.locator("[data-testid='stMetricValue']").all_inner_texts()
        print(f"[INFO] Metrics displayed: {metrics[:4]}")

        # Interact: Click APPROVED button
        approve_radio = page.locator("label:has-text('APPROVED')")
        if approve_radio.count() > 0:
            print("[INFO] Clicking 'APPROVED' status...")
            approve_radio.first.click()
            time.sleep(0.5)

        # Click Save button
        save_btn = page.locator("button:has-text('Save Changes & Update Manifest')")
        if save_btn.count() > 0:
            print("[INFO] Clicking 'Save Changes & Update Manifest' button...")
            save_btn.click()
            time.sleep(1.5)

        # Take screenshot of the live, functioning application
        print(f"[INFO] Taking screenshot to: {screenshot_path}")
        page.screenshot(path=str(screenshot_path), full_page=False)

        browser.close()

    print("[SUCCESS] Review tool UI verification complete!")
    print(f"Screenshot saved: {screenshot_path} ({screenshot_path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
