import os
import asyncio
import logging
from playwright.async_api import async_playwright
from datetime import datetime

logger = logging.getLogger(__name__)

async def capture_website_screenshot(url: str, output_dir: str = "artifacts") -> str:
    """
    Captures a full-page screenshot of a website given its URL.
    Returns the file path to the saved screenshot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_url = url.replace("https://", "").replace("http://", "").replace("/", "_").split("?")[0][:30]
    filename = f"{safe_url}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    logger.info(f"Starting website screenshot capture for URL: {url}")
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            
            logger.info(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait a bit for animations/lazy loading to settle
            await asyncio.sleep(2)
            
            # Capture full page screenshot
            await page.screenshot(path=filepath, full_page=True, type="png")
            logger.info(f"Successfully captured screenshot of {url} as {filepath}")
            
            await browser.close()
            return filepath
            
    except Exception as e:
        logger.error(f"Error in capture_website_screenshot: {e}")
        raise e
